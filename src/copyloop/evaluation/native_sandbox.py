"""Restrict EvalPlus candidate workers without namespace creation (Linux x86-64)."""
import ctypes
import errno
import os
from pathlib import Path
import platform
import resource
import socket
import sys
import tempfile
import time

_libc = ctypes.CDLL(None, use_errno=True)


class _Ruleset(ctypes.Structure):
    _fields_ = [('handled_access_fs', ctypes.c_uint64)]


class _PathRule(ctypes.Structure):
    _pack_ = 1
    _fields_ = [('allowed_access', ctypes.c_uint64), ('parent_fd', ctypes.c_int)]


class _ArgCompare(ctypes.Structure):
    _fields_ = [('arg', ctypes.c_uint), ('op', ctypes.c_uint),
                ('datum_a', ctypes.c_uint64), ('datum_b', ctypes.c_uint64)]


def restrict(work):
    if platform.machine() != 'x86_64':
        raise RuntimeError('Native scoring sandbox requires Linux x86-64')
    abi = _libc.syscall(444, 0, 0, 1)
    if abi < 1:
        raise OSError(ctypes.get_errno(), 'Landlock unavailable')
    # ABI 1 filesystem operations, plus TRUNCATE when the kernel supports ABI 3.
    handled = (1 << 13) - 1
    if abi >= 3:
        handled |= 1 << 14
    attrs = _Ruleset(handled)
    ruleset = _libc.syscall(444, ctypes.byref(attrs), ctypes.sizeof(attrs), 0)
    if ruleset < 0:
        raise OSError(ctypes.get_errno(), 'Landlock ruleset creation failed')
    try:
        paths = [(Path(sys.prefix), 1 | 4 | 8),
                 (Path('/usr/lib'), 1 | 4 | 8),
                 (Path('/lib'), 1 | 4 | 8),
                 (Path('/lib64'), 1 | 4 | 8),
                 (Path('/usr/share/zoneinfo'), 4 | 8),
                 (Path('/etc/ld.so.cache'), 4),
                 (Path('/proc/meminfo'), 4),
                 (Path('/dev/urandom'), 4),
                 (Path('/dev/null'), 2 | 4),
                 (Path(work), handled & ~((1 << 6) | (1 << 9) | (1 << 10) | (1 << 11)))]
        for path, access in paths:
            if not path.exists():
                continue
            fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
            try:
                if not path.is_dir():
                    access &= 1 | 2 | 4 | (1 << 14)
                rule = _PathRule(access, fd)
                if _libc.syscall(445, ruleset, 1, ctypes.byref(rule), 0):
                    raise OSError(ctypes.get_errno(), 'Landlock path rule failed')
            finally:
                os.close(fd)
        if _libc.prctl(38, 1, 0, 0, 0):  # PR_SET_NO_NEW_PRIVS
            raise OSError(ctypes.get_errno(), 'no_new_privs failed')
        if _libc.syscall(446, ruleset, 0):
            raise OSError(ctypes.get_errno(), 'Landlock restriction failed')
    finally:
        os.close(ruleset)

    seccomp = ctypes.CDLL('libseccomp.so.2', use_errno=True)
    seccomp.seccomp_init.argtypes = [ctypes.c_uint32]
    seccomp.seccomp_init.restype = ctypes.c_void_p
    seccomp.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    seccomp.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
    seccomp.seccomp_rule_add_array.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint, ctypes.POINTER(_ArgCompare)]
    seccomp.seccomp_load.argtypes = [ctypes.c_void_p]
    seccomp.seccomp_release.argtypes = [ctypes.c_void_p]
    ctx = seccomp.seccomp_init(0x7fff0000)  # allow ordinary computation
    if not ctx:
        raise RuntimeError('seccomp_init failed')
    try:
        denied = '''socket socketpair connect bind listen accept accept4 sendto sendmsg
            recvfrom recvmsg ptrace process_vm_readv process_vm_writev pidfd_getfd
            pidfd_open pidfd_send_signal kill tkill tgkill clone clone3 fork vfork
            execve execveat setsid setpgid unshare setns mount umount2 pivot_root chroot
            bpf perf_event_open userfaultfd io_uring_setup io_uring_enter io_uring_register
            open_by_handle_at name_to_handle_at truncate chmod fchmod fchmodat fchmodat2
            utime utimes futimesat utimensat
            setxattr lsetxattr fsetxattr removexattr lremovexattr fremovexattr
            shmget shmat shmdt shmctl msgget msgsnd msgrcv msgctl
            semget semop semtimedop semctl
            chown fchown lchown fchownat setuid setgid setreuid setregid setresuid
            setresgid setgroups capset reboot kexec_load kexec_file_load init_module
            finit_module delete_module swapon swapoff quotactl keyctl add_key request_key'''
        for name in denied.split():
            number = seccomp.seccomp_syscall_resolve_name(name.encode())
            if number >= 0 and seccomp.seccomp_rule_add(ctx, 0x50000 | errno.EPERM, number, 0):
                raise RuntimeError('seccomp rule failed: ' + name)
        # ABI 1 does not mediate truncate separately. O_RDONLY|O_TRUNC must
        # not turn read-only runtime access into a writable-file loophole.
        for name, arg in [('open', 1), ('openat', 2)]:
            number = seccomp.seccomp_syscall_resolve_name(name.encode())
            comparison = _ArgCompare(arg, 7, os.O_ACCMODE | os.O_TRUNC, os.O_TRUNC)
            if seccomp.seccomp_rule_add_array(ctx, 0x50000 | errno.EPERM, number, 1, ctypes.byref(comparison)):
                raise RuntimeError('seccomp truncate rule failed')
        # openat2 stores flags in a pointer that seccomp cannot inspect.
        number = seccomp.seccomp_syscall_resolve_name(b'openat2')
        if number >= 0 and seccomp.seccomp_rule_add(ctx, 0x50000 | errno.EPERM, number, 0):
            raise RuntimeError('seccomp openat2 rule failed')
        if seccomp.seccomp_load(ctx):
            raise RuntimeError('seccomp_load failed')
    finally:
        seccomp.seccomp_release(ctx)


def install_evalplus(worker_root):
    import evalplus.eval as evaluator
    original = evaluator.unsafe_execute
    from multiprocessing.popen_fork import Popen

    def wait_for_exit(process, timeout=None):
        # Closing inherited pipes makes CPython's sentinel readable before exit.
        # Poll waitpid instead, so a candidate cannot turn join(timeout) into an
        # unbounded wait by closing its sentinel pipe (deliberately or otherwise).
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            status = process.poll()
            if status is not None:
                return status
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                return None
            time.sleep(0.01 if remaining is None else min(0.01, remaining))

    Popen.wait = wait_for_exit

    def sandboxed_execute(*args, **kwargs):
        work = tempfile.mkdtemp(prefix='worker-', dir=worker_root)
        # Shared result arrays are already mapped. Close inherited controller FDs.
        os.closerange(3, resource.getrlimit(resource.RLIMIT_NOFILE)[0])
        os.environ.clear()
        tempfile.tempdir = work
        os.chdir(work)
        restrict(work)
        Path(work, 'sandbox_applied').write_text('landlock+seccomp\n')
        return original(*args, **kwargs)

    evaluator.unsafe_execute = sandboxed_execute


if __name__ == '__main__':
    work = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix='native-sandbox-probe-')
    restrict(work)
    Path(work, 'allowed').write_text('ok')
    for operation in (lambda: Path('/etc/passwd').read_text(),
                      lambda: Path('/tmp/native-sandbox-forbidden').write_text('bad'),
                      lambda: socket.socket(),
                      lambda: os.kill(os.getppid(), 0),
                      lambda: os.fork()):
        try:
            operation()
        except PermissionError:
            continue
        raise RuntimeError('Sandbox restriction did not hold')
    print('Landlock+seccomp probe passed', flush=True)
