"""Own a backend through a pipe-watched guardian, including parent SIGKILL."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time


_GRACE_SECONDS = 10


def _terminate_group(process):
    """Bound one owned group's TERM grace even if its leader has exited."""
    deadline = time.monotonic() + _GRACE_SECONDS
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=max(0, deadline - time.monotonic()))
    except subprocess.TimeoutExpired:
        pass
    while True:
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            break
        time.sleep(min(0.05, remaining))
    process.wait()


def _guardian(liveness_fd, command):
    """This session survives the attempt wrapper's process-group SIGKILL."""
    backend = None
    def interrupted(signum, frame):
        raise SystemExit(128 + signum)
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, interrupted)
    try:
        # If the owner died during launch, avoid starting an unowned backend.
        if select.select([liveness_fd], [], [], 0)[0] and not os.read(liveness_fd, 1):
            return 143
        backend = subprocess.Popen(command, start_new_session=True, close_fds=True)
        while True:
            status = backend.poll()
            if status is not None:
                return status if status >= 0 else 128 - status
            if select.select([liveness_fd], [], [], 0.1)[0]:
                if not os.read(liveness_fd, 1):
                    return 143
    finally:
        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, signal.SIG_IGN)
        os.close(liveness_fd)
        if backend is not None:
            _terminate_group(backend)


@contextmanager
def managed_process(command, **kwargs):
    """Yield the guardian Popen, whose poll/wait reflects backend lifetime.

    cwd/env/stdout/stderr are inherited by the owned backend. The owner's pipe
    writer is never passed to descendants, so owner death always produces EOF.
    """
    unexpected = set(kwargs) - {'cwd', 'env', 'stdout', 'stderr'}
    if unexpected:
        raise TypeError(f'Unsupported backend launch options: {sorted(unexpected)}')
    previous = {}
    read_fd, write_fd = os.pipe()
    guardian = None
    def interrupted(signum, frame):
        raise SystemExit(128 + signum)
    try:
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous[signum] = signal.signal(signum, interrupted)
        guardian = subprocess.Popen(
            [sys.executable, '-B', str(Path(__file__).resolve()), '--guardian', str(read_fd), json.dumps([str(arg) for arg in command])],
            start_new_session=True, close_fds=True, pass_fds=(read_fd,), **kwargs,
        )
        os.close(read_fd)
        read_fd = None
        yield guardian
    finally:
        for signum in previous:
            signal.signal(signum, signal.SIG_IGN)
        if read_fd is not None:
            os.close(read_fd)
        os.close(write_fd)
        try:
            if guardian is not None:
                # EOF asks the independent guardian to clean the backend group.
                try:
                    guardian.wait(timeout=_GRACE_SECONDS + 5)
                except subprocess.TimeoutExpired:
                    guardian.terminate()
                    guardian.wait(timeout=_GRACE_SECONDS + 5)
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)


if __name__ == '__main__':
    if len(sys.argv) != 4 or sys.argv[1] != '--guardian':
        raise SystemExit('Internal backend guardian entrypoint')
    raise SystemExit(_guardian(int(sys.argv[2]), json.loads(sys.argv[3])))
