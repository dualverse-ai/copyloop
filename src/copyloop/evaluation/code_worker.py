"""Run EvalPlus with Bubblewrap or kernel restrictions on each candidate worker."""
from evalplus.evaluate import evaluate
from pathlib import Path
import subprocess
import sys
import tempfile


def repair_find_zero_accounting():
    """Repair an installed EvalPlus bug without changing its polynomial oracle.

    Its successful find_zero branch continues before recording test progress,
    causing every correct solution to fail the final completeness check.
    This process-local patch also applies when using Bubblewrap.
    """
    import inspect
    import evalplus.eval as evaluator
    source = inspect.getsource(evaluator.unsafe_execute)
    old = 'assert abs(_poly(*inp, out)) <= atol\n                            continue'
    new = ('assert abs(_poly(*inp, out)) <= atol\n'
           '                            details[i] = True\n'
           '                            progress.value += 1\n'
           '                            continue')
    if old in source:
        namespace = {}
        exec(source.replace(old, new, 1), vars(evaluator), namespace)
        evaluator.unsafe_execute = namespace['unsafe_execute']


def main():
    samples = sys.argv[1]
    repair_find_zero_accounting()
    with tempfile.TemporaryDirectory(prefix='humaneval-workers-') as workers:
        native = '--native-sandbox' in sys.argv[2:]
        if native:
            from native_sandbox import install_evalplus
            probe = Path(workers, 'probe')
            probe.mkdir()
            subprocess.run([sys.executable, '-B', str(Path(__file__).with_name('native_sandbox.py')), str(probe)], check=True, timeout=15)
            install_evalplus(workers)
        evaluate(dataset=sys.argv[2], samples=samples, parallel=4, test_details=True, min_time_limit=1.0, gt_time_limit_factor=4.0, i_just_wanna_run=True)
        if native and any(not (path / 'sandbox_applied').is_file() for path in Path(workers).glob('worker-*')):
            raise RuntimeError('Candidate sandbox initialization failed; reject scoring results')


if __name__ == '__main__':
    main()
