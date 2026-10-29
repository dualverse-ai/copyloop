"""Main task scoring; generated code runs only in the retained kernel sandbox."""
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from ..common import PACKAGE, digest, load, read_rows, save, write_rows
from ..process_group import managed_process
from ..summary import completed


def code_scores(prepared, rows, directory):
    from evalplus.sanitize import sanitize
    dataset = prepared['dataset']
    stop = prepared['config']['tasks'][dataset]['stop']
    samples, extraction = [], []
    for row in rows:
        text = row['output_text']
        n = max((n for s in stop for n in range(1, len(s) + 1) if text.endswith(s[:n])), default=0)
        stripped = text[:-n] if n else text
        solution = sanitize(stripped, entrypoint=row['entry_point'])
        samples.append(dict(task_id=row['task_id'], solution=solution))
        extraction.append(dict(task_id=row['task_id'], raw=text, stripped=stripped, sanitized=solution))
    canonical = Path(prepared['canonical_path'])
    source = load(PACKAGE / 'presets/assets.json')[dataset + '_canonical']
    if digest(canonical.read_bytes()) != source['sha256']:
        raise ValueError('Canonical scoring dataset hash mismatch')
    ids = {r['task_id'] for r in rows}
    problems = [r for r in read_rows(canonical) if r['task_id'] in ids]
    if {r['task_id'] for r in problems} != ids:
        raise ValueError('Scoring task IDs missing')
    write_rows(directory / 'samples.jsonl', samples)
    write_rows(directory / 'dataset.jsonl', problems)
    save(directory / 'extraction.json', extraction)
    for subdir in ('tmp', 'cache'):
        (directory / subdir).mkdir()
    ds = 'humaneval' if dataset == 'humaneval' else 'mbpp'
    env = dict(PATH=os.environ.get('PATH', '/usr/bin:/bin'), HOME=str(directory / 'tmp'),
               TMPDIR=str(directory / 'tmp'), XDG_CACHE_HOME=str(directory / 'cache'),
               PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               CUDA_VISIBLE_DEVICES='', LANG='C.UTF-8')
    env['HUMANEVAL_OVERRIDE_PATH' if ds == 'humaneval' else 'MBPP_OVERRIDE_PATH'] = str(directory / 'dataset.jsonl')
    command = ['prlimit', '--as=8589934592', '--cpu=900', '--fsize=2147483648', '--core=0', '--',
               sys.executable, '-B', str(Path(__file__).with_name('code_worker.py')),
               str(directory / 'samples.jsonl'), ds, '--native-sandbox']
    save(directory / 'command.json', command)
    with (directory / 'evaluation.log').open('x') as log:
        with managed_process(command, cwd=directory / 'tmp', env=env, stdout=log,
                             stderr=subprocess.STDOUT) as process:
            code = process.wait(timeout=1800)
    if code:
        raise RuntimeError('Sandboxed scoring failed; inspect evaluation.log (no unrestricted fallback)')
    result = load(directory / 'samples_eval_results.json')['eval']
    if set(result) != ids or any(len(v) != 1 for v in result.values()):
        raise ValueError('Unexpected scoring task IDs or repetitions')
    return [dict(task_id=r['task_id'], correct=result[r['task_id']][0]['base_status'] == 'pass'
                 and result[r['task_id']][0]['plus_status'] == 'pass') for r in rows]


def evaluate(args):
    run = Path(args.run).resolve()
    prepared, config, rows = completed(run)
    directory = run / 'scoring'
    directory.mkdir(exist_ok=False)
    thinking = prepared['config']['thinking']
    versions = {}
    if prepared['dataset'] in ('humaneval', 'mbppplus'):
        details = code_scores(prepared, rows, directory)
        versions['evalplus'] = importlib.metadata.version('evalplus')
    else:
        from .math_score import score_final, score_record
        versions['math-verify'] = importlib.metadata.version('math-verify')
        tokenizer = None
        if thinking:
            from transformers import AutoTokenizer, PreTrainedConfig
            tokenizer = AutoTokenizer.from_pretrained(prepared['model_path'], config=PreTrainedConfig(), local_files_only=True)
        details = []
        for row in rows:
            if thinking:
                scored = score_record(dict(row, checkpoint_kind='thinking', eos_id=2), tokenizer)
                details.append(dict(scored, correct=scored['scoring']['correct']))
            else:
                parsed = SimpleNamespace(eligible=bool(row['output_text'].strip()), final_text=row['output_text'],
                                         malformed=False, unfinished_think=False)
                score = score_final(parsed, dataset=prepared['dataset'], target=row['target'])
                details.append(dict(task_id=row['task_id'], **score))
    save(directory / 'details.json', details)
    summary = dict(count=len(rows), correct_count=sum(d['correct'] for d in details),
                   budget_hits=sum(len(r['output_ids']) >= r['max_new_tokens'] for r in rows),
                   unfinished_thinking=sum(d['segmentation']['unfinished_think'] for d in details) if thinking else None,
                   results_sha256=digest((run / 'results.jsonl').read_bytes()), versions=versions,
                   denominator='all completed requests, including capped, unfinished and unparseable responses; scorer failure aborts scoring')
    save(directory / 'summary.json', summary)
    print(json.dumps(summary, indent=2))
