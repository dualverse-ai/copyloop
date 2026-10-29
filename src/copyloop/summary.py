"""Validate completed runs and aggregate paired decode throughput."""
import math
from pathlib import Path
from .common import digest, load, read_rows, save


def completed(run):
    run = Path(run)
    done, prepared, config = (load(run / name) for name in ('DONE.json', 'prepared.json', 'run.json'))
    path = run / 'results.jsonl'
    if digest(path.read_bytes()) != done['results_sha256']:
        raise ValueError('Completed result hash mismatch')
    rows = read_rows(path)
    expected = [d['task_id'] for d in prepared['records']]
    if [r['task_id'] for r in rows] != expected or len(set(expected)) != len(expected) or done['count'] != len(rows):
        raise ValueError('Missing, reordered or duplicated task IDs')
    if not rows:
        raise ValueError('Empty run')
    from .runner import validate_metrics
    for row, doc in zip(rows, prepared['records'], strict=True):
        validate_metrics({'output_ids': row['output_ids']}, row['counters'], doc['output_budget'], config['method'])
        if row['decode_tokens'] != row['counters']['decode_tokens'] or row['decode_seconds'] != row['counters']['decode_seconds']:
            raise ValueError('Decode metric mismatch')
    return prepared, config, rows


def summarize(args):
    groups, output = {}, []
    for name in args.runs:
        run = Path(name)
        prepared, config, rows = completed(run)
        tokens = sum(r['decode_tokens'] for r in rows)
        seconds = sum(r['decode_seconds'] for r in rows)
        if tokens <= 0 or seconds <= 0 or not math.isfinite(seconds):
            raise ValueError('Run has no measurable decode throughput')
        row = dict(model=prepared['model'], dataset=prepared['dataset'], method=config['method'],
                   run=str(run.resolve()), count=len(rows), full_benchmark=prepared['full_benchmark'],
                   decode_tokens=tokens, decode_seconds=seconds, decode_tps=tokens / seconds,
                   output_tokens=sum(len(r['output_ids']) for r in rows),
                   budget_hits=sum(len(r['output_ids']) >= r['max_new_tokens'] for r in rows),
                   accuracy_percent=None, speedup_vs_ar=None)
        scores = run / 'scoring/summary.json'
        if scores.exists():
            scoring = load(scores)
            if scoring['results_sha256'] != digest((run / 'results.jsonl').read_bytes()) or scoring['count'] != len(rows):
                raise ValueError('Stale or incompatible scoring')
            row['accuracy_percent'] = 100 * scoring['correct_count'] / len(rows)
        signature = [(d['task_id'], d['input_ids'], d['output_budget']) for d in prepared['records']]
        signature += [prepared['config'], prepared['source']]
        group = groups.setdefault((row['model'], row['dataset']), [])
        if any(r['method'] == row['method'] for r, _ in group):
            raise ValueError('Duplicate method run; repetitions require explicit pairing')
        group.append((row, signature))
        output.append(row)
    for group in groups.values():
        references = [(r, s) for r, s in group if r['method'] == 'ar']
        if len(references) != 1:
            raise ValueError('Supply exactly one matched AR run per model and task')
        reference, expected = references[0]
        for row, signature in group:
            if signature != expected:
                raise ValueError('AR and CopyLoop prompts, budgets or configurations differ')
            row['speedup_vs_ar'] = row['decode_tps'] / reference['decode_tps']
    save(args.output, dict(metric='sum(decode_tokens)/sum(decode_seconds); speedup = method TPS / AR TPS', rows=output))
    print(f'Wrote {len(output)} result rows to {args.output}')
