"""Sequential generation with native decode metrics and owned server lifetime."""
import importlib.metadata
import json
import os
import sys
import subprocess
import time
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError
from .common import PACKAGE, digest, load, preset, save, validate_model
from .process_group import managed_process


def launch_plan(prepared, method, port):
    config = prepared['config']
    if config != preset(prepared['model']):
        raise ValueError('Prepared config differs from frozen public preset')
    if method not in ('ar', 'copyloop') or not 1024 <= port <= 55535:
        raise ValueError('Invalid method or server port')
    backend = PACKAGE / '_backends' / config['backend']
    slots = config.get('ar_kv_slots', config['kv_slots']) if method == 'ar' else config['kv_slots']
    command = [sys.executable, '-B', '-m', 'sglang_recurrent.server',
               '--model', prepared['model_path'], '--host', '127.0.0.1', '--port', str(port),
               '--dtype', 'bfloat16', '--attention-backend', 'triton',
               '--max-running-requests', '1', '--random-seed', '0',
               '--mem-fraction-static', str(config['mem_fraction']),
               '--max-total-tokens', str(slots), '--context-length', str(config['context_length']),
               '--cuda-graph-max-bs', '1', '--disable-radix-cache', '--disable-overlap-schedule',
               '--chunked-prefill-size', '-1']
    if method == 'copyloop':
        command += ['--loopspec', '--first', str(config['screen_depth']), '--chain-k', str(config['k']),
                    '--chain-truncate', 'shallow', '--chain-flatten']
    env = {k: v for k, v in os.environ.items() if not k.startswith(
        ('LOOPSPEC_', 'LOOPSKIP_', 'CHAIN_', 'RAVEN_AUDIT_', 'WFD_', 'HUGINN_CAPTURE_', 'K_'))}
    env.update(PYTHONPATH=os.pathsep.join([str(backend), str(PACKAGE / '_support')]),
               PYTHONDONTWRITEBYTECODE='1', CHAIN_TRACE='0',
               LOOPSPEC_BUCKET_GRAPHS='1' if method == 'copyloop' else '0',
               LOOPSPEC_CHAIN_FLATTEN='1' if method == 'copyloop' else '0',
               LOOPSPEC_CHAIN_W_CAP='80', LOOPSPEC_WAVEFRONT='0',
               TOKENIZERS_PARALLELISM='false')
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        env[key] = '8'
    return command, env, backend


def http(url, body=None, timeout=7200):
    request = Request(url, data=json.dumps(body).encode() if body is not None else None,
                      headers={'Content-Type': 'application/json'})
    with urlopen(request, timeout=timeout) as response:
        return response.read()


def tagged(path, offset, tag):
    with Path(path).open('rb') as stream:
        stream.seek(offset)
        lines = stream.read().decode(errors='replace').splitlines()
    records = []
    for line in lines:
        if tag + ' ' in line:
            try:
                records.append(json.loads(line.split(tag + ' ', 1)[1]))
            except json.JSONDecodeError:
                # A concurrent logger may not yet have flushed the whole line.
                continue
    return records


def metrics_after(path, offset):
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        records = tagged(path, offset, 'generation_metrics')
        if records:
            if len(records) != 1:
                raise RuntimeError('Ambiguous per-request decode metrics')
            return records[0]
        time.sleep(.05)
    raise TimeoutError('Missing native decode metrics')


def validate_metrics(response, counters, budget, method):
    ids = response['output_ids']
    if not 0 < len(ids) <= budget or counters['decode_tokens'] != len(ids) - 1:
        raise ValueError('Output budget or native decode token accounting mismatch')
    seconds = counters['decode_seconds']
    import math
    if not math.isfinite(seconds) or seconds < 0 or (len(ids) > 1 and seconds <= 0):
        raise ValueError('Invalid decode interval')
    if method == 'copyloop' and counters['decode_tokens']:
        if counters.get('chain_overflow_events') != 0 or counters.get('eager_decode_calls') != 0:
            raise RuntimeError('CopyLoop overflow or eager fallback; reject performance run')
        if counters.get('recurrent_calls', 0) <= 0:
            raise RuntimeError('Missing graph execution counters')


def generate(args):
    prepared = load(args.prepared)
    command, env, backend = launch_plan(prepared, args.method, args.port)
    plan = dict(model=prepared['model'], method=args.method, command=command,
                records=len(prepared['records']), backend=str(backend),
                environment={k: env[k] for k in env if k.startswith(('LOOPSPEC_', 'CHAIN_', 'OMP_', 'MKL_', 'OPENBLAS_'))})
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        return
    devices = env.get('CUDA_VISIBLE_DEVICES', '')
    if not devices or ',' in devices or devices == '-1':
        raise ValueError('Set CUDA_VISIBLE_DEVICES to one available GPU')
    validate_model(prepared['model_path'], prepared['config'])
    docs = prepared['records']
    if not docs or len({d['task_id'] for d in docs}) != len(docs):
        raise ValueError('Empty or duplicated prepared tasks')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'prepared.json', prepared)
    plan.update(prepared_sha256=digest(Path(args.prepared).read_bytes()),
                cuda_visible_devices=devices, python=sys.version,
                runtime_versions={p: importlib.metadata.version(p) for p in ('torch', 'sglang', 'transformers', 'triton')},
                source_hashes={str(p.relative_to(PACKAGE)): digest(p.read_bytes())
                               for directory in (backend, PACKAGE / '_support') for p in directory.rglob('*.py')})
    save(output / 'run.json', plan)
    logpath = output / 'server.log'
    base = f'http://127.0.0.1:{args.port}'
    stops = prepared['config']['tasks'][prepared['dataset']]['stop']
    def request(doc, budget):
        offset = logpath.stat().st_size
        body = dict(input_ids=doc['input_ids'], stream=False,
                    sampling_params=dict(temperature=0, max_new_tokens=budget, stop=stops, ignore_eos=False))
        started = time.perf_counter()
        response = json.loads(http(base + '/generate', body))
        elapsed = time.perf_counter() - started
        counters = metrics_after(logpath, offset)
        validate_metrics(response, counters, budget, args.method)
        return response, counters, elapsed
    try:
        with logpath.open('x') as log, managed_process(command, cwd=backend, env=env,
                                                      stdout=log, stderr=subprocess.STDOUT) as process:
            began = time.monotonic()
            while True:
                if process.poll() is not None:
                    raise RuntimeError('Server exited; inspect server.log')
                if time.monotonic() - began > 1200:
                    raise TimeoutError('Server startup exceeded 1200 seconds')
                try:
                    http(base + '/health', timeout=2)
                    break
                except (URLError, TimeoutError):
                    time.sleep(1)
            if prepared['config']['thinking']:
                capacity = tagged(logpath, 0, 'thinking_capacity')
                expected = prepared['config']['ar_kv_slots'] if args.method == 'ar' else prepared['config']['kv_slots']
                if not capacity or any(c['max_total_num_tokens'] != expected or set(c['eos_ids']) != {2} for c in capacity):
                    raise ValueError('Thinking KV capacity clamped or EOS mismatch')
                save(output / 'capacity.json', capacity)
            response, counters, elapsed = request(docs[0], 8)
            save(output / 'warmup.json', dict(response=response, counters=counters, e2e_seconds=elapsed))
            with (output / 'results.jsonl').open('x') as results:
                for doc in docs:
                    response, counters, elapsed = request(doc, doc['output_budget'])
                    row = dict(task_id=doc['task_id'], dataset=prepared['dataset'], model=prepared['model'],
                               method=args.method, target=doc['target'], entry_point=doc['entry_point'],
                               output_ids=response['output_ids'], output_text=response['text'],
                               decode_tokens=counters['decode_tokens'], decode_seconds=counters['decode_seconds'],
                               e2e_seconds=elapsed, counters=counters, response=response,
                               max_new_tokens=doc['output_budget'], prompt_thinking=prepared['config']['thinking'])
                    results.write(json.dumps(row, ensure_ascii=False) + '\n')
                    results.flush()
                    print('Completed', doc['task_id'], flush=True)
        save(output / 'DONE.json', dict(count=len(docs), results_sha256=digest((output / 'results.jsonl').read_bytes())))
    except BaseException as error:
        save(output / 'FAILED.json', dict(error=repr(error)))
        raise
