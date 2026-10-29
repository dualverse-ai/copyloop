"""Pinned asset acquisition and exact main-experiment prompt preparation."""
import ast
import gzip
import json
import unicodedata
from pathlib import Path
from urllib.request import urlopen
from .common import PACKAGE, digest, load, preset, read_rows, save, validate_model


def download(args):
    config = preset(args.model)
    plan = {k: config[k] for k in ('repo', 'revision')}
    print(json.dumps(plan, indent=2))
    if args.dry_run:
        return
    from huggingface_hub import snapshot_download
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Choose an empty model destination')
    snapshot_download(repo_id=config['repo'], revision=config['revision'], local_dir=output,
                      allow_patterns=['*.json', '*.safetensors', '*.py', '*.model',
                                      '*.tiktoken', '*.txt', '*.md', '*.jinja', 'LICENSE*'])
    if not (output / 'config.json').is_file() or not list(output.glob('*.safetensors')):
        raise RuntimeError('Incomplete model snapshot')
    save(output / 'copyloop_model.json', plan)


def asset(name, directory, offline=False):
    source = load(PACKAGE / 'presets/assets.json')[name]
    suffix = '.parquet' if source['format'] == 'parquet' else '.jsonl'
    path = Path(directory) / (name + suffix)
    if not path.exists():
        if offline:
            raise FileNotFoundError(path)
        with urlopen(source['url'], timeout=180) as response:
            raw = response.read()
        if source['format'] == 'jsonl-gzip':
            raw = gzip.decompress(raw)
        if digest(raw) != source['sha256']:
            raise ValueError(f'Download checksum mismatch: {name}')
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(raw)
    if digest(path.read_bytes()) != source['sha256']:
        raise ValueError(f'Asset checksum mismatch: {path}')
    if suffix == '.parquet':
        import pyarrow.parquet as pq
        rows = pq.read_table(path).to_pylist()
    else:
        rows = read_rows(path)
    return rows, path.resolve(), source


def base_prompt(dataset, index, row):
    target, entry = None, row.get('entry_point')
    if dataset == 'humaneval':
        tid = row['task_id']
        prompt = ('Please provide a self-contained Python script that solves the following '
                  'problem in a markdown code block:\n```\n' + row['prompt'].strip() +
                  '\n``` Below is a Python script with a self-contained function that solves '
                  'the problem and passes corresponding tests:\n```python')
    elif dataset == 'mbppplus':
        tid = 'Mbpp/' + str(row['task_id'])
        entry = next(n.name for n in ast.parse(row['code']).body if isinstance(n, ast.FunctionDef))
        prompt = ('Write a Python function to solve the following task. Return only Python code.\n\n'
                  + row['prompt'] + '\nYour code should pass these tests:\n'
                  + '\n'.join(row['test_list']) + '\n\n```python\n')
    elif dataset == 'gsm8k':
        tid = f'GSM8K/{index}'
        examples = load(PACKAGE / 'presets/gsm8k_fewshot.json')
        prefix = '\n\n'.join('Q: ' + r['question'] + '\nA: ' + r['target'] for r in examples)
        prompt = prefix + '\n\nQ: ' + row['question'] + '\nA:'
        target = row['answer'].split('####')[-1].strip()
    elif dataset == 'math500':
        tid = f'MATH500/{index}'
        prompt = ('Solve the following mathematics problem. Show your reasoning and put your '
                  'final answer in \\boxed{}.\n\nProblem: ' + row['problem'] + '\n\nSolution:')
        target = row['answer']
    else:
        raise ValueError(dataset)
    return dict(task_id=tid, context=prompt, target=target, entry_point=entry)


def prepare_records(model, dataset, rows, tokenizer):
    config = preset(model)
    task = config['tasks'][dataset]
    if config['thinking']:
        panel = [r for r in load(PACKAGE / 'presets/thinking_panel.json') if r['dataset'] == dataset]
        docs = []
        for item in panel:
            raw = rows[item['source_row']]
            problem = raw['problem']
            normalized = ' '.join(unicodedata.normalize('NFKC', problem).split())
            if digest(normalized.encode()) != item['problem_sha256']:
                raise ValueError('Thinking panel content changed: ' + item['task_id'])
            messages = [{'role': 'user', 'content': problem + '\n\nPlease put your final answer in \\boxed{}.'}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                                  add_generation_prompt=True, enable_thinking=True)
            if not prompt.endswith('<think>\n'):
                raise ValueError('Thinking prefix missing')
            docs.append(dict(task_id=item['task_id'], context=prompt, target=str(raw['answer']), entry_point=None))
    else:
        docs = [base_prompt(dataset, i, row) for i, row in enumerate(rows)]
        if model == 'huginn-3.5b':
            for doc in docs:
                text = doc['context']
                if dataset == 'humaneval':
                    text = ('Please implement the following Python function. Return a self-contained '
                            'Python implementation in a markdown code block.\n\n') + text
                elif dataset == 'mbppplus':
                    text = text.removesuffix('```python\n') + 'Return a self-contained Python implementation in a markdown code block.'
                doc['context'] = tokenizer.apply_chat_template(
                    [{'role': 'system', 'content': 'You are a helpful assistant.'},
                     {'role': 'user', 'content': text}], tokenize=False, add_generation_prompt=True)
    for doc in docs:
        doc['input_ids'] = tokenizer.encode(doc['context'], add_special_tokens=False)
        available = config['context_length'] - len(doc['input_ids']) - (128 if config['thinking'] else 0)
        if available < task['budget']:
            raise ValueError(f"Prompt leaves insufficient output capacity: {doc['task_id']}")
        doc.update(output_budget=task['budget'], prompt_sha256=digest(doc['context'].encode()))
    if len(docs) != task['count']:
        raise ValueError('Unexpected benchmark size')
    if not config['thinking']:
        checks = load(PACKAGE / 'presets/base_prompt_checks.json')[model][dataset]
        for doc, check in zip(docs, checks, strict=True):
            for key in ('task_id', 'prompt_sha256', 'output_budget'):
                if doc[key] != check[key]:
                    raise ValueError(f'Historical prompt mismatch: {key}, {doc["task_id"]}')
            if digest(json.dumps(doc['input_ids'], separators=(',', ':')).encode()) != check['input_ids_sha256']:
                raise ValueError('Historical input token IDs differ')
    return docs


def prepare(args):
    from transformers import AutoTokenizer, PreTrainedConfig
    config = preset(args.model)
    if args.dataset not in config['tasks']:
        raise ValueError(f'Select one of {list(config["tasks"])} for this model')
    validate_model(args.model_path, config)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, config=PreTrainedConfig(), local_files_only=True)
    rows, path, source = asset(args.dataset, args.assets, args.offline)
    docs = prepare_records(args.model, args.dataset, rows, tokenizer)
    canonical = None
    if args.dataset in ('humaneval', 'mbppplus'):
        _, canonical, _ = asset(args.dataset + '_canonical', args.assets, args.offline)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError('--limit must be positive')
        docs = docs[:args.limit]
    save(args.output, dict(schema=1, model=args.model, model_path=str(Path(args.model_path).resolve()),
         config=config, dataset=args.dataset, source=source, asset_path=str(path),
         canonical_path=str(canonical) if canonical else None, records=docs,
         full_benchmark=len(docs) == config['tasks'][args.dataset]['count']))
    print(f'Prepared {len(docs)} prompts: {args.output}')
