"""Final-only math scoring; run in loopskip-score env with math-verify 0.9.0."""
import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path
from .gsm8k_scoring import extract, correct
from .parser import parse_tokens


def score_final(parsed, *, dataset, target):
    result = {'correct': False, 'prediction_parse_ok': False,
              'eligible': parsed.eligible, 'exclusion': None}
    if not parsed.eligible:
        result['exclusion'] = ('malformed' if parsed.malformed else
                               'unfinished_think' if parsed.unfinished_think else 'empty_final')
        return result
    if dataset == 'gsm8k':
        pred = extract(parsed.final_text)
        result.update(prediction=pred, prediction_parse_ok=pred != '[invalid]',
                      correct=bool(correct(pred, str(target))))
    elif dataset in ('aime', 'aime2024', 'aime2025'):
        from .aime import score_integer
        result.update(score_integer(parsed.final_text, target))
    elif dataset == 'beyondaime':
        from .aime import score_integer
        result.update(score_integer(parsed.final_text, target, bounded=False))
    elif dataset in ('math', 'math500'):
        from math_verify import parse, verify, LatexExtractionConfig, ExprExtractionConfig
        gold = parse('\\boxed{' + str(target) + '}', extraction_config=[LatexExtractionConfig()])
        pred = parse(parsed.final_text, extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()])
        result.update(prediction=str(pred), gold=str(gold), gold_parse_ok=bool(gold),
                      prediction_parse_ok=bool(pred), correct=bool(gold and pred and verify(gold, pred)))
    else:
        raise ValueError(f'unsupported dataset {dataset}')
    if dataset in ('aime', 'aime2024', 'aime2025', 'beyondaime'):
        from math_verify import parse, verify, LatexExtractionConfig, ExprExtractionConfig
        gold = parse('\\boxed{' + str(target) + '}', extraction_config=[LatexExtractionConfig()])
        pred = parse(parsed.final_text, extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()])
        result['symbolic_diagnostic'] = {'correct': bool(gold and pred and verify(gold, pred)),
                                         'prediction_parse_ok': bool(pred), 'prediction': str(pred),
                                         'role': 'diagnostic only; does not replace strict integer primary score'}
        result['format_failure'] = not result['prediction_parse_ok']
        result['parsed_wrong_answer'] = result['prediction_parse_ok'] and not result['correct']
    return result


def score_record(record, tokenizer):
    for field in ('output_ids', 'prompt_thinking', 'max_new_tokens', 'dataset', 'target', 'task_id', 'checkpoint_kind', 'eos_id'):
        if field not in record:
            raise ValueError(f'missing required field {field}')
    if type(record['prompt_thinking']) is not bool:
        raise ValueError('prompt_thinking must be actual rendered-prefix boolean')
    if record['checkpoint_kind'] not in ('base', 'thinking'):
        raise ValueError('unknown checkpoint_kind')
    base = record['checkpoint_kind'] == 'base'
    expected_eos = 0 if base else 2
    if record['eos_id'] != expected_eos or tokenizer.eos_token_id != expected_eos:
        raise ValueError('record/tokenizer/checkpoint EOS mismatch')
    if not base and (tokenizer.convert_tokens_to_ids('<think>') != 3 or tokenizer.convert_tokens_to_ids('</think>') != 4):
        raise ValueError('Thinking tag tokenizer mismatch')
    if base and record['prompt_thinking']:
        raise ValueError('base checkpoint must not claim Thinking token-prefix semantics')
    parsed = parse_tokens(record['output_ids'], prompt_thinking=record['prompt_thinking'],
                          open_id=-3 if base else 3, close_id=-4 if base else 4,
                          eos_id=tokenizer.eos_token_id if base else 2,
                          max_new_tokens=record['max_new_tokens'],
                          decode=lambda ids: tokenizer.decode(ids, skip_special_tokens=False))
    identity = {k: record[k] for k in ('arm', 'model', 'budget', 'run_id', 'repeat', 'checkpoint_kind', 'checkpoint_revision', 'max_new_tokens') if k in record}
    return dict(identity, task_id=record['task_id'], dataset=record['dataset'],
                segmentation=parsed.to_dict(),
                scoring=score_final(parsed, dataset=record['dataset'], target=record['target']))


def main():
    p=argparse.ArgumentParser(); p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--tokenizer',required=True)
    args=p.parse_args()
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True,local_files_only=True)
    src=Path(args.input); records=[json.loads(line) for line in src.read_text().splitlines() if line.strip()]
    if any(r.get('checkpoint_kind') != 'base' for r in records):
        assert tok.convert_tokens_to_ids('<think>')==3 and tok.convert_tokens_to_ids('</think>')==4 and tok.eos_token_id==2
    results=[score_record(r,tok) for r in records]
    dst=Path(args.output);dst.parent.mkdir(parents=True,exist_ok=True)
    dst.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in results))
    meta={'input_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'n':len(results),
          'correct':sum(x['scoring']['correct'] for x in results),
          'math_verify_version':importlib.metadata.version('math-verify'),
          'parser_sha256':hashlib.sha256(Path(__file__).with_name('parser.py').read_bytes()).hexdigest(),
          'scorer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    dst.with_suffix('.meta.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps(meta))
if __name__=='__main__':main()
