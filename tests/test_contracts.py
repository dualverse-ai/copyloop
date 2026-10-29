import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from copyloop.common import digest, load, models, save, write_rows
from copyloop.data import base_prompt
from copyloop.evaluation.parser import parse_tokens
from copyloop.runner import launch_plan, tagged, validate_metrics
from copyloop.summary import summarize


class Contracts(unittest.TestCase):
    def test_main_defaults_and_ar_path(self):
        self.assertEqual(len(models()), 6)
        for name, config in models().items():
            self.assertEqual(config['k'], 2 if config['thinking'] else 6)
            prepared = dict(model=name, config=config, model_path='/model')
            ar, _, _ = launch_plan(prepared, 'ar', 30000)
            chain, _, _ = launch_plan(prepared, 'copyloop', 30000)
            self.assertNotIn('--loopspec', ar)
            self.assertNotIn('--second', chain)
            self.assertEqual(chain[chain.index('--first') + 1], str(config['screen_depth']))
            if config['thinking']:
                self.assertEqual(ar[ar.index('--max-total-tokens') + 1], '9408')
                self.assertEqual(chain[chain.index('--max-total-tokens') + 1], '37632')

    def test_experiment_environment_is_not_inherited(self):
        config = models()['ouro-1.4b']
        with patch.dict(os.environ, {'CHAIN_AUDIT_CONTROL': '/bad', 'LOOPSPEC_CHAIN_K': '8',
                                    'LOOPSKIP_ABLATE_NO_SCREEN': '1', 'K_BREADTH_CONTROL': '/bad'}):
            _, env, _ = launch_plan(dict(model='ouro-1.4b', config=config, model_path='/model'), 'copyloop', 30000)
        for key in ('CHAIN_AUDIT_CONTROL', 'LOOPSPEC_CHAIN_K', 'LOOPSKIP_ABLATE_NO_SCREEN', 'K_BREADTH_CONTROL'):
            self.assertNotIn(key, env)
        self.assertEqual(env['LOOPSPEC_WAVEFRONT'], '0')

    def test_first_token_and_rejected_candidate_accounting(self):
        counters = dict(decode_tokens=2, decode_seconds=.4, chain_overflow_events=0,
                        eager_decode_calls=0, recurrent_calls=8)
        validate_metrics(dict(output_ids=[11, 12, 2]), counters, 8, 'copyloop')
        with self.assertRaises(ValueError):
            validate_metrics(dict(output_ids=[11, 12, 2]), dict(counters, decode_tokens=3), 8, 'copyloop')
        with self.assertRaises(RuntimeError):
            validate_metrics(dict(output_ids=[11, 12, 2]), dict(counters, chain_overflow_events=1), 8, 'copyloop')
        with self.assertRaises(ValueError):
            validate_metrics(dict(output_ids=[11, 12, 2]), dict(counters, decode_seconds=float('nan')), 8, 'ar')

    def test_log_metrics_exclude_prior_warmup(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'server.log'
            first = 'INFO generation_metrics {"decode_tokens":7}\n'
            path.write_text(first + 'INFO generation_metrics {"decode_tokens":19}\n')
            self.assertEqual(tagged(path, len(first.encode()), 'generation_metrics'), [{'decode_tokens': 19}])

    def test_thinking_final_only_and_unfinished(self):
        parsed = parse_tokens([9, 4, 12, 2], prompt_thinking=True, max_new_tokens=8,
                              decode=lambda ids: ' '.join(map(str, ids)))
        self.assertEqual(parsed.final_ids, [12])
        self.assertTrue(parsed.eligible)
        unfinished = parse_tokens([9, 10], prompt_thinking=True, max_new_tokens=2, decode=str)
        self.assertTrue(unfinished.cap_hit)
        self.assertTrue(unfinished.unfinished_think)
        self.assertFalse(unfinished.eligible)

    def test_prompt_templates_do_not_add_bos_or_chat_to_base(self):
        p = base_prompt('humaneval', 0, dict(task_id='HumanEval/0', entry_point='f', prompt='def f():\n    pass\n'))
        self.assertTrue(p['context'].endswith('```python'))
        self.assertIn('\n``` Below is a Python script', p['context'])
        g = base_prompt('gsm8k', 3, dict(question='Synthetic question?', answer='work #### 4'))
        self.assertEqual(g['task_id'], 'GSM8K/3')
        self.assertEqual(g['context'].count('Q: '), 4)
        self.assertEqual(g['target'], '4')

    def make_run(self, directory, method, durations, input_ids=None):
        directory.mkdir()
        docs = [dict(task_id=str(i), input_ids=input_ids or [17], output_budget=100) for i in range(2)]
        rows = []
        for i, (tokens, seconds) in enumerate(zip([10, 90], durations)):
            counters = dict(decode_tokens=tokens, decode_seconds=seconds, chain_overflow_events=0,
                            eager_decode_calls=0, recurrent_calls=10)
            rows.append(dict(task_id=str(i), output_ids=[1] * (tokens + 1), decode_tokens=tokens,
                             decode_seconds=seconds, counters=counters, max_new_tokens=100))
        save(directory / 'prepared.json', dict(model='synthetic', dataset='math500', config={},
             source={}, records=docs, full_benchmark=False))
        save(directory / 'run.json', dict(method=method))
        write_rows(directory / 'results.jsonl', rows)
        save(directory / 'DONE.json', dict(count=2, results_sha256=digest((directory / 'results.jsonl').read_bytes())))

    def test_ratio_of_aggregate_rates_and_pairing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_run(root / 'ar', 'ar', [1, 9])
            self.make_run(root / 'cl', 'copyloop', [1, 4])
            args = SimpleNamespace(runs=[root / 'ar', root / 'cl'], output=root / 'summary.json')
            summarize(args)
            rows = load(args.output)['rows']
            self.assertEqual(rows[1]['decode_tps'], 20)
            self.assertEqual(rows[1]['speedup_vs_ar'], 2)
            self.make_run(root / 'mismatch', 'copyloop', [1, 4], input_ids=[18])
            with self.assertRaises(ValueError):
                summarize(SimpleNamespace(runs=[root / 'ar', root / 'mismatch'], output=root / 'bad.json'))
            with self.assertRaises(ValueError):
                summarize(SimpleNamespace(runs=[root / 'cl'], output=root / 'bad.json'))

    def test_corruption_and_duplicate_runs_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_run(root / 'ar', 'ar', [1, 9])
            with self.assertRaises(ValueError):
                summarize(SimpleNamespace(runs=[root / 'ar', root / 'ar'], output=root / 'bad.json'))
            with (root / 'ar/results.jsonl').open('a') as stream:
                stream.write('{}\n')
            with self.assertRaises(ValueError):
                summarize(SimpleNamespace(runs=[root / 'ar'], output=root / 'bad.json'))


if __name__ == '__main__':
    unittest.main()
