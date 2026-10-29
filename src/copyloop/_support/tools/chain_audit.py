"""Per-request position-keyed Raven initialization and AR margin capture."""
import json
import os
from pathlib import Path

from tools.raven_audit_backend import initial, make_table, digest


def install():
    from sglang.srt.model_executor.model_runner import ModelRunner
    if getattr(ModelRunner, '_chain_audit_installed', False):
        return
    ModelRunner._chain_audit_installed = True
    original = ModelRunner.forward
    active = {}

    def forward(runner, batch, *args, **kwargs):
        control_path = os.environ.get('CHAIN_AUDIT_CONTROL')
        control = json.loads(Path(control_path).read_text()) if control_path and Path(control_path).exists() else None
        pre = runner.model.pre_executor
        if control and active.get('identity') != control['identity']:
            active.clear()
            active.update(control)
            runner.chain_validate = control.get('validate_mapping', False)
            if pre.has_separate_hidden_state:
                parameter = next(pre.parameters())
                table = make_table(pre, parameter.device, parameter.dtype, control['seed'])
                if hasattr(pre, '_audit_table'):
                    pre._audit_table.copy_(table)
                else:
                    pre._audit_table = table
                    pre._audit_observed = table.clone()
                with open(control['capture'], 'a') as stream:
                    stream.write(json.dumps({'kind': 'initial_table', 'seed': control['seed'], 'hash': digest(table)}) + '\n')
        result = original(runner, batch, *args, **kwargs)
        if control and control.get('capture_logits') and getattr(batch, 'recurrent_role', None) is None:
            logits = result.logits_output.next_token_logits[-1].float()
            values, indices = logits.topk(2)
            with open(control['capture'], 'a') as stream:
                stream.write(json.dumps({'kind': 'logits', 'position': int(batch.positions[-1]),
                                         'top1': int(indices[0]), 'top2': int(indices[1]),
                                         'margin': float(values[0] - values[1])}) + '\n')
        return result

    ModelRunner.forward = forward
