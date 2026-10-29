"""Position-indexed Raven randomness and graph-external reference diagnostics."""
import hashlib
import json
import os
from pathlib import Path

import torch

PADDING_CALLS = 0


def register_padding(module, width):
    def before(layer, inputs):
        global PADDING_CALLS
        layer._audit_padded_rows = None
        if inputs and inputs[0].ndim == 2 and 0 < inputs[0].shape[0] < width:
            values = inputs[0]
            count = values.shape[0]
            padded = torch.cat((values, values.new_zeros((width - count, values.shape[1]))), dim=0)
            layer._audit_padded_rows = count
            PADDING_CALLS += 1
            return (padded, *inputs[1:])

    def after(layer, inputs, output):
        count = layer._audit_padded_rows
        if count is None:
            return output
        if isinstance(output, tuple):
            return (output[0][:count], *output[1:])
        return output[:count]

    module.register_forward_pre_hook(before)
    module.register_forward_hook(after)


def install_padding(width):
    from sglang.srt.layers.linear import LinearBase
    original = LinearBase.__init__

    def initialize(module, *args, **kwargs):
        original(module, *args, **kwargs)
        register_padding(module, width)

    LinearBase.__init__ = initialize


def digest(value):
    return hashlib.sha256(value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def make_table(executor, device, dtype, seed):
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(seed)
        return executor.make_initial_hidden(8192, device=device, dtype=dtype)


def initial(executor, positions, hidden):
    if not hasattr(executor, '_audit_table'):
        executor._audit_table = make_table(executor, hidden.device, hidden.dtype, 0)
        executor._audit_observed = torch.empty_like(executor._audit_table)
    values = executor._audit_table.index_select(0, positions.long())
    executor._audit_observed.index_copy_(0, positions.long(), values)
    return values


def install():
    from sglang.srt.model_executor.model_runner import ModelRunner
    if getattr(ModelRunner, '_raven_audit_installed', False):
        return
    ModelRunner._raven_audit_installed = True
    padding_width = int(os.environ.get('RAVEN_AUDIT_LINEAR_PADDING', '0'))
    if padding_width:
        install_padding(padding_width)
    original = ModelRunner.forward
    active = {}

    def forward(runner, forward_batch, *args, **kwargs):
        control_file = os.environ.get('RAVEN_AUDIT_CONTROL')
        control = json.loads(Path(control_file).read_text()) if control_file and Path(control_file).exists() else None
        executor = runner.model.pre_executor
        if control and active.get('identity') != control['identity']:
            if padding_width and not PADDING_CALLS:
                raise AssertionError('Linear shape perturbation was not invoked during graph warmup')
            table = make_table(executor, executor._audit_table.device, executor._audit_table.dtype, control['seed'])
            executor._audit_table.copy_(table)
            active.clear()
            active.update(identity=control['identity'], table_hash=digest(table))
            with open(control['capture'], 'a') as stream:
                stream.write(json.dumps({'kind': 'initial_table', 'seed': control['seed'], 'hash': active['table_hash'], 'shape': list(table.shape),
                                         'linear_padding_width': padding_width, 'padding_calls_during_eager_or_capture': PADDING_CALLS}) + '\n')
        result = original(runner, forward_batch, *args, **kwargs)
        if not control:
            return result
        role = getattr(forward_batch, 'recurrent_role', None)
        output = result.logits_output
        if role in (None, 'pre'):
            positions = forward_batch.positions.long()
            if role == 'pre':
                positions = output.positions.long()
            expected = executor._audit_table.index_select(0, positions)
            actual = executor._audit_observed.index_select(0, positions)
            if not torch.equal(expected, actual):
                raise AssertionError('Initial-state graph replay does not match request/position table')
            if role == 'pre' and not torch.equal(expected, output.hidden):
                raise AssertionError('LoopSpec pre output differs from coupled initial state')
            with open(control['capture'], 'a') as stream:
                stream.write(json.dumps({'kind': 'initial_check', 'count': len(positions), 'equal': True,
                                         'role': role, 'positions': positions.cpu().tolist()}) + '\n')
        if role is None and control['capture_logits']:
            logits = output.next_token_logits[-1].float()
            values, indices = logits.topk(2)
            row = {'kind': 'logits', 'position': int(forward_batch.positions[-1]),
                   'top1': int(indices[0]), 'top2': int(indices[1]),
                   'margin': float(values[0] - values[1]), 'logits_hash': digest(logits),
                   'initial_equal': True}
            with open(control['capture'], 'a') as stream:
                stream.write(json.dumps(row) + '\n')
        return result

    ModelRunner.forward = forward
