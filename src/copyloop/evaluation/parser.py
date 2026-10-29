"""Strict token-ID reasoning/final segmentation. Never infer tags from decoded text."""
from dataclasses import asdict, dataclass
from typing import Callable

@dataclass
class Parsed:
    prompt_thinking: bool
    final_ids: list[int]
    thinking_ids: list[int]
    final_text: str
    thinking_text: str
    malformed: bool
    errors: list[str]
    unfinished_think: bool
    empty_final: bool
    eos: bool
    eos_index: int | None
    final_start_index: int | None
    close_index: int | None
    total_tokens: int
    thinking_tokens: int
    final_tokens: int
    control_tokens: int
    token_zero_count: int
    cap_hit: bool
    length_reached_budget: bool
    truncated_no_eos: bool
    eligible: bool
    trailing_ids: list[int]
    trailing_tokens: int
    accounting_conserved: bool

    def to_dict(self):
        return asdict(self)

def parse_tokens(output_ids: list[int], *, prompt_thinking: bool,
                 decode: Callable[[list[int]], str], max_new_tokens: int,
                 open_id: int = 3, close_id: int = 4, eos_id: int = 2) -> Parsed:
    if len({open_id, close_id, eos_id}) != 3:
        raise ValueError('control token IDs must be distinct')
    if max_new_tokens <= 0:
        raise ValueError('max_new_tokens must be positive')
    state = 'think' if prompt_thinking else 'final'
    opened = prompt_thinking
    closed = False
    think, final, errors = [], [], []
    eos_index = final_start = close_index = None
    controls = 0
    trailing = []
    for i, token in enumerate(output_ids):
        if eos_index is not None:
            errors.append('tokens_after_eos')
            trailing = list(output_ids[i:])
            break
        if token == eos_id:
            eos_index = i
            controls += 1
        elif token == open_id:
            controls += 1
            if opened or closed:
                errors.append('nested_or_repeated_open')
            elif final and decode(final).strip():
                errors.append('text_before_generated_open')
            elif final:
                # Whitespace before a generated opening tag is formatting, not an answer.
                think.extend(final)
                final.clear()
                final_start = None
            opened = True
            state = 'think'
        elif token == close_id:
            controls += 1
            if state != 'think' or closed:
                errors.append('unmatched_or_repeated_close')
            else:
                close_index = i
            closed = True
            state = 'final'
        elif state == 'think':
            think.append(token)
        else:
            if final_start is None:
                final_start = i
            final.append(token)
    if len(output_ids) > max_new_tokens:
        errors.append('output_exceeds_budget')
    final_text = decode(final)
    think_text = decode(think)
    malformed = bool(errors)
    unfinished = state == 'think'
    empty_final = not final_text.strip()
    return Parsed(prompt_thinking, final, think, final_text, think_text,
                  malformed, sorted(set(errors)), unfinished, empty_final,
                  eos_index is not None, eos_index, final_start, close_index,
                  len(output_ids), len(think), len(final), controls,
                  output_ids.count(0), len(output_ids) >= max_new_tokens,
                  len(output_ids) >= max_new_tokens,
                  len(output_ids) >= max_new_tokens and eos_index is None,
                  not malformed and not unfinished and not empty_final,
                  trailing, len(trailing),
                  len(output_ids) == len(think)+len(final)+controls+len(trailing))


def prompt_initial_state(prompt_ids: list[int], *, open_id=3, close_id=4) -> bool:
    """Inspect assistant suffix only: user text can legitimately contain tag IDs.

    Caller must supply rendered assistant-prefix IDs, not whole conversation.
    """
    opens = prompt_ids.count(open_id)
    closes = prompt_ids.count(close_id)
    if opens == 0 and closes == 0:
        return False
    if opens == 1 and closes == 0:
        return True
    raise ValueError('unsupported assistant prefix tag state')
