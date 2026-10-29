# Main-experiment protocol

Base tasks contain HumanEval+ 164, MBPP+ 378, GSM8K 1,319 and MATH-500 500
examples. GSM8K's **1,024 is the output token budget**, not the sample count.
HumanEval+ and MBPP+ also use 1,024 output tokens. Base MATH-500 uses 2,048,
except Raven-TinyLlama, which uses 1,024 within its 2,048-token context.

Each Thinking model uses 30 AIME 2024, 30 AIME 2025, 64 BeyondAIME and 64
MATH-500 examples, all at an 8,192-token output budget. The package stores frozen
source-row IDs and normalized-problem hashes; it never draws a new random subset.
The Thinking tokenizer must render the official `<think>\n` assistant prefix.

Base prompts reproduce the original three-shot GSM8K and zero-shot code/math
formats. Huginn wraps these with its original system/user chat protocol. Base
prompt hashes and input-ID hashes are checked against the complete historical
manifests. Task text and targets are obtained from the pinned upstream downloads.
`prepare --offline` requires previously downloaded, checksum-verified assets.

Generation is greedy, seed 0, BF16, one request at a time, Triton attention and
CUDA graphs. There is one screen checkpoint and no second branching depth.
The native stochastic initialization of Raven/Huginn is preserved; it is not an
output-identity guarantee. BF16 execution differences can change outputs across
AR and CopyLoop. Report lengths and quality together with speed.

## Timing

The primary statistic is `sum(decode_tokens) / sum(decode_seconds)` across all
requests in a cell. Speedup is method TPS divided by paired AR TPS. It is neither
mean per-request TPS nor an end-to-end latency ratio.

Native counters count committed output IDs through the stop boundary, minus the
first completion token. Rejected speculative candidates are excluded. EOS or
stop-boundary tokens count when retained in those output IDs. The runner checks
`decode_tokens == len(output_ids) - 1` for every request.

Timing excludes model loading, graph capture, explicit warmup, prefill and the
first output token. AR uses host result-processing boundaries. CopyLoop starts
after the first committed-token callback and ends after synchronized executor
completion and host output enqueue; decision readback and final synchronization
are within that measured path. These are host elapsed times, not CUDA-event
kernel-only timings. HTTP time is recorded separately and is not used for TPS.
The original timer implementations are retained unchanged.

One unscored eight-token warmup precedes measured requests. Metrics are consumed
before the next request begins. Each measured request must have exactly one
native metrics record. Failed/partial runs have no `DONE.json` and are rejected
by scoring and aggregation. There is no automatic retry or silent mixing of
attempts; retain failures and use a new output directory. Summaries reject
duplicate methods, unmatched task IDs, prompts, budgets or model presets.

## Scoring

HumanEval+/MBPP+ use the complete EvalPlus tests and the historical sanitation and
stop-prefix removal rule. Candidate workers use the original Linux
Landlock/seccomp restrictions; failure aborts evaluation. The retained EvalPlus
`find_zero` progress-accounting repair does not change its polynomial oracle.

GSM8K uses the retained flexible numeric extraction/exact-match scorer.
MATH-500 uses math-verify 0.9.0. AIME uses strict integer extraction; BeyondAIME
uses the same rule without AIME's 0--999 bound. Thinking scoring uses token-ID
segmentation and only the final answer. Unfinished thinking, malformed tags,
empty/unparseable final answers and budget-limited responses stay in the
denominator. Scorer infrastructure failures abort instead of dropping examples.

Every run stores prepared inputs, configuration, command, runtime versions, raw
responses, committed token IDs, native counters, warmup and per-request HTTP time.
The summary also reports total output tokens and budget hits. Scoring reports
unfinished Thinking separately from budget hits.
