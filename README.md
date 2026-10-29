# CopyLoop: Fast Looped Transformer Decoding via Context Copying and Shallow Screening

**Wenyu Du and Stephen Chung**

**Paper:** `xxx` (submitted; public link pending)

CopyLoop accelerates recurrent language-model decoding by copying candidate tails
from context, screening them at an early recurrent depth, and verifying the
surviving continuation at full depth.

The paper reports **2.98×–10.70× decode-throughput speedups on Base models**
and **2.74×–3.52× on Thinking models**, relative to full-depth AR across six
checkpoints from Ouro, Raven and Huginn. See the [paper abstract](docs/paper.md).

This repository contains the single-checkpoint decoder and scripts for its main
benchmarks. AR is included as the throughput reference. The public command line
exposes AR and CopyLoop; calibration sweeps, adaptive controllers, ablations and
multi-GPU campaign schedulers are outside this release.

| Model | Recurrent depth | Screen depth | Copy limit K | Main benchmarks |
|---|---:|---:|---:|---|
| Ouro-1.4B | 4 | 1 | 6 | HumanEval+, MBPP+, GSM8K, MATH-500 |
| Ouro-2.6B | 4 | 1 | 6 | HumanEval+, MBPP+, GSM8K, MATH-500 |
| Raven-TinyLlama | 32 | 4 | 6 | HumanEval+, MBPP+, GSM8K, MATH-500 |
| Huginn-3.5B | 32 | 8 | 6 | HumanEval+, MBPP+, GSM8K, MATH-500 |
| Ouro-1.4B-Thinking | 4 | 1 | 2 | AIME 2024/2025, BeyondAIME, MATH-500 |
| Ouro-2.6B-Thinking | 4 | 1 | 2 | AIME 2024/2025, BeyondAIME, MATH-500 |

## Install

Use Linux and Python 3.11 in a fresh environment. The research environment used
CUDA 13, BF16, Triton attention and one active request on an RTX 5090. Runtime
dependencies are listed in `requirements-gpu.txt`. See [installation](docs/installation.md).

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-gpu.txt
python -m pip install -e '.[data,eval]'
copyloop presets
```

The CLI itself has no GPU dependencies. `pip install -e .` suffices for help,
configuration inspection, and dry-run generation plans.

## Reproduce a task

Download weights and datasets from their original hosts. This repository ships
neither weights nor benchmark records. The small GSM8K few-shot template is part
of the prompt protocol. Downloads use pinned revisions and dataset checksums.

```bash
copyloop download --model ouro-1.4b --output models/ouro-1.4b
copyloop prepare --model ouro-1.4b --model-path models/ouro-1.4b \
  --dataset humaneval --assets data/assets --output data/ouro14-he.json

# Inspect before launching; this command needs no GPU.
copyloop generate --prepared data/ouro14-he.json --method copyloop \
  --output outputs/ouro14-he-copyloop --dry-run

# Run sequentially on a GPU available to you. Start with --limit 2 at prepare time.
CUDA_VISIBLE_DEVICES=0 copyloop generate --prepared data/ouro14-he.json \
  --method ar --output outputs/ouro14-he-ar
CUDA_VISIBLE_DEVICES=0 copyloop generate --prepared data/ouro14-he.json \
  --method copyloop --output outputs/ouro14-he-copyloop
copyloop evaluate --run outputs/ouro14-he-ar
copyloop evaluate --run outputs/ouro14-he-copyloop
copyloop summarize --runs outputs/ouro14-he-ar outputs/ouro14-he-copyloop \
  --output outputs/ouro14-he-summary.json
```

Replace the model and dataset arguments for other cells. Thinking example:

```bash
copyloop download --model ouro-1.4b-thinking --output models/ouro-1.4b-thinking
copyloop prepare --model ouro-1.4b-thinking --model-path models/ouro-1.4b-thinking \
  --dataset aime2024 --output data/ouro14-thinking-aime24.json
```

Thinking uses the official template with thinking enabled, checkpoint 1, **K=2**
and an 8,192-token output limit. The same generation/evaluation commands apply.
Details of task sizes, scoring and timing are in [experiments](docs/experiments.md).

## Layout

- `src/copyloop/`: CLI, data preparation, generation, scoring and aggregation.
- `src/copyloop/presets/`: model defaults, pinned assets, task IDs and prompt hashes.
- `src/copyloop/_backends/`: serving code, with the measured backend variants preserved.
- `src/copyloop/_support/`: transitive runtime support required by those backends.
- `backends/`: untouched extraction snapshots for provenance, excluded from wheels.
- `SOURCE_MANIFEST.json`, `PACKAGE_SOURCE_MAP.json`: original and packaged source provenance.
- `tests/`: CPU contracts for configuration, prompt construction and measurement.

The backend keeps the historical `sglang_recurrent` namespace internally.
See [implementation](docs/implementation.md) for the algorithm entry points.

## Citation

If you use CopyLoop, please cite:

```bibtex
@misc{du_chung_copyloop,
  title = {{CopyLoop}: Fast Looped Transformer Decoding via Context Copying and Shallow Screening},
  author = {Du, Wenyu and Chung, Stephen},
  note = {Submitted},
  url = {xxx}
}
```

The paper link `xxx` is an intentional placeholder while the submission is
pending. Machine-readable metadata is available
in [CITATION.cff](CITATION.cff).

## License and acknowledgments

Original CopyLoop contributions are licensed under [Apache-2.0](LICENSE). The
recurrent-serving backend builds on [LoopSpec](https://github.com/kaist-flexml-lab/loopspec)
and SGLang. See [NOTICE.md](NOTICE.md) for third-party attribution and license scope.
