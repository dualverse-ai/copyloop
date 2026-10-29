# CopyLoop — attribution and licensing

Copyright 2026 Wenyu Du and Stephen Chung.

Original CopyLoop contributions are licensed under the Apache License, Version
2.0, provided in `LICENSE`. This choice applies to contributions the CopyLoop
authors own; it does not grant a new license to third-party source code.

## Recurrent-serving backend

The serving implementation derives from
[LoopSpec](https://github.com/kaist-flexml-lab/loopspec), pinned at commit
`41aa710ed7c9ce908cffa03597be5d1df19d7b56`, with subsequent local CopyLoop changes.
This attribution applies to the extraction snapshots in `backends/` and the
runtime variants in `src/copyloop/_backends/`. The original and copied source
hashes are recorded in `SOURCE_MANIFEST.json` and `PACKAGE_SOURCE_MAP.json`.

## Dependencies, prompts and assets

The runtime uses SGLang, PyTorch, Triton, Transformers and FlashInfer, which retain
their respective licenses. Evaluation uses EvalPlus and math-verify. GSM8K
prompt examples and extraction semantics follow the recorded
lm-evaluation-harness 0.4.12 protocol. See `licenses/README.md`.

Model weights and benchmark records are downloaded from their original providers
and retain their respective licenses. They are not included in the source
distribution. The small GSM8K few-shot prompt template is included as part of the
experiment protocol.

The layout reference was `dualverse-ai/lowbit-ngram`; its model implementations
and project-specific third-party notices are not copied into this repository.
