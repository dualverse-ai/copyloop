# CopyLoop: Fast Looped Transformer Decoding via Context Copying and Shallow Screening

**Wenyu Du, Stephen Chung**

**Status:** Submitted. **Paper URL:** `xxx`.

## Abstract

Looped Transformers increase computational capacity through the repeated application of shared blocks, but their recurrent depth makes conventional full-depth decoding expensive. While existing speculative pipelines accelerate generation by processing consecutive token positions at varying depths, they are typically limited by sequential decoding overheads. We introduce CopyLoop, a training-free decoding algorithm that combines low-cost context-copy proposals with shallow screening and continuous recurrent scheduling. Rather than step-by-step generation, CopyLoop proposes multi-token suffixes from repeated context. It then evaluates them early at a shallow recurrent depth to efficiently filter out incorrect branches, processing newer blocks concurrently while earlier blocks are still progressing toward full-depth verification. We evaluate CopyLoop across six model checkpoints from the Ouro, Raven, and Huginn families, covering code generation and multi-step mathematical reasoning. CopyLoop achieves decoding speedups of 2.98x--10.70x over full-depth autoregressive decoding on Base models and 2.74x--3.52x on Thinking models, consistently outperforming or closely matching all competing baselines across evaluation tasks. Furthermore, ablation studies confirm a strong synergy between shallow screening and continuous scheduling, while our analyses show that CopyLoop requires only lightweight model calibration and yields greater speedups on longer generations with reusable contexts.

Title, author order and abstract were supplied by the authors. The public paper
URL is intentionally set to `xxx` while the submission is pending. The reported speedups above are
paper results, not measurements produced during repository packaging.
