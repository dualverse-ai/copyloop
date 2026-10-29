# Installation and environment

Use a new Python 3.11 environment. Install `requirements-gpu.txt`, then
`pip install -e '.[data,eval]'`. Code scoring additionally requires Linux
Landlock, `libseccomp.so.2`, and the `prlimit` command from util-linux.

Set `CUDA_VISIBLE_DEVICES` to one GPU explicitly before generation. The CLI does
not reserve devices, launch background queues, or inspect a station registry.
Use separate ports and output directories if running concurrent processes.
Large recorded KV pools, especially Raven's, may not fit smaller GPUs; the main
presets target the original hardware. Changing resource limits requires a new
paired AR/CopyLoop measurement.

The measured source stack was Python 3.11.17, torch 2.11.0+cu130,
Triton 3.6.0, SGLang 0.5.14, Transformers 5.8.1 and FlashInfer 0.6.12.
Configure the CUDA 13 toolkit and compiler for your installation, for example
`CUDA_HOME`, `PATH` and `LD_LIBRARY_PATH`. No paths from the research host are
embedded in the launcher. The pip NVIDIA toolkit layout may use `lib` rather than
`lib64`; verify your toolchain rather than assuming the system CUDA installation
matches PyTorch.

Before reporting measurements, run `python -m pip check`, record `pip freeze`,
the GPU/driver and toolkit versions, and complete small AR and CopyLoop inference
runs plus scoring. 

The CLI core can be inspected without those dependencies:

```bash
python -m pip install -e . --no-deps
copyloop --help
copyloop download --model ouro-1.4b --output models/ouro14 --dry-run
python -m unittest discover -s tests -v
```
