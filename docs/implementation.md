# Implementation map

The package selects a private backend root and starts `sglang_recurrent.server`
in a separate process. This preserves the tested module imports without installing
four competing copies of the same namespace globally. Package-relative paths
work from a wheel as well as an editable checkout.

The four variants correspond to Ouro-1.4B/Raven, Ouro-2.6B, Huginn and Thinking.
Their graph bucket lists, pool sizing and model adapters differ. They are retained
until a unified backend can be checked on GPUs; deduplicating them is not part of
this packaging pass. Algorithm files have not been rewritten.

Within each variant:

- `loopspec/lookup_chain.py`: lookup candidates and copied tails.
- `loopspec/chain_pipeline.py`: chain construction, shallow truncation and branches.
- `loopspec/executor.py`: recurrent execution and verified commitment.
- `loopspec/request.py`: request state and committed output.
- `modeling/`: recurrent model adapters and serving state.
- `server/cli.py`, `server/loopspec_runner.py`: construction and resource allocation.
- `server/metrics.py`, scheduler files: native decode accounting.

Historical optional paths (including second-depth and WaveFront internals) remain
in the frozen serving files because removing them would change source structure.
They have no public experiment entry points in this release. The public launcher
clears experimental control variables and selects only single-checkpoint CopyLoop
or AR. Runtime audit support is bundled to close transitive imports; its optional
control files are disabled by the launcher.

`SOURCE_MANIFEST.json` verifies the untouched extraction snapshots.
`PACKAGE_SOURCE_MAP.json` identifies the runtime/scorer copies and adaptations.
No paper result numbers are inferred from the new launcher. Numerical fidelity
and performance after packaging still require GPU smoke and paired measurement.
