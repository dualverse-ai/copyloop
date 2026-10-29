# CopyLoop Release Source Snapshot

This directory is the first extraction pass for a standalone CopyLoop repository. It contains exact backend snapshots used by the paper's six main-table models. No runtime code has been edited in this pass.

| Release snapshot | Main-table models | Source |
|---|---|---|
| `backends/ouro14/` | Ouro-1.4B Base, Raven-TinyLlama | `scripts/base_main_single/backends/ouro14/backend_chain/` |
| `backends/ouro26/` | Ouro-2.6B Base | `scripts/base_main_single/backends/ouro26/backend_chain/` |
| `backends/huginn/` | Huginn-3.5B | `scripts/base_main_single/backends/huginn/backend_chain/` |
| `backends/thinking/` | Ouro-1.4B Thinking, Ouro-2.6B Thinking | `scripts/thinking_main_completion_20261027/backends/thinking_chain/` |

Raven-TinyLlama resolves to the same Ouro-1.4B backend path in the frozen Base main manifest. The backend is model-config driven; it is not a separate Raven-only implementation.

## Scope

These directories preserve the original serving-backend extraction. The installable
package and public command-line interface are under `src/copyloop/`. Source paths
and per-file SHA-256 digests are recorded in `SOURCE_MANIFEST.json`.

Do not edit these snapshots in place. Make release changes in a separate working copy and retain the manifest as the provenance baseline.

The local extraction manifest records 223 files, including nine historical Python
bytecode cache files. Source distributions exclude these nine caches and include
the 214 source/configuration snapshot files. The original manifest is retained
unchanged as an extraction record; use the distribution inventory for packaged
file checks. The working installable implementation is under `src/copyloop/`.
