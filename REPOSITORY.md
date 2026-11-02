# Repository provenance and data availability

This repository combines the existing `C:\projects\VPD` research workspace with
the ancestry of `goodfire-ai/param-decomp`. The initial MoralityLabAI commit is
based on upstream commit `177d0cce809896d9d84b2bd04bbc2e319c64e9d0`.

The upstream tree is retained unchanged under `upstream/goodfire/`, including its
MIT license, source, tests, documentation and package configuration. Keeping it
there preserves both projects' packaging files and avoids applying Goodfire's
root test configuration to this workspace's tests. Work on Goodfire's original
package from that directory, following its README. The root `pyproject.toml`
continues to configure the MoralityLabAI authorization and experiment packages.

The adapted Nanbeige VPD implementation has its own earlier provenance record in
`experiments/nanbeige_looped/vpd_backend_provenance.json` and retains Goodfire's
license notice. The current upstream snapshot does not change which implementation
produced the archived experiments.

## Included

- Research source, configuration, fixtures, tests and execution scripts.
- Versioned paper sources, PDFs, figures, compact supplements and verification records.
- Small VPD reports/specifications and the original and extended dataset indices.
- State-carry report, numerical checks and experiment closeout.

## Local data dependencies

Model weights, raw activation archives, downloaded datasets, runtime binaries,
virtual environments, cloud launch records and temporary work are excluded.
The dataset index describes measurements held in those local archives; a Git
clone alone does not supply every indexed asset. Index SHA256 values identify
the expected files and are not a claim that those files are distributed here.
The manuscript supplements preserve historical paths and manifests unchanged.
Git attributes disable automatic newline conversion so archived SHA256 checks
remain valid when this repository is cloned on another platform.

Some analysis and regression tests read local experiment artifacts. Run them
with the corresponding artifacts available; a missing artifact must not be
interpreted as an experimental failure or replaced with fabricated data. The
dependency-light authorization tests can be run with the root package installed.

The latest manuscript is selected by `output/paper_versions/index.json`.
Numbered manuscript versions are immutable. Future edits should create the next
version and update the archive pointer.

## Remotes

- `origin`: `https://github.com/MoralityLabAI/VPD.git`
- `upstream`: `https://github.com/goodfire-ai/param-decomp.git`

The initial local checkout fetches the upstream tip shallowly. GitHub retains the
fork's original ancestry; use `git fetch --unshallow upstream` if full local
upstream history is needed. Future upstream updates require a deliberate update
of the vendored tree rather than overwriting this workspace's root configuration.
