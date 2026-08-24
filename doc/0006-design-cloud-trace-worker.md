# Cloud trace worker bootstrap

Status: Implemented

## Decision

Provision an Ubuntu trace-search worker with `deploy/setup_cloud_worker.sh` and
the minimal dependencies in `deploy/requirements-mc.txt`. Screen the host first
with `python -m util.probe_cloud_host CLOUDN`: `lscpu` and `nproc` are enough to
keep Platinum 8468 / 8352V or drop Gold 6133 and E5-2698 v4, matching the
Level 2 fleet in `doc/0011-exp-level2-search-efficiency.md`. Do not bootstrap a
`drop` host.
An unseen CPU model is `unknown` until it is classified. Same-SKU hosts can
still differ after keep (cloud3 vs cloud6), so search throughput ranks keepers.

The bootstrap script is idempotent: it clones or fast-forwards the repository,
preserves its virtual environment, installs the project, imports a user-supplied
Contra ROM, and runs an emulator smoke test.

The repository never contains or downloads the copyrighted ROM. The operator
must set `CONTRA_ROM_PATH` to a legally obtained ROM whose hash stable-retro
recognizes. Repository URL, checkout path, Python binary, and PyPI mirror remain
environment overrides so the same script works on Gitee and other cloud hosts.

The checkout already contains the committed action priors under
`src/agent/priors/`. New workers do not copy human recordings or rebuild a
prior from GCS; they search with the frozen YAML. Refreshing a prior is an
operator snapshot (`python -m util.build_action_prior --gcs-root …`), not
part of bootstrap.

The bootstrap deliberately does not rewrite Ubuntu package mirrors, edit shell
startup files, delete environments, start an unbounded search, or install the
old agent's unrelated PPO stack. Search launch and output synchronization are
explicit operator actions after the smoke test passes.
