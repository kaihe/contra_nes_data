# Cloud trace worker bootstrap

Status: Implemented

## Decision

Provision an Ubuntu trace-search worker with `deploy/setup_cloud_worker.sh` and
the minimal dependencies in `deploy/requirements-mc.txt`. The script is
idempotent: it clones or fast-forwards the repository, preserves its virtual
environment, installs the project, imports a user-supplied Contra ROM, and runs
an emulator smoke test.

The repository never contains or downloads the copyrighted ROM. The operator
must set `CONTRA_ROM_PATH` to a legally obtained ROM whose hash stable-retro
recognizes. Repository URL, checkout path, Python binary, and PyPI mirror remain
environment overrides so the same script works on Gitee and other cloud hosts.

The bootstrap deliberately does not rewrite Ubuntu package mirrors, edit shell
startup files, delete environments, start an unbounded search, or install the
old agent's unrelated PPO stack. Search launch and output synchronization are
explicit operator actions after the smoke test passes.
