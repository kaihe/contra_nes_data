# Deploy an MC-trace worker

This runbook provisions an Ubuntu 22.04/24.04 CPU machine for Contra MC search.
The ROM and generated traces remain outside Git.

## 1. Clone from Gitee

Log in to the worker and clone the branch containing the deployment tooling:

```bash
mkdir -p /root/code
git clone --branch feat/cloud-trace-worker \
  https://gitee.com/kaihe_2020/contra_nes_data.git \
  /root/code/contra_nes_data
cd /root/code/contra_nes_data
```

After this feature branch is merged into `main`, omit `--branch`.

## 2. Transfer the ROM

Run this on the workstation, not the cloud worker:

```bash
set -a
source .env
set +a
ssh -p "$CLOUD_SSH_PORT" "$CLOUD_SSH_USER@$CLOUD_SSH_HOST" \
  'mkdir -p /root/roms && chmod 700 /root/roms'
scp -P "$CLOUD_SSH_PORT" \
  /home/kaihe/code/contra_agent/contra/integration/Contra-Nes/rom.nes \
  "$CLOUD_SSH_USER@$CLOUD_SSH_HOST:/root/roms/Contra.nes"
```

Expected ROM SHA-256:

```text
26541a5550ee22deeb3d5484e4a96130219b58cff74d068fb1eb6567fa5e5519
```

Do not add the ROM to Git or upload it to Gitee.

## 3. Configure and bootstrap

Create `/root/code/contra_nes_data/.env` on the worker:

```dotenv
CONTRA_REPO_URL=https://gitee.com/kaihe_2020/contra_nes_data.git
CONTRA_PROJECT_DIR=/root/code/contra_nes_data
CONTRA_PYTHON=python3
CONTRA_PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
CONTRA_ROM_PATH=/root/roms/Contra.nes
```

Load it and run the idempotent bootstrap:

```bash
cd /root/code/contra_nes_data
set -a
source .env
set +a
bash deploy/setup_cloud_worker.sh
```

Success ends with `Cloud worker ready: Contra-Nes`. Re-running the script updates
the checkout with `git pull --ff-only` and reuses `.venv`.

## 4. Start Level 1 generation

The current fast full-level configuration is:

```bash
cd /root/code/contra_nes_data
source .venv/bin/activate
mkdir -p game_trace/mc_trace/level1
python -u -m agent.mc_search \
  --level 1 --runs 40000 \
  --rollouts 16 --rollout-len 24 --settle-margin 8 \
  --max-rewind 15 --workers "$(nproc)" \
  --max-time 600 --max-actions 6000 \
  --goal level_up --no-verbose \
  2>&1 | tee game_trace/mc_trace/level1/search.log
```

Run it inside `tmux` or a systemd service if it must survive SSH disconnects.
Outputs appear in `game_trace/mc_trace/level1/` and are ignored by Git.

## 5. Operate and retrieve

```bash
# Count winning traces on the worker
find game_trace/mc_trace/level1 -maxdepth 1 -name '*.npz' | wc -l

# Copy results back from the workstation
rsync -av --partial -e "ssh -p $CLOUD_SSH_PORT" \
  "$CLOUD_SSH_USER@$CLOUD_SSH_HOST:/root/code/contra_nes_data/game_trace/mc_trace/level1/" \
  game_trace/mc_trace/level1/
```

Use `Ctrl-C` for a clean stop. Before terminating a cloud instance, copy all
trace files back and verify the local count.
