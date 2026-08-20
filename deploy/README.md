# Deploy an MC-trace worker

This runbook provisions an Ubuntu 22.04/24.04 CPU machine for Contra MC search.
The ROM and generated traces remain outside Git.

## 1. Clone from Gitee

Log in to the worker and clone the branch containing the deployment tooling:

```bash
mkdir -p /root/code
git clone --branch feat/cloud-worker-loop \
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

## 4. Connect Google Cloud Storage

Create a dedicated GCS bucket and grant the worker service account Storage
Object Creator and Storage Object Viewer on that bucket. The worker uses Google
Application Default Credentials (ADC). On a non-GCP machine, place its credential
configuration outside the repository and point ADC to it:

```bash
install -d -m 700 /root/.config/contra
install -m 600 /secure/source/worker-credential.json \
  /root/.config/contra/gcs-worker.json
export GOOGLE_APPLICATION_CREDENTIALS=/root/.config/contra/gcs-worker.json
```

Prefer Workload Identity Federation over a long-lived service-account key when
the cloud provider supports it. Never put either credential form in Git or `.env`.

## 5. Start the persistent Level 1 worker

The current fast full-level configuration is:

```bash
cd /root/code/contra_nes_data
source .venv/bin/activate
mkdir -p game_trace/worker_spool
python -u -m worker.search_loop \
  --gcs-root "gs://BUCKET/contra-mc-tracehouse/schema-v1/level1/full" \
  --spool-dir game_trace/worker_spool \
  --level 1 \
  --rollouts 16 --rollout-len 24 --settle-margin 8 \
  --max-rewind 15 --workers "$(nproc)" \
  --max-time 600 --max-actions 6000 \
  --goal level_up \
  2>&1 | tee game_trace/worker_spool/search.log
```

Run it inside `tmux` or a systemd service. Every win is saved locally at once.
At 100 wins the worker seals and uploads the batch while search continues into
the next one. Restarting the same command resumes its open and sealed batches.

## 6. Stop, inspect, and retire

```bash
# Count locally spooled traces
find game_trace/worker_spool -path '*/traces/*.npz' | wc -l

# Permanently retire this worker and upload its final partial batch
python -m worker.search_loop \
  --gcs-root "gs://BUCKET/contra-mc-tracehouse/schema-v1/level1/full" \
  --spool-dir game_trace/worker_spool --flush
```

`Ctrl-C` stops after the active search returns and leaves a partial batch open
for the next launch. Use `--flush` only before permanently deleting the worker.
Do not delete the spool until the corresponding GCS acknowledgement exists.

## 7. Import existing traces

Legacy NPZs use the same 100-trace protocol through a finite pseudo-search loop.
Quote the glob so Python—not the shell—expands it. Missing boss loadout metadata
is recovered by replay; source NPZs remain byte-identical and are hard-linked
into the spool when both paths are on the same filesystem.

```bash
python -u -m worker.legacy_import \
  --gcs-root "gs://BUCKET/contra-mc-tracehouse/schema-v1/level1/full" \
  --spool-dir game_trace/legacy_upload_spool/level1-full \
  --worker-id local-legacy-level1-full \
  --trace-glob 'game_trace/mc_trace/level1/*.npz'
```

Use a separate spool and matching GCS prefix for each homogeneous collection,
for example `level1/boss` and `level2/full`. Restarting the same command skips
journaled traces, resumes uploads, and finishes the existing open batch. Once
all sources are consumed, the importer explicitly uploads the final partial
batch.

Canonical bulk collections may provide an exact catalog-selected source list
and use larger archives. Boss archives use 1,000 traces and record their scope
explicitly in every manifest row:

```bash
python -u -m worker.legacy_import \
  --gcs-root "gs://BUCKET/contra-mc-tracehouse/schema-v1/level1/boss" \
  --spool-dir game_trace/legacy_upload_spool/level1-boss-80k \
  --worker-id level1-boss-canonical-80k \
  --trace-list /path/to/catalog-selection.txt \
  --batch-size 1000 \
  --trace-scope boss_fight
```
