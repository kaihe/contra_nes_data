#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="${CONTRA_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
python_bin="${CONTRA_PYTHON:-${repo_root}/.venv/bin/python}"
catalog="${CONTRA_CATALOG:-${repo_root}/game_trace/datahouse/catalog.sqlite}"
trace_root="${CONTRA_BOSS_TRACE_ROOT:-${repo_root}/game_trace/mc_trace/boss_level1}"
gcs_root="${GCS_BOSS_ROOT:-gs://contra_nes_trace/contra-mc-tracehouse/schema-v1/level1/boss}"
spool_root="${CONTRA_BOSS_EXTRA_SPOOL:-${repo_root}/game_trace/legacy_upload_spool/level1-boss-extra-v1}"

declare -A expected=(
  [spread]=10538
  [laser]=549
  [regular]=550
  [flamethrower]=384
)

command -v sqlite3 >/dev/null || { echo "missing command: sqlite3" >&2; exit 2; }
command -v zstd >/dev/null || { echo "missing command: zstd" >&2; exit 2; }
if [[ "${python_bin}" == */* ]]; then
  test -x "${python_bin}" || { echo "missing Python: ${python_bin}" >&2; exit 2; }
else
  command -v "${python_bin}" >/dev/null || { echo "missing Python: ${python_bin}" >&2; exit 2; }
fi
test -r "${catalog}" || { echo "missing catalog: ${catalog}" >&2; exit 2; }
test -d "${trace_root}" || { echo "missing trace root: ${trace_root}" >&2; exit 2; }
: "${GOOGLE_APPLICATION_CREDENTIALS:?set GOOGLE_APPLICATION_CREDENTIALS to the GCS uploader key}"

canonical_selection() {
  local weapon="$1"
  sqlite3 "${catalog}" "
    SELECT '${trace_root}/' || e.source_trace
    FROM shards s
    JOIN shard_episodes se ON se.shard_id = s.id
    JOIN episodes e ON e.fingerprint = se.fingerprint
    WHERE s.level = 1 AND s.task = 'boss' AND s.weapon = '${weapon}'
    ORDER BY s.ordinal, se.ordinal;"
}

validate_extra() {
  local weapon="$1"
  local all selected extra
  all="$(find "${trace_root}" -maxdepth 1 -type f -name "*_${weapon}_*.npz" | wc -l)"
  selected="$(canonical_selection "${weapon}" | wc -l)"
  extra=$((all - selected))
  [[ "${extra}" == "${expected[${weapon}]}" ]] || {
    echo "${weapon}: expected ${expected[${weapon}]} extra traces, found ${extra}" >&2
    exit 2
  }
}

run_import() {
  local weapon="$1"
  local spool="${spool_root}/${weapon}"
  mkdir -p "${spool}"
  "${python_bin}" -u -m worker.legacy_import \
    --gcs-root "${gcs_root}" \
    --spool-dir "${spool}" \
    --worker-id "level1-boss-extra-${weapon}-v1" \
    --trace-glob "${trace_root}/*_${weapon}_*.npz" \
    --exclude-list <(canonical_selection "${weapon}") \
    --batch-size 1000 \
    --trace-scope boss_fight \
    --collection-id level1-boss-extra-v1 \
    >>"${spool}/import.log" 2>&1
}

pids=()
for weapon in spread laser regular flamethrower; do
  validate_extra "${weapon}"
  run_import "${weapon}" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=$?
done
exit "${status}"
