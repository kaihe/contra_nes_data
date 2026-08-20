#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="${CONTRA_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
python_bin="${CONTRA_PYTHON:-${repo_root}/.venv/bin/python}"
catalog="${CONTRA_CATALOG:-${repo_root}/game_trace/datahouse/catalog.sqlite}"
trace_root="${CONTRA_BOSS_TRACE_ROOT:-${repo_root}/game_trace/mc_trace/boss_level1}"
gcs_root="${GCS_BOSS_ROOT:-gs://contra_nes_trace/contra-mc-tracehouse/schema-v1/level1/boss}"
spool_root="${CONTRA_BOSS_UPLOAD_SPOOL:-${repo_root}/game_trace/legacy_upload_spool/level1-boss-80k}"

for command in sqlite3 zstd; do
  command -v "${command}" >/dev/null || { echo "missing command: ${command}" >&2; exit 2; }
done
test -r "${catalog}" || { echo "missing catalog: ${catalog}" >&2; exit 2; }
test -d "${trace_root}" || { echo "missing trace root: ${trace_root}" >&2; exit 2; }
if [[ "${python_bin}" == */* ]]; then
  test -x "${python_bin}" || { echo "missing Python: ${python_bin}" >&2; exit 2; }
else
  command -v "${python_bin}" >/dev/null || { echo "missing Python: ${python_bin}" >&2; exit 2; }
fi
: "${GOOGLE_APPLICATION_CREDENTIALS:?set GOOGLE_APPLICATION_CREDENTIALS to the GCS uploader key}"

selection_sql() {
  local weapon="$1"
  sqlite3 "${catalog}" "
    SELECT '${trace_root}/' || e.source_trace
    FROM shards s
    JOIN shard_episodes se ON se.shard_id = s.id
    JOIN episodes e ON e.fingerprint = se.fingerprint
    WHERE s.level = 1 AND s.task = 'boss' AND s.weapon = '${weapon}'
    ORDER BY s.ordinal, se.ordinal;"
}

validate_selection() {
  local weapon="$1" count missing
  count="$(selection_sql "${weapon}" | wc -l)"
  [[ "${count}" == 40000 ]] || {
    echo "${weapon}: expected 40000 catalog episodes, found ${count}" >&2
    exit 2
  }
  missing="$(selection_sql "${weapon}" | while IFS= read -r path; do
    [[ -f "${path}" ]] || echo "${path}"
  done | wc -l)"
  [[ "${missing}" == 0 ]] || {
    echo "${weapon}: ${missing} selected source traces are missing" >&2
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
    --worker-id "level1-boss-${weapon}-canonical-40k" \
    --trace-list <(selection_sql "${weapon}") \
    --batch-size 1000 \
    --trace-scope boss_fight \
    --collection-id level1-boss-canonical-80k-v1 \
    >>"${spool}/import.log" 2>&1
}

validate_selection laser
validate_selection spread

run_import laser &
laser_pid=$!
run_import spread &
spread_pid=$!

status=0
wait "${laser_pid}" || status=$?
wait "${spread_pid}" || status=$?
exit "${status}"
