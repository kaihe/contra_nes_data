#!/usr/bin/env bash
# Bootstrap an Ubuntu 22.04/24.04 machine for Contra MC-trace generation.
#
# Required:
#   CONTRA_ROM_PATH=/secure/path/to/Contra.nes
# Optional:
#   CONTRA_REPO_URL=git@gitee.com:kaihe_2020/contra_nes_data.git
#   CONTRA_PROJECT_DIR=$HOME/code/contra_nes_data
#   CONTRA_PYTHON=python3
#   CONTRA_PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

set -Eeuo pipefail

repo_url="${CONTRA_REPO_URL:-git@gitee.com:kaihe_2020/contra_nes_data.git}"
project_dir="${CONTRA_PROJECT_DIR:-${HOME}/code/contra_nes_data}"
python_bin="${CONTRA_PYTHON:-python3}"
pip_index="${CONTRA_PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
rom_path="${CONTRA_ROM_PATH:-}"

if [[ -z "${rom_path}" || ! -f "${rom_path}" ]]; then
    echo "CONTRA_ROM_PATH must name a readable, legally obtained Contra ROM file." >&2
    exit 2
fi

if ! command -v sudo >/dev/null 2>&1; then
    echo "sudo is required to install Ubuntu system packages." >&2
    exit 2
fi

sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential git python3-dev python3-pip python3-venv

if [[ -d "${project_dir}/.git" ]]; then
    git -C "${project_dir}" pull --ff-only
elif [[ -e "${project_dir}" ]]; then
    echo "${project_dir} exists but is not a Git checkout; refusing to overwrite it." >&2
    exit 2
else
    mkdir -p "$(dirname "${project_dir}")"
    git clone "${repo_url}" "${project_dir}"
fi

if [[ ! -x "${project_dir}/.venv/bin/python" ]]; then
    "${python_bin}" -m venv "${project_dir}/.venv"
fi

venv_python="${project_dir}/.venv/bin/python"
"${venv_python}" -m pip install --upgrade pip setuptools wheel \
    --index-url "${pip_index}"
"${venv_python}" -m pip install \
    --index-url "${pip_index}" \
    -r "${project_dir}/deploy/requirements-mc.txt"
"${venv_python}" -m pip install --no-deps -e "${project_dir}"

# stable-retro's importer scans a directory and imports recognized ROM hashes.
"${venv_python}" -m stable_retro.import "$(dirname "${rom_path}")"

"${venv_python}" - <<'PY'
from pathlib import Path

import stable_retro as retro

from agent.mc_search import make_search_env
from util.replay import GAME, INTTYPE

rom = Path(retro.data.get_romfile_path(GAME, INTTYPE))
if not rom.is_file():
    raise SystemExit(f"stable-retro did not import the required {GAME} ROM")
env = make_search_env(1, retro.Observations.RAM)
try:
    state = env.em.get_state()
    if not state:
        raise SystemExit("Level 1 initial state is empty")
finally:
    env.close()
print(f"Cloud worker ready: {GAME}, Level 1 state bytes={len(state)}")
PY

echo "Activate with: source ${project_dir}/.venv/bin/activate"
echo "The bootstrap does not start a search; choose the run target explicitly."
