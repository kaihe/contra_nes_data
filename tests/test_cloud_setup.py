import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "deploy" / "setup_cloud_worker.sh"


def test_cloud_setup_is_valid_bash_and_keeps_rom_external():
    assert os.access(SCRIPT, os.X_OK)
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    source = SCRIPT.read_text()
    assert "CONTRA_ROM_PATH" in source
    assert "stable_retro.import" in source
    assert "git -C \"${project_dir}\" pull --ff-only" in source
    assert '[[ "${EUID}" -eq 0 ]]' in source
    assert "rm -rf" not in source
    assert ".bashrc" not in source


def test_cloud_requirements_are_mc_only():
    requirements = (ROOT / "deploy" / "requirements-mc.txt").read_text()
    assert "stable-retro==" in requirements
    assert "numpy==" in requirements
    assert "PyYAML==" in requirements
    assert "torch" not in requirements
    assert "stable-baselines" not in requirements
