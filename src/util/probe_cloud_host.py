"""Classify a cloud host from `lscpu` before bootstrap.

Level 2 throughput on the production workers tracked CPU model, not RAM or
steal. Probe over SSH and keep or drop without running search:

    python -m util.probe_cloud_host CLOUD7
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

# Observed Level 2 wins/hour from doc/0011, same 64/48/8/30 search.
CPU_CLASSES = (
    ("Platinum 8468", "keep", "99-143 wins/hour on 8-core workers"),
    ("Platinum 8352V", "keep", "80 wins/hour on 8-core workers"),
    ("Gold 6133", "drop", "35-53 wins/hour on 8-core workers"),
    ("E5-2698 v4", "drop", "55 wins/hour on 8-core workers"),
)

REMOTE = r"""
python3 - << 'PY'
import json, os
from pathlib import Path

cpu = {}
for line in os.popen("lscpu").read().splitlines():
    if ":" in line:
        key, value = line.split(":", 1)
        cpu[key.strip()] = value.strip()
stat = Path("/proc/stat").read_text().splitlines()[0].split()
keys = ["user", "nice", "system", "idle", "iowait", "irq", "softirq", "steal"]
cpu_stat = {key: int(value) for key, value in zip(keys, stat[1:1 + len(keys)])}
mem_kb = None
for line in Path("/proc/meminfo").read_text().splitlines():
    if line.startswith("MemTotal:"):
        mem_kb = int(line.split()[1])
        break
print(json.dumps({
    "cpu_model": cpu.get("Model name"),
    "nproc": os.cpu_count(),
    "mem_gib": None if mem_kb is None else round(mem_kb / (1024 * 1024), 1),
    "steal_frac": cpu_stat["steal"] / max(1, sum(cpu_stat.values())),
    "loadavg": os.getloadavg(),
}))
PY
"""


def classify(cpu_model: str | None, nproc: int | None) -> dict:
    """Return keep, drop, or unknown from the hardware fields we can read immediately."""
    if not nproc or nproc < 8:
        return {
            "decision": "drop",
            "cpu_class": None,
            "reason": f"need 8 logical CPUs, host has {nproc}",
        }
    model = cpu_model or ""
    for needle, decision, reason in CPU_CLASSES:
        if needle in model:
            return {"decision": decision, "cpu_class": needle, "reason": reason}
    return {
        "decision": "unknown",
        "cpu_class": None,
        "reason": f"unseen CPU model {cpu_model!r}; do not bootstrap until classified",
    }


def load_worker_ssh(name: str, env_path: Path) -> dict:
    env = {}
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key] = value
    prefix = name.upper()
    try:
        return {
            "host": env[f"{prefix}_SSH_HOST"],
            "port": int(env[f"{prefix}_SSH_PORT"]),
            "user": env[f"{prefix}_SSH_USER"],
            "password": env[f"{prefix}_SSH_PASSWORD"],
        }
    except KeyError as exc:
        raise SystemExit(f"missing {exc} in {env_path}") from exc


def probe_ssh(host: str, port: int, user: str, password: str) -> dict:
    import paramiko
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=host, port=port, username=user, password=password,
                   timeout=20, banner_timeout=20, auth_timeout=20)
    try:
        _, stdout, stderr = client.exec_command(REMOTE, timeout=20)
        out = stdout.read().decode()
        err = stderr.read().decode()
        rc = stdout.channel.recv_exit_status()
    finally:
        client.close()
    if rc != 0:
        raise RuntimeError(err.strip() or f"remote probe failed with status {rc}")
    return json.loads(out)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Keep or drop a cloud host from lscpu before bootstrap")
    parser.add_argument("worker", nargs="?", help="CLOUD6-style name from .env")
    parser.add_argument("--env", default=None)
    args = parser.parse_args()
    if not args.worker:
        raise SystemExit("usage: python -m util.probe_cloud_host CLOUD7")
    repo = Path(__file__).resolve().parents[2]
    env_path = Path(args.env) if args.env else repo / ".env"
    ssh = load_worker_ssh(args.worker, env_path)
    facts = probe_ssh(ssh["host"], ssh["port"], ssh["user"], ssh["password"])
    result = classify(facts.get("cpu_model"), facts.get("nproc"))
    result.update(worker=args.worker.upper(), **facts)
    json.dump(result, sys.stdout, indent=2)
    sys.stdout.write("\n")
    if result["decision"] == "keep":
        raise SystemExit(0)
    if result["decision"] == "drop":
        raise SystemExit(2)
    raise SystemExit(3)


if __name__ == "__main__":
    main()
