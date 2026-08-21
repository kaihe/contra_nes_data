"""SQLite catalog for immutable, taxonomy-addressed datahouse token shards."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS shards (
  id INTEGER PRIMARY KEY,
  path TEXT NOT NULL UNIQUE,
  sha256 TEXT NOT NULL UNIQUE,
  level INTEGER NOT NULL CHECK(level BETWEEN 1 AND 8),
  task TEXT NOT NULL CHECK(task IN ('boss', 'kill', 'full')),
  weapon TEXT NOT NULL,
  encoder_sha256 TEXT NOT NULL,
  ordinal INTEGER NOT NULL CHECK(ordinal >= 0),
  episodes INTEGER NOT NULL CHECK(episodes > 0),
  frames INTEGER NOT NULL CHECK(frames >= 0),
  UNIQUE(level, task, weapon, encoder_sha256, ordinal)
);
CREATE TABLE IF NOT EXISTS episodes (
  fingerprint TEXT PRIMARY KEY,
  uid TEXT NOT NULL UNIQUE,
  source_trace TEXT NOT NULL,
  action_steps INTEGER NOT NULL CHECK(action_steps > 0)
);
CREATE TABLE IF NOT EXISTS shard_episodes (
  shard_id INTEGER NOT NULL REFERENCES shards(id) ON DELETE RESTRICT,
  fingerprint TEXT NOT NULL REFERENCES episodes(fingerprint) ON DELETE RESTRICT,
  ordinal INTEGER NOT NULL CHECK(ordinal >= 0),
  PRIMARY KEY(shard_id, fingerprint),
  UNIQUE(shard_id, ordinal)
);
CREATE INDEX IF NOT EXISTS shard_selection ON shards
  (level, task, weapon, encoder_sha256, ordinal);
CREATE TABLE IF NOT EXISTS episode_boundaries (
  fingerprint TEXT PRIMARY KEY REFERENCES episodes(fingerprint) ON DELETE RESTRICT,
  observation_steps INTEGER NOT NULL CHECK(observation_steps > 1),
  boss_observation_index INTEGER NOT NULL,
  source_gcs_uri TEXT NOT NULL,
  source_sha256 TEXT NOT NULL,
  CHECK(boss_observation_index BETWEEN 0 AND observation_steps - 1)
);
CREATE TABLE IF NOT EXISTS collections (
  name TEXT PRIMARY KEY,
  level INTEGER NOT NULL CHECK(level BETWEEN 1 AND 8),
  task TEXT NOT NULL CHECK(task IN ('boss', 'kill', 'full')),
  encoder_sha256 TEXT NOT NULL,
  manifest_path TEXT NOT NULL UNIQUE,
  manifest_sha256 TEXT NOT NULL,
  episodes INTEGER NOT NULL CHECK(episodes > 0)
);
CREATE TABLE IF NOT EXISTS collection_episodes (
  collection_name TEXT NOT NULL REFERENCES collections(name) ON DELETE RESTRICT,
  fingerprint TEXT NOT NULL REFERENCES episodes(fingerprint) ON DELETE RESTRICT,
  ordinal INTEGER NOT NULL CHECK(ordinal >= 0),
  PRIMARY KEY(collection_name, fingerprint),
  UNIQUE(collection_name, ordinal)
);
"""


@dataclass(frozen=True)
class Shard:
    path: str
    sha256: str
    level: int
    task: str
    weapon: str
    encoder_sha256: str
    ordinal: int
    episodes: int
    frames: int


def connect(path: str | Path) -> sqlite3.Connection:
    """Open and initialise a catalog, safe for policy's read-only query path."""
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.executescript(SCHEMA)
    return db


def register_shard(db: sqlite3.Connection, shard: Shard,
                   episodes: Iterable[tuple[str, str, str, int]]) -> None:
    """Atomically register a hash-verified tar and all its episode provenance.

    Each episode tuple is ``(fingerprint, uid, source_trace, action_steps)``.
    A fingerprint cannot enter a second shard: duplicated episode data is rejected
    by the producer instead of being discovered later by a policy experiment.
    """
    rows = list(episodes)
    if len(rows) != shard.episodes:
        raise ValueError(f"shard says {shard.episodes} episodes, received {len(rows)}")
    with db:
        cursor = db.execute(
            "INSERT INTO shards(path,sha256,level,task,weapon,encoder_sha256,ordinal,episodes,frames) "
            "VALUES(?,?,?,?,?,?,?,?,?)",
            (shard.path, shard.sha256, shard.level, shard.task, shard.weapon,
             shard.encoder_sha256, shard.ordinal, shard.episodes, shard.frames))
        shard_id = int(cursor.lastrowid)
        for ordinal, (fingerprint, uid, source_trace, action_steps) in enumerate(rows):
            existing = db.execute("SELECT fingerprint FROM shard_episodes WHERE fingerprint=?",
                                  (fingerprint,)).fetchone()
            if existing:
                raise ValueError(f"episode already belongs to another shard: {fingerprint}")
            db.execute("INSERT INTO episodes(fingerprint,uid,source_trace,action_steps) VALUES(?,?,?,?)",
                       (fingerprint, uid, source_trace, action_steps))
            db.execute("INSERT INTO shard_episodes(shard_id,fingerprint,ordinal) VALUES(?,?,?)",
                       (shard_id, fingerprint, ordinal))


def select_shards(db: sqlite3.Connection, *, level: int, task: str, weapon: str,
                  encoder_sha256: str, episode_budget: int) -> list[Shard]:
    """Return the smallest deterministic whole-shard prefix meeting a budget."""
    if episode_budget < 1:
        raise ValueError("episode_budget must be positive")
    rows = db.execute(
        "SELECT path,sha256,level,task,weapon,encoder_sha256,ordinal,episodes,frames "
        "FROM shards WHERE level=? AND task=? AND weapon=? "
        "AND encoder_sha256=? ORDER BY ordinal",
        (level, task, weapon, encoder_sha256)).fetchall()
    selected, total = [], 0
    for row in rows:
        selected.append(Shard(**dict(row)))
        total += int(row["episodes"])
        if total >= episode_budget:
            return selected
    raise ValueError(f"catalog has {total} matching episodes, below requested {episode_budget}")


def register_boundaries(db: sqlite3.Connection,
                        rows: Iterable[tuple[str, int, int, str, str]]) -> None:
    """Record full-episode observation counts, boss boundaries, and raw identity."""
    with db:
        db.executemany(
            "INSERT INTO episode_boundaries(fingerprint,observation_steps,"
            "boss_observation_index,source_gcs_uri,source_sha256) VALUES(?,?,?,?,?)",
            list(rows))


def register_collection(db: sqlite3.Connection, *, name: str, level: int,
                        task: str, encoder_sha256: str, manifest_path: str,
                        manifest_sha256: str, fingerprints: Iterable[str]) -> None:
    """Atomically publish ordered collection membership after all shards exist."""
    members = list(fingerprints)
    with db:
        db.execute(
            "INSERT INTO collections(name,level,task,encoder_sha256,manifest_path,"
            "manifest_sha256,episodes) VALUES(?,?,?,?,?,?,?)",
            (name, level, task, encoder_sha256, manifest_path, manifest_sha256,
             len(members)))
        db.executemany(
            "INSERT INTO collection_episodes(collection_name,fingerprint,ordinal) "
            "VALUES(?,?,?)",
            ((name, fingerprint, ordinal)
             for ordinal, fingerprint in enumerate(members)))
