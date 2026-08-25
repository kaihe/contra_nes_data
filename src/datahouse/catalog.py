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
CREATE TABLE IF NOT EXISTS frame_shards (
  id INTEGER PRIMARY KEY,
  path TEXT NOT NULL UNIQUE,
  sha256 TEXT NOT NULL UNIQUE,
  level INTEGER NOT NULL CHECK(level BETWEEN 1 AND 8),
  task TEXT NOT NULL CHECK(task IN ('boss', 'kill', 'full')),
  weapon TEXT NOT NULL,
  format TEXT NOT NULL,
  frame_height INTEGER NOT NULL CHECK(frame_height > 0),
  frame_width INTEGER NOT NULL CHECK(frame_width > 0),
  ordinal INTEGER NOT NULL CHECK(ordinal >= 0),
  episodes INTEGER NOT NULL CHECK(episodes > 0),
  frames INTEGER NOT NULL CHECK(frames >= 0),
  UNIQUE(level, task, weapon, format, ordinal)
);
CREATE TABLE IF NOT EXISTS frame_shard_episodes (
  shard_id INTEGER NOT NULL REFERENCES frame_shards(id) ON DELETE RESTRICT,
  fingerprint TEXT NOT NULL REFERENCES episodes(fingerprint) ON DELETE RESTRICT,
  ordinal INTEGER NOT NULL CHECK(ordinal >= 0),
  PRIMARY KEY(shard_id, fingerprint),
  UNIQUE(shard_id, ordinal)
);
CREATE INDEX IF NOT EXISTS frame_shard_selection ON frame_shards
  (level, task, weapon, format, ordinal);
CREATE TABLE IF NOT EXISTS feature_shards (
  id INTEGER PRIMARY KEY,
  path TEXT NOT NULL UNIQUE,
  sha256 TEXT NOT NULL UNIQUE,
  level INTEGER NOT NULL CHECK(level BETWEEN 1 AND 8),
  task TEXT NOT NULL CHECK(task IN ('boss', 'kill', 'full')),
  weapon TEXT NOT NULL,
  representation TEXT NOT NULL,
  encoder_sha256 TEXT NOT NULL,
  boundary TEXT NOT NULL,
  dtype TEXT NOT NULL,
  channels INTEGER NOT NULL CHECK(channels > 0),
  feature_height INTEGER NOT NULL CHECK(feature_height > 0),
  feature_width INTEGER NOT NULL CHECK(feature_width > 0),
  ordinal INTEGER NOT NULL CHECK(ordinal >= 0),
  episodes INTEGER NOT NULL CHECK(episodes > 0),
  frames INTEGER NOT NULL CHECK(frames >= 0),
  UNIQUE(level, task, weapon, representation, encoder_sha256, ordinal)
);
CREATE TABLE IF NOT EXISTS feature_shard_episodes (
  shard_id INTEGER NOT NULL REFERENCES feature_shards(id) ON DELETE RESTRICT,
  fingerprint TEXT NOT NULL REFERENCES episodes(fingerprint) ON DELETE RESTRICT,
  ordinal INTEGER NOT NULL CHECK(ordinal >= 0),
  PRIMARY KEY(shard_id, fingerprint),
  UNIQUE(shard_id, ordinal)
);
CREATE INDEX IF NOT EXISTS feature_shard_selection ON feature_shards
  (level, task, weapon, representation, encoder_sha256, ordinal);
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


@dataclass(frozen=True)
class FrameShard:
    """One published raw-frame tar. ``format`` plays the role ``encoder_sha256``
    plays for token shards: the representation discriminator that must not be mixed
    inside a single training set."""

    path: str
    sha256: str
    level: int
    task: str
    weapon: str
    format: str
    frame_height: int
    frame_width: int
    ordinal: int
    episodes: int
    frames: int


@dataclass(frozen=True)
class FeatureShard:
    """One immutable intermediate-feature tar.

    ``representation`` versions the producer boundary and serialization contract;
    ``encoder_sha256`` keeps features from different frozen weights disjoint.
    """

    path: str
    sha256: str
    level: int
    task: str
    weapon: str
    representation: str
    encoder_sha256: str
    boundary: str
    dtype: str
    channels: int
    feature_height: int
    feature_width: int
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


def register_frame_shard(db: sqlite3.Connection, shard: FrameShard,
                         fingerprints: Iterable[str]) -> None:
    """Atomically register a hash-verified frame tar against existing episodes.

    Unlike :func:`register_shard` this never creates ``episodes`` rows. A frame shard
    republishes episodes the token producer already cataloged, and referencing those
    same rows is what makes episode-set identity between the two releases a join
    rather than a convention. An unknown fingerprint is therefore an error: it means
    the frame release drifted from the token release.
    """
    rows = list(fingerprints)
    if len(rows) != shard.episodes:
        raise ValueError(f"shard says {shard.episodes} episodes, received {len(rows)}")
    if len(set(rows)) != len(rows):
        raise ValueError("duplicate fingerprint within one frame shard")
    with db:
        for fingerprint in rows:
            if db.execute("SELECT 1 FROM episodes WHERE fingerprint=?",
                          (fingerprint,)).fetchone() is None:
                raise ValueError(f"episode is not cataloged: {fingerprint}")
            clash = db.execute(
                "SELECT s.ordinal FROM frame_shard_episodes fse "
                "JOIN frame_shards s ON s.id = fse.shard_id "
                "WHERE fse.fingerprint=? AND s.level=? AND s.task=? AND s.weapon=? "
                "AND s.format=?",
                (fingerprint, shard.level, shard.task, shard.weapon,
                 shard.format)).fetchone()
            if clash is not None:
                raise ValueError(f"episode already in frame shard {clash[0]}: {fingerprint}")
        cursor = db.execute(
            "INSERT INTO frame_shards(path,sha256,level,task,weapon,format,"
            "frame_height,frame_width,ordinal,episodes,frames) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (shard.path, shard.sha256, shard.level, shard.task, shard.weapon,
             shard.format, shard.frame_height, shard.frame_width, shard.ordinal,
             shard.episodes, shard.frames))
        shard_id = int(cursor.lastrowid)
        for ordinal, fingerprint in enumerate(rows):
            db.execute("INSERT INTO frame_shard_episodes(shard_id,fingerprint,ordinal) "
                       "VALUES(?,?,?)", (shard_id, fingerprint, ordinal))


def token_prefix_fingerprints(db: sqlite3.Connection, *, level: int, task: str,
                              weapon: str, shard_count: int) -> list[str]:
    """Fingerprints of the first ``shard_count`` token shards, in consumer order.

    This reproduces how ``contra_nes_policy`` selects a training set — order by
    ordinal, take a whole-shard prefix — so a frame release can be pinned to exactly
    the episodes an existing baseline was measured on.
    """
    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    available = int(db.execute(
        "SELECT COUNT(*) FROM shards WHERE level=? AND task=? AND weapon=?",
        (level, task, weapon)).fetchone()[0])
    if available < shard_count:
        raise ValueError(f"catalog has {available} {weapon} shards, asked for {shard_count}")
    return [str(row[0]) for row in db.execute(
        "SELECT fse.fingerprint FROM shard_episodes fse "
        "JOIN shards s ON s.id = fse.shard_id "
        "WHERE s.level=? AND s.task=? AND s.weapon=? AND s.ordinal < ? "
        "ORDER BY s.ordinal, fse.ordinal",
        (level, task, weapon, shard_count)).fetchall()]


def frame_shard_fingerprints(db: sqlite3.Connection, *, level: int, task: str,
                             weapon: str, format: str) -> set[str]:
    """Fingerprints already published as frames, for resumable builds."""
    return {str(row[0]) for row in db.execute(
        "SELECT fse.fingerprint FROM frame_shard_episodes fse "
        "JOIN frame_shards s ON s.id = fse.shard_id "
        "WHERE s.level=? AND s.task=? AND s.weapon=? AND s.format=?",
        (level, task, weapon, format)).fetchall()}


def register_feature_shard(db: sqlite3.Connection, shard: FeatureShard,
                           fingerprints: Iterable[str]) -> None:
    """Register a feature tar that republishes existing episode membership."""
    rows = list(fingerprints)
    if len(rows) != shard.episodes:
        raise ValueError(f"shard says {shard.episodes} episodes, received {len(rows)}")
    if len(set(rows)) != len(rows):
        raise ValueError("duplicate fingerprint within one feature shard")
    with db:
        for fingerprint in rows:
            if db.execute("SELECT 1 FROM episodes WHERE fingerprint=?",
                          (fingerprint,)).fetchone() is None:
                raise ValueError(f"episode is not cataloged: {fingerprint}")
            clash = db.execute(
                "SELECT s.ordinal FROM feature_shard_episodes fse "
                "JOIN feature_shards s ON s.id=fse.shard_id "
                "WHERE fse.fingerprint=? AND s.level=? AND s.task=? AND s.weapon=? "
                "AND s.representation=? AND s.encoder_sha256=?",
                (fingerprint, shard.level, shard.task, shard.weapon,
                 shard.representation, shard.encoder_sha256)).fetchone()
            if clash is not None:
                raise ValueError(
                    f"episode already in feature shard {clash[0]}: {fingerprint}")
        cursor = db.execute(
            "INSERT INTO feature_shards(path,sha256,level,task,weapon,representation,"
            "encoder_sha256,boundary,dtype,channels,feature_height,feature_width,"
            "ordinal,episodes,frames) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (shard.path, shard.sha256, shard.level, shard.task, shard.weapon,
             shard.representation, shard.encoder_sha256, shard.boundary, shard.dtype,
             shard.channels, shard.feature_height, shard.feature_width, shard.ordinal,
             shard.episodes, shard.frames))
        shard_id = int(cursor.lastrowid)
        db.executemany(
            "INSERT INTO feature_shard_episodes(shard_id,fingerprint,ordinal) "
            "VALUES(?,?,?)",
            ((shard_id, fingerprint, ordinal)
             for ordinal, fingerprint in enumerate(rows)))


def feature_shard_fingerprints(db: sqlite3.Connection, *, level: int, task: str,
                               weapon: str, representation: str,
                               encoder_sha256: str) -> set[str]:
    """Fingerprints already published for one exact feature producer."""
    return {str(row[0]) for row in db.execute(
        "SELECT fse.fingerprint FROM feature_shard_episodes fse "
        "JOIN feature_shards s ON s.id=fse.shard_id "
        "WHERE s.level=? AND s.task=? AND s.weapon=? AND s.representation=? "
        "AND s.encoder_sha256=?",
        (level, task, weapon, representation, encoder_sha256)).fetchall()}


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
