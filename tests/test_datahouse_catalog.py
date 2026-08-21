from datahouse.catalog import Shard, connect, register_shard, select_shards


def test_catalog_selects_whole_shard_prefix_and_tracks_trace(tmp_path):
    db = connect(tmp_path / "catalog.sqlite")
    base = dict(level=1, task="boss", weapon="spread", encoder_sha256="encoder")
    register_shard(db, Shard(path="one.tar", sha256="one", ordinal=0,
                             episodes=2, frames=10, **base),
                   [("a", "a", "trace-a.npz", 4), ("b", "b", "trace-b.npz", 6)])
    register_shard(db, Shard(path="two.tar", sha256="two", ordinal=1,
                             episodes=2, frames=12, **base),
                   [("c", "c", "trace-c.npz", 5), ("d", "d", "trace-d.npz", 7)])
    selected = select_shards(db, episode_budget=3, **base)
    assert [shard.path for shard in selected] == ["one.tar", "two.tar"]
    assert sum(shard.episodes for shard in selected) == 4
    assert db.execute("SELECT source_trace FROM episodes WHERE uid='c'").fetchone()[0] == "trace-c.npz"
