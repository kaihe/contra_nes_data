"""Phase A: per-class entity positions in HF export JSON."""

import numpy as np

from env.entity import HEATMAP_CLASSES
from task_maker.export_hf import (
    _xy_list,
    entities_frame,
    verify_entities_vs_centroids,
)


def test_xy_list_shapes():
    assert _xy_list(np.zeros((0, 2), dtype=np.int16)) == []
    assert _xy_list(np.array([10, 20], dtype=np.int16)) == [[10, 20]]
    assert _xy_list(np.array([[1, 2], [3, 4]], dtype=np.int16)) == [[1, 2], [3, 4]]


def test_entities_frame_keys_and_player_shape():
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[0x0334] = 100   # player x
    ram[0x031A] = 80    # player y
    # one live enemy (non-bullet) in slot 0
    ram[0x04B8] = 1
    ram[0x033E] = 50
    ram[0x0324] = 60
    ram[0x0528] = 0x10
    # one enemy bullet in slot 1
    ram[0x04B8 + 1] = 1
    ram[0x033E + 1] = 70
    ram[0x0324 + 1] = 90
    ram[0x0528 + 1] = 0x01
    # one player bullet
    ram[0x0388] = 1
    ram[0x03C8] = 110
    ram[0x03B8] = 120
    ram[0x0448] = 0

    frame = entities_frame(ram)
    assert set(frame) == set(HEATMAP_CLASSES)
    assert frame["player"] == [[100, 80]]
    assert frame["enemies"] == [[50, 60]]
    assert frame["enemy_bullets"] == [[70, 90]]
    assert frame["player_bullets"] == [[110, 120]]


def test_verify_centroids_subset_of_enemy_slots():
    entities = {
        "player": [[[10, 10]], [[10, 10]], [[10, 10]]],
        "player_bullets": [[], [], []],
        "enemies": [[[50, 60], [1, 2]], [[50, 60]], []],
        "enemy_bullets": [[], [], [[56, 135]]],
    }
    centroids = [[[50, 60]], [[50, 60]], [[56, 135]]]
    visibility = [True, True, True]
    # frame 2: item slot briefly typed as bullet — still counts as a match
    checked, bad = verify_entities_vs_centroids(
        centroids, visibility, entities, goal_when="first")
    assert checked == 3 and bad == 0

    centroids_bad = [[[50, 60]], [[99, 99]], [[56, 135]]]
    checked, bad = verify_entities_vs_centroids(
        centroids_bad, visibility, entities, goal_when="first")
    assert checked == 3 and bad == 1


def test_verify_skips_traverse_goals():
    entities = {"enemies": [[[1, 1]]]}
    checked, bad = verify_entities_vs_centroids(
        [[[9, 9]]], [True], entities, goal_when="last")
    assert checked == 0 and bad == 0
