import numpy as np

from datahouse.global_motion import (align_frame, estimate_translation, luminance,
                                     overlap, trimmed_score)


def test_luminance_uses_declared_integer_weights():
    rgb = np.asarray([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]], dtype=np.uint8)
    assert luminance(rgb).tolist() == [[77 * 255 >> 8, 150 * 255 >> 8,
                                       29 * 255 >> 8]]


def test_overlap_and_alignment_follow_previous_to_current_sign():
    previous = np.arange(30, dtype=np.uint8).reshape(5, 6)
    aligned, mask = align_frame(previous, -2, 1)
    left, right = overlap(previous, aligned, -2, 1)
    assert np.array_equal(left, right)
    assert mask.sum() == 4 * 4
    assert np.all(aligned[0] == 0)
    assert np.all(aligned[:, -2:] == 0)


def test_coarse_to_fine_recovers_translation_despite_moving_outlier():
    rng = np.random.default_rng(7)
    gray = rng.integers(0, 256, size=(64, 64), dtype=np.uint8)
    previous = np.repeat(gray[..., None], 3, axis=2)
    current, _ = align_frame(previous, -3, 2)
    current[25:31, 40:46] = 255
    estimate = estimate_translation(previous, current, max_shift=8)
    assert (estimate["dx"], estimate["dy"]) == (-3, 2)
    assert estimate["score"] < trimmed_score(luminance(previous), luminance(current), 0, 0)
