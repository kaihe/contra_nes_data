import numpy as np
import pytest

from datahouse.views import view_bounds


@pytest.mark.parametrize("view,obs,actions", [
    ("start_to_boss", (0, 4), (0, 3)),
    ("boss_fight", (3, 11), (3, 10)),
    ("full", (0, 11), (0, 10)),
])
def test_view_bounds_are_aligned_and_partial_actions_do_not_overlap(view, obs, actions):
    observation_slice, action_slice = view_bounds(view, 10, 3)
    assert (observation_slice.start, observation_slice.stop) == obs
    assert (action_slice.start, action_slice.stop) == actions
    assert len(np.arange(11)[observation_slice]) == len(np.arange(10)[action_slice]) + 1


def test_reader_requires_one_scalar_view():
    with pytest.raises(ValueError, match="exactly one"):
        view_bounds(("full", "boss_fight"), 10, 3)
