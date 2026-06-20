import numpy as np
from src.dsl import Hypothesis
from src.executor import apply_hypothesis

def test_tile_2x2():
    grid = np.array([
        [1, 2],
        [3, 4]
    ])
    hyp = Hypothesis(id="t1", description="", condition="", transform="tile_2x2")
    res = apply_hypothesis(hyp, grid)
    assert res is not None
    assert res.shape == (4, 4)
    assert res[0, 0] == 1 and res[0, 2] == 1

def test_gravity_down():
    grid = np.array([
        [0, 1],
        [0, 0],
        [2, 0]
    ])
    hyp = Hypothesis(id="t2", description="", condition="", transform="gravity_down")
    res = apply_hypothesis(hyp, grid)
    assert res is not None
    assert res[2, 0] == 2
    assert res[2, 1] == 1
    assert res[0, 1] == 0
