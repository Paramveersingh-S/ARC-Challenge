import numpy as np
from src.dsl import Hypothesis
from src.executor import apply_hypothesis

def test_flip_h():
    grid = np.array([
        [1, 2],
        [3, 4]
    ])
    hyp = Hypothesis(id="1", description="", condition="", transform="flip_h")
    result = apply_hypothesis(hyp, grid)
    assert np.array_equal(result, np.array([[2, 1], [4, 3]]))

def test_recolor():
    grid = np.array([
        [1, 1],
        [0, 1]
    ])
    hyp = Hypothesis(id="2", description="", condition="", transform="recolor(1, 2)")
    result = apply_hypothesis(hyp, grid)
    assert np.array_equal(result, np.array([[2, 2], [0, 2]]))
