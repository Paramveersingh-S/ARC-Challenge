import numpy as np
from typing import List
from src.dsl import Hypothesis
from src.executor import apply_hypothesis

def update_beliefs(hypotheses: List[Hypothesis],
                   test_grid: np.ndarray,
                   true_output: np.ndarray) -> List[Hypothesis]:
    """
    After observing true_output for test_grid:
    - Hypotheses consistent with truth: increase confidence
    - Inconsistent ones: mark falsified
    """
    for hyp in hypotheses:
        if hyp.falsified: continue
        predicted = apply_hypothesis(hyp, test_grid)

        if predicted is None:
            # Can't apply — reduce confidence but don't falsify
            hyp.confidence *= 0.5
            continue

        if predicted.shape == true_output.shape and np.array_equal(predicted, true_output):
            hyp.support += 1
            hyp.confidence = min(1.0, hyp.confidence * 1.5)
        else:
            hyp.falsified = True
            hyp.confidence = 0.0

    return [h for h in hypotheses if not h.falsified]
