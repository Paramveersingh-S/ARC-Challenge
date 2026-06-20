import numpy as np
from typing import List, Tuple
from src.dsl import Hypothesis
from src.executor import apply_hypothesis

def information_gain(hypothesis: Hypothesis, other_hyps: List[Hypothesis],
                     test_grid: np.ndarray) -> float:
    """
    Expected info gain of applying hypothesis to test_grid.
    High gain = this test would eliminate many other hypotheses.
    """
    # Predict outcome under this hypothesis
    predicted = apply_hypothesis(hypothesis, test_grid)
    # Count how many other hyps would agree vs disagree
    agreements = 0
    for h in other_hyps:
        pred_h = apply_hypothesis(h, test_grid)
        if predicted is not None and pred_h is not None and predicted.shape == pred_h.shape and np.array_equal(predicted, pred_h):
            agreements += 1

    # Shannon: high disagreement = high info gain
    p = agreements / max(len(other_hyps), 1)
    if p == 0 or p == 1: return 0.0
    return -(p * np.log2(p) + (1-p) * np.log2(1-p))

def select_experiment(hypotheses: List[Hypothesis],
                      available_grids: List[np.ndarray]) -> Tuple[Hypothesis, np.ndarray]:
    """
    Pick the (hypothesis, test_grid) pair that maximises information gain.
    For ARC-AGI-2: test_grid = held-out input from training examples.
    For ARC-AGI-3: test_grid = current environment observation.
    """
    best_gain = -1
    best_pair = (hypotheses[0], available_grids[0])

    active_hyps = [h for h in hypotheses if not h.falsified]

    for hyp in active_hyps[:10]:   # budget: top-10 candidates
        for grid in available_grids:
            gain = information_gain(hyp, active_hyps, grid)
            if gain > best_gain:
                best_gain = gain
                best_pair = (hyp, grid)

    return best_pair
