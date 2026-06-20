import numpy as np
from src.solver import solve_task

def test_solve_task_basic():
    task = {
        "train": [
            {"input": [[1, 1]], "output": [[2, 2]]},
            {"input": [[3]], "output": [[2]]}
        ],
        "test": [
            {"input": [[4, 4, 4]]}
        ]
    }
    # Basic test
    pred = solve_task(task, max_iterations=5)
    assert pred is not None
