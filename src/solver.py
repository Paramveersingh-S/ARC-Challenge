import json
from pathlib import Path
import numpy as np
from src.generator import generate_hypotheses
from src.experiment import select_experiment
from src.beliefs import update_beliefs
from src.executor import apply_hypothesis

def solve_task(task: dict, max_iterations: int = 20) -> np.ndarray:
    """
    Main entry point. task = {"train": [...], "test": [...]}
    Returns predicted output for test input.
    """
    train_examples = task["train"]
    test_input = np.array(task["test"][0]["input"])

    # Step 1: Generate hypotheses from training examples
    hypotheses = generate_hypotheses(train_examples)
    print(f"  Generated {len(hypotheses)} initial hypotheses")

    # Step 2: Validate against training examples iteratively
    available_grids = [np.array(ex["input"]) for ex in train_examples]
    true_outputs    = [np.array(ex["output"]) for ex in train_examples]

    for iteration in range(max_iterations):
        active = [h for h in hypotheses if not h.falsified]
        if not active:
            print(f"  All hypotheses falsified at iteration {iteration}")
            break
        if len(active) == 1:
            print(f"  Single hypothesis remaining: {active[0]}")
            break

        # Select most informative experiment
        hyp, grid_to_test = select_experiment(active, available_grids)
        # Find index of grid_to_test in available_grids
        idx = 0
        for i, g in enumerate(available_grids):
            if np.array_equal(g, grid_to_test):
                idx = i
                break

        # Update beliefs
        hypotheses_remaining = update_beliefs(
            active, available_grids[idx], true_outputs[idx]
        )
        hypotheses = hypotheses_remaining

        print(f"  Iter {iteration}: {len(hypotheses)} hypotheses remain")

    # Step 3: Execute best hypothesis on test input
    active = sorted([h for h in hypotheses if not h.falsified],
                    key=lambda h: (-h.support, -h.confidence))

    if not active:
        # Fallback: return test input unchanged
        return test_input

    best = active[0]
    print(f"  Applying: {best}")
    result = apply_hypothesis(best, test_input)

    return result if result is not None else test_input

def solve_all(data_dir: Path) -> dict:
    """Run solver on all tasks, return {task_id: predicted_output}."""
    predictions = {}
    task_files = sorted(data_dir.glob("*.json"))

    for tf in task_files:
        task = json.loads(tf.read_text())
        task_id = tf.stem
        try:
            pred = solve_task(task)
            predictions[task_id] = pred.tolist()
            print(f"✓ {task_id}")
        except Exception as e:
            print(f"✗ {task_id}: {e}")
            predictions[task_id] = np.zeros((3,3), dtype=int).tolist()

    return predictions
