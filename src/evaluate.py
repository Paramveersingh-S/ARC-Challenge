import json
from pathlib import Path
import numpy as np
from src.solver import solve_task

def format_submission(predictions: dict, output_path: str = "submission.json"):
    """
    ARC-AGI-2 submission format:
    { "task_id": {"attempt_1": [[...]], "attempt_2": [[...]]} }
    Submit 2 attempts per task for partial credit.
    """
    submission = {}
    for task_id, pred in predictions.items():
        # Second attempt: try a simple transformation as fallback
        pred_arr = np.array(pred)
        alt_pred = np.rot90(pred_arr).tolist()  # simple fallback
        submission[task_id] = {
            "attempt_1": pred,
            "attempt_2": alt_pred
        }
    with open(output_path, "w") as f:
        json.dump(submission, f)
    print(f"Saved {len(submission)} task predictions to {output_path}")
    return submission

def evaluate_on_training(data_dir: Path, n_tasks: int = 50) -> dict:
    """Self-evaluate before submission. ARC-AGI-2 metric: exact grid match."""
    task_files = sorted(data_dir.glob("*.json"))[:n_tasks]
    correct = 0
    total = 0

    for tf in task_files:
        task = json.loads(tf.read_text())
        for test_case in task["test"]:
            true_out = np.array(test_case["output"])
            pred = solve_task({"train": task["train"],
                               "test": [{"input": test_case["input"]}]})
            if pred.shape == true_out.shape and np.array_equal(pred, true_out):
                correct += 1
            total += 1

    acc = correct / total if total > 0 else 0
    print(f"Accuracy: {correct}/{total} = {acc:.1%}")
    return {"correct": correct, "total": total, "accuracy": acc}
