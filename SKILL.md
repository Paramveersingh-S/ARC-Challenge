---
name: arc-prize-active-hypothesis-agent
description: >
  Complete implementation guide for building an Active Hypothesis Testing Agent
  for ARC Prize 2026 (ARC-AGI-2 and ARC-AGI-3 tracks). Use this skill whenever
  the user is working on ARC-AGI tasks, abstract visual reasoning, program
  synthesis, grid-world agents, or anything related to the ARC Prize competition.
  Also triggers for: hypothesis-driven search, inductive logic programming,
  symbolic-neural hybrids, DSL-based reasoning, or test-time adaptation on
  grid puzzles. This skill is critical — always load it before writing any
  ARC-related code.
---

# ARC Prize 2026 — Active Hypothesis Testing Agent

## Project Identity

**Challenge:** ARC Prize 2026 (arcprize.org)
**Tracks:** ARC-AGI-2 (static reasoning, 85% target) + ARC-AGI-3 (interactive environments, agentic)
**Core Insight:** Humans approach ARC tasks as *scientists*, not pattern matchers. They form hypotheses about rules, design mental experiments to test them, update their beliefs, and only then execute. This agent replicates that loop.
**Submission Format:** Kaggle notebook, offline evaluation, no external API calls, open-source (MIT/CC0)
**Prize Pool:** $2M total

---

## What Makes This Novel

Most ARC submissions are either:
1. Pure LLM prompting (fails on ARC-AGI-3 — frontier models score <1%)
2. Program synthesis with brute-force enumeration (too slow)
3. Fine-tuned transformers (overfit to ARC-AGI-1 distribution)

This agent's novelty: **interleaved hypothesis generation + experimental design + Bayesian update**, where each candidate rule is falsified via cheaply-constructed test cases *before* committing to it. On ARC-AGI-3, the agent treats each environment as an experiment, actively choosing moves that maximally discriminate between competing world-model hypotheses.

---

## Architecture Overview

```
Input (grid examples or interactive env)
        │
        ▼
┌─────────────────────────────────┐
│   PERCEPTION MODULE             │
│   - Object decomposition        │
│   - Symmetry / pattern detect   │
│   - Attribute extraction        │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│   HYPOTHESIS GENERATOR          │
│   - DSL-based rule proposals    │
│   - LLM-seeded hypotheses       │
│   - Prior from example counts   │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│   ACTIVE EXPERIMENT DESIGNER    │
│   - Info-gain scoring per hyp   │
│   - Cheapest falsifying test    │
│   - For ARC-AGI-3: action plan  │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│   BAYESIAN BELIEF UPDATER       │
│   - Posterior over hypotheses   │
│   - Prune inconsistent rules    │
│   - Confidence threshold gate   │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│   EXECUTOR                      │
│   - Render output grid          │
│   - Submit action (AGI-3)       │
│   - Fallback: top-K beam        │
└─────────────────────────────────┘
```

---

## Step 1: Environment Setup

```bash
# Kaggle notebook — run in first cell
!pip install numpy scipy scikit-learn tqdm --quiet

# Directory layout expected by this skill
# /kaggle/input/arc-prize-2026/
#   arc-agi2/
#     training/   ← JSON files, each with train + test pairs
#     evaluation/
#   arc-agi3/
#     environments/  ← interactive env specs
```

```python
import json, os, copy, itertools
from pathlib import Path
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any

DATA_ROOT = Path("/kaggle/input/arc-prize-2026")
AGI2_TRAIN = DATA_ROOT / "arc-agi2" / "training"
AGI3_ENVS  = DATA_ROOT / "arc-agi3" / "environments"
```

---

## Step 2: Grid Perception and Object Extraction

This is the foundation — every hypothesis is formed *over objects*, not raw pixels.

```python
@dataclass
class ARCObject:
    """A connected region of same-color cells."""
    color: int
    cells: List[Tuple[int,int]]
    bbox: Tuple[int,int,int,int]   # row_min, col_min, row_max, col_max
    shape_signature: frozenset      # canonical shape hash

    @property
    def size(self): return len(self.cells)

    @property
    def height(self): return self.bbox[2] - self.bbox[0] + 1

    @property
    def width(self): return self.bbox[3] - self.bbox[1] + 1

    @property
    def is_rectangle(self):
        return self.size == self.height * self.width

    @property
    def center(self):
        rows = [c[0] for c in self.cells]
        cols = [c[1] for c in self.cells]
        return (sum(rows)/len(rows), sum(cols)/len(cols))


def extract_objects(grid: np.ndarray, background=0) -> List[ARCObject]:
    """BFS connected-component extraction, 4-connectivity."""
    visited = np.zeros_like(grid, dtype=bool)
    objects = []
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if visited[r,c] or grid[r,c] == background:
                continue
            color = grid[r,c]
            cells = []
            queue = [(r,c)]
            while queue:
                cr, cc = queue.pop()
                if visited[cr,cc]: continue
                visited[cr,cc] = True
                cells.append((cr,cc))
                for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr,nc = cr+dr, cc+dc
                    if 0<=nr<grid.shape[0] and 0<=nc<grid.shape[1]:
                        if not visited[nr,nc] and grid[nr,nc]==color:
                            queue.append((nr,nc))
            rows = [x[0] for x in cells]
            cols = [x[1] for x in cells]
            bbox = (min(rows), min(cols), max(rows), max(cols))
            # Canonical shape: normalize to origin
            r0,c0 = min(rows), min(cols)
            sig = frozenset((r-r0, c-c0) for r,c in cells)
            objects.append(ARCObject(color, cells, bbox, sig))
    return objects


def grid_attributes(grid: np.ndarray) -> dict:
    """High-level grid-level attributes for hypothesis seeding."""
    objs = extract_objects(grid)
    colors = set(int(v) for v in grid.flatten() if v != 0)
    return {
        "shape": grid.shape,
        "num_colors": len(colors),
        "colors": sorted(colors),
        "num_objects": len(objs),
        "objects": objs,
        "has_symmetry_h": np.array_equal(grid, np.fliplr(grid)),
        "has_symmetry_v": np.array_equal(grid, np.flipud(grid)),
        "has_diagonal_sym": (grid.shape[0]==grid.shape[1] and
                              np.array_equal(grid, grid.T)),
    }
```

---

## Step 3: The DSL — Hypothesis Space

Hypotheses are programs in a small Domain Specific Language. This bounds the search space and makes each hypothesis *checkable*.

```python
# ── DSL Primitives ──────────────────────────────────────────────────────────
# Each rule is: Condition → Transformation
# Conditions: size_gt(n), color_is(c), is_rect, is_largest, count_eq(n), ...
# Transformations: recolor(c), move(dr,dc), rotate_90, flip_h, flip_v,
#                  scale(f), copy_to(position), delete, fill_rect, ...

DSL_CONDITIONS = {
    "is_largest":   lambda objs, o: o == max(objs, key=lambda x: x.size),
    "is_smallest":  lambda objs, o: o == min(objs, key=lambda x: x.size),
    "is_rect":      lambda objs, o: o.is_rectangle,
    "is_square":    lambda objs, o: o.is_rectangle and o.height == o.width,
    "color_is":     lambda objs, o, c: o.color == c,
    "size_gt":      lambda objs, o, n: o.size > n,
    "count_color":  lambda objs, o, c: sum(1 for x in objs if x.color==c),
}

DSL_TRANSFORMS = {
    "recolor":     "change object color to c",
    "move_to":     "translate object by (dr, dc)",
    "rotate_90":   "rotate object 90 degrees CW",
    "flip_h":      "flip object horizontally",
    "flip_v":      "flip object vertically",
    "scale_up":    "scale object by factor f",
    "delete":      "remove object from grid",
    "fill_bg":     "fill bounding box with background color",
    "copy_pattern":"tile the object across grid",
    "gravity":     "move object in direction until collision",
}

@dataclass
class Hypothesis:
    id: str
    description: str
    condition: str       # DSL condition name + args
    transform: str       # DSL transform name + args
    confidence: float = 1.0
    support: int = 0     # number of examples consistent with this
    falsified: bool = False

    def __repr__(self):
        return f"H[{self.id}](conf={self.confidence:.2f}): IF {self.condition} THEN {self.transform}"
```

---

## Step 4: Hypothesis Generator

```python
def generate_hypotheses(examples: List[dict]) -> List[Hypothesis]:
    """
    Seed initial hypothesis pool from:
    1. Structural diff between input/output grids
    2. Object-level attribute changes
    3. Symmetry/count observations
    """
    hyps = []
    hid = 0

    for ex in examples:
        inp = np.array(ex["input"])
        out = np.array(ex["output"])
        in_attrs  = grid_attributes(inp)
        out_attrs = grid_attributes(out)

        # Shape change? → scaling or cropping hypothesis
        if inp.shape != out.shape:
            hyps.append(Hypothesis(
                id=f"h{hid:03d}", description="output is scaled/cropped version of input",
                condition="always", transform=f"resize_to({out.shape})",
                support=1))
            hid += 1

        # Color mapping? → recolor hypotheses
        in_colors  = set(int(v) for v in inp.flatten())
        out_colors = set(int(v) for v in out.flatten())
        if in_colors != out_colors:
            added = out_colors - in_colors
            removed = in_colors - out_colors
            if len(added)==1 and len(removed)==1:
                hyps.append(Hypothesis(
                    id=f"h{hid:03d}",
                    description=f"recolor {list(removed)[0]} → {list(added)[0]}",
                    condition=f"color_is({list(removed)[0]})",
                    transform=f"recolor({list(added)[0]})",
                    support=1))
                hid += 1

        # Object count change? → copy/delete hypothesis
        dn = out_attrs["num_objects"] - in_attrs["num_objects"]
        if dn > 0:
            hyps.append(Hypothesis(
                id=f"h{hid:03d}",
                description=f"output has {dn} more objects — copy/replicate rule",
                condition="is_largest", transform=f"copy_{dn}_times",
                support=1))
            hid += 1
        elif dn < 0:
            hyps.append(Hypothesis(
                id=f"h{hid:03d}",
                description=f"output has {abs(dn)} fewer objects — deletion rule",
                condition="is_smallest", transform="delete",
                support=1))
            hid += 1

        # Symmetry appeared? → mirror hypothesis
        if out_attrs["has_symmetry_h"] and not in_attrs["has_symmetry_h"]:
            hyps.append(Hypothesis(
                id=f"h{hid:03d}", description="flip_h to create symmetry",
                condition="always", transform="flip_h_and_merge", support=1))
            hid += 1

    # Deduplicate by description
    seen = set()
    unique = []
    for h in hyps:
        if h.description not in seen:
            seen.add(h.description)
            unique.append(h)
    return unique
```

---

## Step 5: Active Experiment Designer (Core Novelty)

This module answers: *"Which hypothesis should I try to falsify next, and how?"*

```python
def information_gain(hypothesis: Hypothesis, other_hyps: List[Hypothesis],
                     test_grid: np.ndarray) -> float:
    """
    Expected info gain of applying hypothesis to test_grid.
    High gain = this test would eliminate many other hypotheses.
    """
    # Predict outcome under this hypothesis
    predicted = apply_hypothesis(hypothesis, test_grid)
    # Count how many other hyps would agree vs disagree
    agreements = sum(1 for h in other_hyps
                     if apply_hypothesis(h, test_grid) == predicted)
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


def apply_hypothesis(hyp: Hypothesis, grid: np.ndarray) -> Optional[np.ndarray]:
    """
    Execute DSL program represented by hypothesis on a grid.
    Returns predicted output, or None if hypothesis is inapplicable.
    """
    result = grid.copy()
    objects = extract_objects(result)

    try:
        if "recolor" in hyp.transform:
            # parse: recolor(from_c, to_c)
            parts = hyp.transform.replace("recolor(","").replace(")","").split(",")
            fc, tc = int(parts[0]), int(parts[1])
            result[result == fc] = tc

        elif "flip_h" in hyp.transform:
            result = np.fliplr(result)

        elif "flip_v" in hyp.transform:
            result = np.flipud(result)

        elif "rotate_90" in hyp.transform:
            result = np.rot90(result, k=1)

        elif "delete" in hyp.transform:
            # delete smallest object
            if objects:
                smallest = min(objects, key=lambda o: o.size)
                for r,c in smallest.cells:
                    result[r,c] = 0

        elif "gravity" in hyp.transform:
            # simplified: fall down
            for col in range(result.shape[1]):
                column = result[:,col]
                non_zero = column[column != 0]
                zeros = column[column == 0]
                result[:,col] = np.concatenate([zeros, non_zero])

        elif "resize_to" in hyp.transform:
            # extract target shape from string
            import re
            match = re.search(r'\((\d+),\s*(\d+)\)', hyp.transform)
            if match:
                h, w = int(match.group(1)), int(match.group(2))
                from scipy.ndimage import zoom
                factors = (h/result.shape[0], w/result.shape[1])
                result = zoom(result, factors, order=0).astype(int)

        return result
    except Exception:
        return None
```

---

## Step 6: Bayesian Belief Updater

```python
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
```

---

## Step 7: Full Solver Loop

```python
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
        hyp, grid_idx = select_experiment(active, available_grids)
        idx = available_grids.index(grid_idx) if grid_idx in available_grids else 0

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
```

---

## Step 8: ARC-AGI-3 Agentic Extension

For interactive environments, the agent must *act* to learn. The same hypothesis loop applies, but experiments are real environment interactions.

```python
class ARC3Agent:
    """
    Agent for ARC-AGI-3 interactive environments.
    Maintains a world model, forms hypotheses about goals,
    and chooses actions that maximise info gain about the win condition.
    """

    def __init__(self):
        self.hypotheses: List[Hypothesis] = []
        self.world_model: dict = {}
        self.action_history: List[Any] = []
        self.observation_history: List[Any] = []
        self.goal_hypothesis: Optional[str] = None

    def observe(self, obs: dict):
        """Process new environment observation."""
        self.observation_history.append(obs)
        self._update_world_model(obs)

        # First observation: generate initial goal hypotheses
        if len(self.observation_history) == 1:
            self.hypotheses = self._seed_goal_hypotheses(obs)

    def _seed_goal_hypotheses(self, obs: dict) -> List[Hypothesis]:
        """From first observation, generate candidate win conditions."""
        hyps = []
        grid = np.array(obs.get("grid", []))

        # Hypothesis classes for interactive envs
        goal_templates = [
            "reach_specific_cell",
            "match_target_pattern",
            "collect_all_objects_of_color",
            "sort_objects_by_attribute",
            "create_symmetry",
            "fill_empty_region",
            "replicate_pattern_n_times",
        ]
        for i, template in enumerate(goal_templates):
            hyps.append(Hypothesis(
                id=f"goal_{i:02d}",
                description=template,
                condition="always",
                transform=template,
                confidence=1.0/len(goal_templates)
            ))
        return hyps

    def _update_world_model(self, obs: dict):
        """Update internal world model from observation."""
        self.world_model["last_obs"] = obs
        self.world_model["step"] = len(self.observation_history)

        if "reward" in obs:
            self.world_model["last_reward"] = obs["reward"]
        if "done" in obs:
            self.world_model["done"] = obs["done"]

    def select_action(self, available_actions: List[Any]) -> Any:
        """
        Choose action with highest expected info gain about goal hypothesis.
        Falls back to random if no clear signal.
        """
        active_hyps = [h for h in self.hypotheses if not h.falsified]

        if not active_hyps or len(available_actions) == 0:
            import random
            return random.choice(available_actions) if available_actions else None

        # Score each action by expected disambiguation power
        scored = []
        for action in available_actions[:20]:  # cap for compute
            # Simulate expected outcome (simplified)
            predicted_reward = self._predict_reward(action, active_hyps)
            info_gain_score  = self._action_info_gain(action, active_hyps)
            # Balance: exploit if confident, explore if uncertain
            confidence = max(h.confidence for h in active_hyps)
            score = (confidence * predicted_reward +
                     (1 - confidence) * info_gain_score)
            scored.append((score, action))

        scored.sort(reverse=True)
        return scored[0][1]

    def _predict_reward(self, action, hyps: List[Hypothesis]) -> float:
        """Weighted average predicted reward under hypothesis ensemble."""
        total_weight = sum(h.confidence for h in hyps)
        if total_weight == 0: return 0.0
        # Simplified: high-confidence hypotheses vote on action quality
        return sum(h.confidence * self._hyp_action_score(h, action)
                   for h in hyps) / total_weight

    def _hyp_action_score(self, hyp: Hypothesis, action) -> float:
        """How well does this action align with this hypothesis goal?"""
        # Domain-specific scoring — extend per environment type
        if "reach" in hyp.description and hasattr(action, 'direction'):
            return 0.6  # movement actions favored for reach goals
        return 0.3      # baseline

    def _action_info_gain(self, action, hyps: List[Hypothesis]) -> float:
        """How much info does this action give us?"""
        # Novel actions (not tried yet) have higher info gain
        if action not in self.action_history[-5:]:
            return 0.8
        return 0.2

    def update_after_step(self, action, new_obs: dict):
        """Update beliefs after receiving environment feedback."""
        reward = new_obs.get("reward", 0)
        self.action_history.append(action)

        # Falsify hypotheses inconsistent with observed reward
        for hyp in self.hypotheses:
            if hyp.falsified: continue
            expected = self._hyp_action_score(hyp, action)
            if reward > 0.5 and expected < 0.3:
                hyp.confidence *= 1.5
            elif reward < 0.1 and expected > 0.7:
                hyp.confidence *= 0.5
                if hyp.confidence < 0.05:
                    hyp.falsified = True

        self.observe(new_obs)
```

---

## Step 9: Kaggle Submission Format

```python
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
```

---

## Evaluation and Debugging

```python
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
```

---

## Key Extension Points for Improvement

1. **Better hypothesis seeding**: Replace the rule-based generator with an LLM (offline, ONNX-quantized) that reads the example images and proposes natural language rules, which are then parsed into DSL.

2. **Neural object detector**: Replace BFS with a small CNN trained on ARC-AGI-1 to detect multi-part objects that share no direct adjacency.

3. **Meta-learning**: Train a small model offline on ARC-AGI-1 training data to predict which hypothesis *class* will work for a given task structure. Use it to bias the initial prior.

4. **ARC-AGI-3 world model**: Replace `_predict_reward` with a learned transition model trained on environment replays.

5. **Ensemble multiple hypotheses**: Instead of picking the single best hypothesis, render a weighted blend using top-K candidates.

---

## References to Read Before Implementing

- François Chollet, "On the Measure of Intelligence" (2019) — original ARC paper, defines fluid intelligence
- ARC-AGI-3 technical paper: arXiv:2603.24621 — interactive environment format
- "Program Synthesis with Large Language Models" (Austin et al., 2021)
- "DreamCoder: Growing generalizable, interpretable knowledge with wake-sleep Bayesian program learning" (Ellis et al., 2021) — closest prior work to this approach
- ARC Prize 2025 solution writeups on arcprize.org/blog — learn what worked last year
