# ARC Prize 2026 — Active Hypothesis Testing Agent
## Master Project Prompt

---

## Context & Mission

You are building a competition entry for **ARC Prize 2026** (arcprize.org), the world's leading open research competition for measuring AI fluid intelligence. The prize pool is $2M. You are an undergrad researcher with ML/deep learning expertise, running everything on **Kaggle notebooks (free T4/P100 GPUs)** with 10–20 hours per week available.

**Your core thesis:** Humans solve ARC tasks by acting like scientists — they don't pattern-match, they *hypothesize, experiment, and update*. Your agent replicates this loop explicitly.

**Two tracks to target:**
- `ARC-AGI-2`: Static grid reasoning. Input/output grid examples. Goal: reach 85% exact-match accuracy on 400 private tasks.
- `ARC-AGI-3`: Interactive environments. No instructions given. Agent must explore, infer goals, and solve turn-based puzzles. Humans score 100%; frontier LLMs score <1%.

**Submission rules:**
- Kaggle notebook only, offline (no internet during eval, no Claude/GPT API)
- All code MIT/CC0 licensed and open-sourced
- 2 attempts per task (partial credit)

---

## What Already Exists (Don't Reinvent)

- ARC-AGI-1 solution approaches: `o3`-style reasoning, program synthesis, DSL enumeration
- DreamCoder: Bayesian program learning (your closest prior work)
- `arc-dsl` open-source library: pre-built ARC transformations in Python (pip install)
- `py-arckit`: Dataset loader, visualization, metric computation
- ARC-AGI-3 Developer Preview: ~50 environment types publicly released

**Your novelty:** Active experiment design (information-gain-based hypothesis selection) + Bayesian belief updating. No one has applied this framework explicitly to ARC.

---

## Phase-by-Phase Build Plan

### Phase 1 (Week 1–2): Foundation
**Goal:** Solve 30%+ of ARC-AGI-2 training tasks

Tasks:
1. Load ARC-AGI-2 dataset with `py-arckit`
2. Implement grid perception: `extract_objects()`, `grid_attributes()`
3. Build DSL with 15+ transformations (see SKILL.md)
4. Implement `generate_hypotheses()` from training examples
5. Implement `apply_hypothesis()` executor
6. Run brute-force: apply all hypotheses to all training examples, pick majority
7. Evaluate on 100 training tasks → target ≥30% accuracy

**Deliverable:** Baseline Kaggle notebook scoring >0% on public leaderboard

### Phase 2 (Week 3–4): Active Experiment Loop
**Goal:** Reach 50%+ training accuracy

Tasks:
1. Implement `information_gain()` scoring function
2. Implement `select_experiment()` — picks most discriminating (hypothesis, grid) pair
3. Implement `update_beliefs()` — Bayesian confidence updating
4. Wire into `solve_task()` full loop
5. Profile: which task types does the agent still fail on?
6. Add 10+ more DSL transforms for failing categories (e.g., gravity, tiling, fractal)

**Deliverable:** Notebook with full active-testing loop; write up results as a methods section

### Phase 3 (Week 5–6): ARC-AGI-3 Extension
**Goal:** Score >5% on ARC-AGI-3 (5x better than frontier LLMs)

Tasks:
1. Study the 50 public ARC-AGI-3 environment types; categorise them
2. Implement `ARC3Agent` class (see SKILL.md)
3. Implement world model update from observations
4. Implement action selection via info-gain
5. Test on Developer Preview environments
6. Add environment-type classifier to seed better priors per env type

**Deliverable:** Agent that beats GPT-5.4 Pro on ARC-AGI-3 Developer Preview

### Phase 4 (Week 7–8): Polish and Submit
**Goal:** Final submission + arxiv paper draft

Tasks:
1. Run on full evaluation sets
2. Format `submission.json` per Kaggle spec
3. Write solution writeup (required for prize eligibility)
4. Draft 4-page arxiv paper: "Active Hypothesis Testing for Abstract Reasoning"

---

## File Structure to Build

```
arc-hypothesis-agent/
├── notebook.ipynb          ← main Kaggle submission notebook
├── src/
│   ├── perception.py       ← extract_objects, grid_attributes
│   ├── dsl.py              ← DSL primitives, Hypothesis dataclass
│   ├── generator.py        ← generate_hypotheses
│   ├── experiment.py       ← information_gain, select_experiment
│   ├── beliefs.py          ← update_beliefs, Bayesian logic
│   ├── executor.py         ← apply_hypothesis
│   ├── solver.py           ← solve_task, solve_all
│   ├── agi3_agent.py       ← ARC3Agent class
│   └── evaluate.py         ← scoring, submission formatter
├── tests/
│   ├── test_perception.py  ← unit tests for object extraction
│   ├── test_dsl.py         ← test each DSL transform
│   └── test_solver.py      ← end-to-end on 10 training tasks
├── paper/
│   └── draft.md            ← arxiv paper draft
└── skills/
    └── SKILL.md            ← this file (for AI assistant context)
```

---

## Critical Implementation Details

### Object Extraction Edge Cases
- ARC uses 4-connectivity (not 8). Never use diagonal neighbors.
- Color 0 = background in most tasks, but NOT ALL. Check if 0 appears in output.
- A "grid" is always a 2D list of ints 0–9. Max size is 30x30.
- Some tasks have multiple disconnected objects of the same color — treat as separate.

### DSL Must-Have Transforms (implement these first)
```
1.  recolor(old_c → new_c)
2.  flip_horizontal
3.  flip_vertical  
4.  rotate_90_cw
5.  rotate_180
6.  delete_smallest_object
7.  delete_largest_object
8.  gravity_down (objects fall to bottom)
9.  gravity_up
10. tile_2x2 (repeat pattern)
11. crop_to_bounding_box
12. fill_holes (flood-fill interior)
13. outline_only (keep perimeter cells)
14. sort_rows_by_color
15. mirror_and_append (create symmetry)
```

### Hypothesis Confidence Update Rule
```python
# Correct prediction: reward
hyp.confidence = min(1.0, hyp.confidence * 1.5)
# Wrong prediction: falsify
hyp.falsified = True  # hard falsification, not soft decay
# Inapplicable: soft penalty
hyp.confidence *= 0.7
```

### ARC-AGI-3 Action Space
- Actions are typically: `{"type": "move", "direction": "up/down/left/right"}`
  or `{"type": "select", "row": r, "col": c}`
- Each environment has 3–30 max steps
- Reward: 0 until win condition met, then 1.0 (sparse)
- Key insight: exploration is mandatory — commit the first 1/3 of steps to info-gathering

### Submission Format (exact)
```json
{
  "task_id_here": {
    "attempt_1": [[0,1,2],[3,4,5]],
    "attempt_2": [[0,0,0],[0,0,0]]
  }
}
```

---

## Evaluation Targets

| Metric | Baseline (random) | Target Phase 1 | Target Final |
|---|---|---|---|
| ARC-AGI-2 training acc | 0% | 30% | 60%+ |
| ARC-AGI-2 eval acc | 0% | — | 40%+ |
| ARC-AGI-3 Developer Preview | 0% | 5% | 15%+ |
| Submission valid | ✓ | ✓ | ✓ |

---

## Paper Angle (for arxiv submission)

**Title:** "Active Hypothesis Testing for Abstract Visual Reasoning"

**Key claims:**
1. Information-gain-based experiment selection outperforms brute-force hypothesis enumeration by X% on ARC-AGI-2
2. The framework generalises to interactive environments (ARC-AGI-3) with zero task-specific training
3. Human-interpretable: each prediction comes with a natural-language explanation of the rule used

**Ablation table to run:**
- Random hypothesis selection vs. info-gain selection
- Hard falsification vs. soft Bayesian update
- With LLM seeding vs. without (offline ONNX model)

---

## Resources

- Dataset: kaggle.com/competitions/arc-prize-2026-arc-agi-2
- ARC viewer: arcprize.org/play (for manual understanding)
- Community Discord: discord.gg/arcprize
- Prior solutions: arcprize.org/blog (2024, 2025 writeups)
- DreamCoder paper: arxiv.org/abs/2006.08381
- ARC-AGI-3 paper: arxiv.org/abs/2603.24621
