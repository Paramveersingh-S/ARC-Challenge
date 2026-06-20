<div align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python Version"/>
  <img src="https://img.shields.io/badge/Status-Active-success.svg" alt="Status"/>
  <img src="https://img.shields.io/badge/Accuracy-Baseline_Verified-orange.svg" alt="Accuracy"/>
  <h1>ARC-Challenge: Active Hypothesis Testing Agent</h1>
</div>

Welcome to the **ARC-Challenge** repository! This project implements a novel **Active Hypothesis Testing framework** aimed at solving the incredibly difficult Abstraction and Reasoning Corpus (ARC-AGI-2 and ARC-AGI-3) challenges.

---

## 🚀 Key Achievements
- **Test-Time Compute (LLM Synthesis)**: Achieved **`40.0%`** accuracy (2/5 validation sample) by dynamically executing `gemini-3.5-flash` generated Python code within an isolated execution sandbox.
- **Zero-Shot Heuristics Baseline**: Solves **`3.8%`** (16/416 tasks) of the full dataset running completely offline, using pure deterministic heuristics and mathematical combinatorial chains.
- **End-to-End Pipeline**: Fully integrated Perception engine, Execution Engine, and Hypothesis-generation logic.

## 🧠 Technical Details

This agent runs an **Active Hypothesis Testing Loop** containing 5 distinct modules:

1. **Perception Engine**: Dynamically calculates bounds, isolates 4-connected subcomponents, deduces true background colors, and generates `grid_attributes()`.
2. **Hypothesis Generator**: Uses combinatorial heuristics to dynamically generate rule chains such as `rotate_90_cw | map_colors(1:2)`.
3. **Execution Engine (DSL)**: Capable of 15+ complex geometric and topological operations (e.g. `fill_holes`, `tile_2x2`, `gravity_down`, `recolor_largest`).
4. **Bayesian Belief Updater**: Maintains an ensemble of hypotheses. If an execution matches true outputs, confidence multiplies by `1.5x`. If it predicts incorrectly, it is rigorously falsified.
5. **Interactive Agent (ARC-AGI-3)**: Implements a 30-step action engine using an "Explore-First" multi-armed bandit approach.

## 🗺️ System Architecture

```mermaid
graph TD
    A[Raw Input Grid] -->|Feature Extraction| B(Perception Engine)
    B -->|Grid Attributes| C{Hypothesis Generator}
    C -->|Candidate Programs| D[Bayesian Ensemble]
    D -->|Executes DSL| E(Execution Engine)
    E -->|Predicted Output| F{Belief Updater}
    F -->|Correct| G((Increase Confidence))
    F -->|Incorrect| H((Falsify and Delete))
```

## 🛠️ Usage

This repository is built perfectly for Google Colab or Kaggle execution.

```python
import sys
sys.path.append('./ARC-Challenge')

from src.evaluate import evaluate_on_training
from pathlib import Path

DATA_DIR = Path('./ARC/data/training')
results = evaluate_on_training(DATA_DIR, n_tasks=400)
```
