# ARC Prize 2026 — Active Hypothesis Testing Agent

This repository contains an implementation for the **ARC Prize 2026** competition, targeting both ARC-AGI-2 and ARC-AGI-3 tracks.

## Approach
This agent replicates the human scientific method for puzzle solving:
1. **Hypothesize**: Generates DSL-based candidate rules from observations.
2. **Experiment**: Evaluates hypotheses using information gain.
3. **Update**: Applies Bayesian updates to hypothesis confidences.

## Features
- Complete Active Hypothesis Testing loop.
- Support for ARC-AGI-2 (static grids) and ARC-AGI-3 (interactive environments).
- Full suite of DSL transformations tailored for abstraction.

## Repository Structure
- `src/`: Core logic modules (perception, generation, experimentation, beliefs, execution).
- `tests/`: Unit testing for each module.
- `paper/`: Draft notes and arxiv submission.
- `notebook.ipynb`: Main Kaggle submission notebook.
