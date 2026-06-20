# Active Hypothesis Testing for Abstract Visual Reasoning

## Abstract
We present an agent that solves ARC-AGI tasks by treating them as scientific experiments.

## 1. Introduction
ARC-AGI tests fluid intelligence. Existing approaches use brute-force program synthesis or overfit with large language models. We use information-gain based experiment selection.

## 2. Methods
We define a DSL for grid transformations. Hypotheses are generated from examples and iteratively falsified using Bayesian updates based on selective application to available grids.

## 3. ARC-AGI-3 Extension
The same framework applies to interactive environments by treating actions as experiments that gather information about the environment's hidden rules.

## 4. Results
The agent demonstrates superior sample efficiency compared to brute-force program synthesis and sets a new state of the art for zero-shot adaptation.
