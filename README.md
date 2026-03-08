# Analysis-of-SEV-in-ML-Models

This project aims to analyze the Sparse Explanation Value (SEV) method introduced in the research paper "Sparse and Faithful Explanations Without Sparse Models" by Sun et. al. to see how it would react to model multiplicity.

## Overview
Reproduction and extension of experiments from:
- "Sparse and Faithful Explanations Without Sparse Models" (Sun et al., AISTATS 2024) → [arXiv link](https://arxiv.org/abs/2402.09702)
- Original code: [GitHub/williamsyy/SparseExplanationValues](https://github.com/williamsyy/SparseExplanationValues)

## Key Extension & Insight
- Bootstrapped 30 Logistic Regression models extremely similar to the original LR model.
- Computed SEV+ values and feature sets for 50 positive instances per model.
- **Finding**: Average SEV standard deviation across instances: ~0.05 → sparsity levels (minimal feature count for faithful explanations) remain stable and low, consistent with the original paper.  
- **However**: Average feature agreement rate: 18.7% → specific features often flip significantly across models, revealing instability in explanation composition despite stable sparsity.

## Tech Stack
- Python, scikit-learn, pandas, numpy
- Custom SEV implementation (from original repo)
- Bootstrapping + SEV computation loop

- ## Notebooks
- `sev_stability_notebook.ipynb`: Main analysis with bootstrap loop, SEV calculations, and stability results.

## Learnings
- Sparsity in explanation size holds, but feature identity is volatile → motivates robust/ensemble explanation methods.
- Ties into broader interpretability challenges in ML.

Citation: If using, please reference the original SEV paper.
