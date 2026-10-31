---
name: report-writing
description: Use when writing the final markdown report for a data-science or ML task (EDA findings, model results, artifacts, caveats). Defines structure and honesty rules.
---

# Report writing

Write `/artifacts/reports/final_report.md` with these sections, in order:

1. **Objective** - one sentence restating the user's request and the target definition.
2. **Data** - source file, rows x columns, target balance, cleaning actions (from `eda_summary.json`).
3. **Method** - split strategy, preprocessing, model, hyperparameter search budget, threshold selection.
4. **Results** - table of ROC-AUC, PR-AUC, F1, precision, recall on the held-out test set; confusion matrix; top 10 features.
5. **Artifacts** - bullet list of exact paths (model .pkl/.onnx, plots, metrics JSON, experiments.jsonl).
6. **Caveats and next steps** - leakage risks, class imbalance, drift, calibration, what was NOT tried.

Rules:
- Every number must come from a file or tool output you have seen; quote it, never estimate it.
- Reference plots by path rather than embedding binary data.
- State negative results (e.g. ONNX export failed) plainly.
