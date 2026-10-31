---
name: xgboost-churn-training
description: Use to train, tune and evaluate an XGBoost binary classifier (e.g. churn) on a cleaned parquet dataset - leakage-safe split, bounded hyperparameter loop, threshold tuning, plots, .pkl/.onnx export and experiment logging.
---

# XGBoost binary-classification training

Every `run_python` call is a fresh process. Put the pipeline in ONE script per stage and persist intermediate state to disk.

## Stage 1 - split and preprocess
```python
import pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

df = pd.read_parquet("/workspace/data/processed/churn_clean.parquet")
y = df.pop("Churn")
X_tr, X_te, y_tr, y_te = train_test_split(df, y, test_size=0.2, stratify=y, random_state=42)
X_tr, X_va, y_tr, y_va = train_test_split(X_tr, y_tr, test_size=0.2, stratify=y_tr, random_state=42)

cat = X_tr.select_dtypes(exclude="number").columns.tolist()
num = X_tr.select_dtypes(include="number").columns.tolist()
pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat)],
                        remainder="passthrough", verbose_feature_names_out=False).fit(X_tr)  # fit on TRAIN only
joblib.dump(pre, "/workspace/artifacts/models/preprocessor.pkl")
```
Persist the splits as `/workspace/data/processed/{train,val,test}.parquet` so later stages reuse them.

## Stage 2 - bounded hyperparameter search (<= 30 trials)
- Use `optuna` (TPE) or `RandomizedSearchCV` with `StratifiedKFold(5)` on the TRAIN split.
- Search: `max_depth 3-8`, `learning_rate 0.01-0.3 (log)`, `subsample 0.6-1`, `colsample_bytree 0.5-1`, `min_child_weight 1-10`, `reg_lambda 0.1-10 (log)`, `gamma 0-5`.
- Imbalance: `scale_pos_weight = n_neg / n_pos`.
- Optimise PR-AUC (`average_precision`) unless the user asked otherwise.
- Call `log_experiment` for the best 3 trials (params + CV metrics).

## Stage 3 - final fit and evaluation
```python
from xgboost import XGBClassifier
model = XGBClassifier(**best_params, n_estimators=2000, early_stopping_rounds=50,
                      eval_metric="aucpr", tree_method="hist", random_state=42, n_jobs=-1)
model.fit(Xtr, y_tr, eval_set=[(Xva, y_va)], verbose=False)
```
- Choose the decision threshold on the VALIDATION split (max F1, or the cost the user specified); then evaluate ONCE on test.
- Report ROC-AUC, PR-AUC, F1, precision, recall, confusion matrix, and the baseline (prevalence) PR-AUC for context.

## Stage 4 - artifacts
- `/workspace/artifacts/models/xgb_churn.pkl` via `joblib.dump({"preprocessor": pre, "model": model, "threshold": t}, ...)`.
- ONNX (best effort, wrap in try/except and report failure): `onnxmltools.convert_xgboost(model, initial_types=[("input", FloatTensorType([None, n_features]))])` -> `/workspace/artifacts/models/xgb_churn.onnx`. The ONNX graph covers the XGBoost model only; the preprocessor stays in the .pkl.
- Plots to `/workspace/plots/training/`: ROC, precision-recall, confusion matrix, feature importance (gain), calibration curve.
- `/workspace/artifacts/reports/metrics.json` with params, threshold and test metrics.
- Final `log_experiment` call with the test metrics and every artifact path.

## Leakage guard (automatic, pre-execution)
Every `run_python` script is statically screened BEF# Skill: XGBoost Post-EDA Model Training & Optimization

## Purpose
Execute a consistent, deterministic, and auditable XGBoost training pipeline across classification, regression, and ranking/recommendation tasks. Dynamically adjust architectures, loss functions, hyperparameter tuning, evaluation metrics, and validation splits based on class imbalance, temporal elements, or structural grouping identified during exploratory data analysis.

## Parameters
- **dataset_path**: String path to the cleaned source dataset (e.g., Parquet format).
- **task_type**: String value of either `classification` (binary or multiclass), `regression`, or `recommendation` (ranking).
- **target_feature**: String name of the column containing the target variable.
- **imbalance_profile**: Dictionary providing `is_imbalanced` (Boolean) and `minority_ratio` (Float). Mandatory for classification.
- **temporal_profile**: Dictionary providing `is_temporal` (Boolean), `timestamp_column` (String), and `group_key` (Optional String).
- **recommendation_profile**: Dictionary providing `qid_column` (String representing Query/User ID) and `relevance_degrees` (Integer upper bound). Mandatory for ranking tasks.
- **output_directory**: (Optional) Base path for artifacts. Defaults to `/workspace`.

---

## Code Execution Rules
Every execution step is treated as a fresh process. The pipeline must be written in modular, monolithic scripts per stage that explicitly persist intermediate tables and state arrays to disk.

### Stage 1: Leakage-Safe Splitting and Preprocessing
- **Validation Strategy Integrity:** Build splits based on dataset architecture:
    - **Temporal (`is_temporal=True`):** Reject random shuffling or standard stratified K-Fold. Enforce a chronological walk-forward or single forward-chaining threshold split. Ensure no training sample timestamp is $\ge$ any validation/test sample timestamp.
    - **Recommendation:** Maintain group isolation using `qid_column`. A single Query/User ID must never bridge multiple splits.
    - **Standard Tabular Data:** Fall back to stratified splits for classification, or uniform random splits for regression.
- **Preprocessing Execution:**
    - Standardize categories via `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`.
    - Isolate identifiers as keys or raw strings; do not treat them as numerical features.
    - Fit transformers **exclusively on the training split** to prevent preprocessing data leakage.
    - Save splits as parquet tables and serialize the preprocessor to disk.

### Stage 2: Bounded Hyperparameter Optimization ($\le$ 30 Trials)
- **Framework:** Run `optuna` (TPE) or `RandomizedSearchCV` utilizing an optimization validation loop that mirrors the global dataset architecture (e.g., `TimeSeriesSplit`, `GroupKFold`, or `StratifiedKFold`).
- **Search Spaces:** `max_depth: 3-8`, `learning_rate: 0.01-0.3 (log)`, `subsample: 0.6-1.0`, `colsample_bytree: 0.5-1.0`, `min_child_weight: 1-10`, `reg_lambda: 0.1-10 (log)`, `gamma: 0-5`. Force `tree_method="hist"` for deterministic evaluation speed.
- **Objective and Imbalance Alignment:**
    - **Imbalanced Binary Classification:** Tune Area Under Precision-Recall Curve (`eval_metric="aucpr"` or `average_precision`). Compute `scale_pos_weight = count(negative_samples) / count(positive_samples)`.
    - **Multiclass Imbalance:** Compute custom sample weights using `sklearn.utils.class_weight.compute_sample_weight` and pass directly into the fitness routine.
    - **Regression:** Optimize `rmse` (or `mae` for heavy-tailed / high-outlier spaces using `reg:absoluteerror` or `reg:tweedie`).
    - **Recommendation/Ranking:** Pass an explicit group run-length structure array to a native `XGBRanker` interface using `qid`. Optimize `ndcg` or `map` metrics (e.g., `ndcg@10`).
- **Experiment Logging:** Log parameters and optimization curves for the top 3 configurations to the deployment register.

### Stage 3: Final Fit & Target Evaluation
- Instantiate the final model utilizing the optimal parameter payload, `n_estimators=2000`, and an explicit early-stopping threshold (`early_stopping_rounds=50`).
- Feed the training split and validation split concurrently into the engine's `eval_set` to log convergence trajectories and pinpoint overfitting inflection points.
- **Decision Boundary Optimization:** Tune the optimal decision threshold explicitly on the **validation split** (e.g., maximize F1 score or satisfy a user-specified cost metric).
- Run evaluation **exactly once on the final test split** using the tuned validation threshold.
- Compute all relevant cross-metrics: ROC-AUC, PR-AUC, F1, precision, recall, confusion matrix arrays, and baseline performance metrics (e.g., target prevalence) for context.

### Stage 4: Artifact Serialization & Reporting
- Export a master model deployment bundle (`.pkl`) containing the fitted pipeline, core preprocessors, the model booster, and the optimized decision boundary threshold.
- **ONNX Export:** Construct a best-effort native conversion step mapping the XGBoost graph structure to a deployment-ready `.onnx` array.
- Save execution plots at `dpi=300` into the output path: ROC curves, Precision-Recall curves, a confusion matrix heatmap, a calibration curve, and an isolated feature importance plot sorted by absolute fractional metric `gain`. Filter the feature importance visual to the **top 20 features** if cardinality becomes unreadable.
- Export a unified data file containing final parameter profiles, exact operational thresholds, and clean test metrics.

---

## Leakage Guard (Static Pre-Execution Screening)
All pipeline scripts are statically analyzed prior to sandbox execution. Violation of blocked rules halts execution and returns error signatures immediately.

| Rule | Severity | Triggering Condition | Corrective Engineering Action |
| :--- | :--- | :--- | :--- |
| **LK001** | Block | The feature matrix still contains the explicit target label column. | Pop target labels away from feature arrays prior to ingestion. |
| **LK002** | Block | A transformer/scaler/encoder is fit across full arrays before a split. | Execute splitting operations first; run `.fit()` only on training subsets. |
| **LK003** | Block (Waivable) | Features are derived directly from labels (e.g., full-frame target encoding). | Compute encoding parameters using `y_train` lines exclusively. |
| **LK004** | Block | Held-out evaluation test splits are passed into `fit` or `eval_set`. | Limit evaluation datasets to validation blocks; reserve test data for final scoring. |
| **LK005** | Warn (Waivable) | Operational thresholds or hyperparameters are tuned via test labels. | Isolate hyperparameter searching and boundary tuning to validation matrices. |
| **LK006** | Block (Waivable) | High-risk columns identified as target leakage suspects are retained. | Drop specific leakage columns or specify a restricted whitelist of safe features. |

### Syntactic Requirements & Waiver Protocols
- **Variable Tokens:** Use `y` / `y_*` for target attributes, `X` / `X_*` for features, and `*_test` / `*_te` exclusively for isolated, final evaluation partitions.
- **Waiver Format:** Use the precise, inline format code to bypass a waivable validation failure:
  `# leakage-waive: LK006 <Provide a clear architectural reason explaining why this is a false positive; minimum 10 characters>`

---

## Final Delivery Report Format
The final response file returned by the engine must follow this structure:
1. **Executive Framework:** High-level overview of the mapped task architecture, matrix footprints, and target balance rates.
2. **Structural Configurations:** Specific objective functions, evaluation targets, optimization parameters, and balancing metrics used.
3. **Validation Guardrail Logs:** Verifiable logs detailing split thresholds, chronological limits, or group boundaries proving isolation.
4. **Convergence Profile:** Iteration trajectory history showing the early-stopping cutoff and validation loss curves.
5. **Feature Attribution Analysis:** Explicit ranking of feature gains, coupled with path pointers to the generated feature importance visuals.
6. **Deployment Blueprint:** A complete, copy-pasteable Python/JSON parameters configuration dictionary to reproduce the exact model configuration.
   ORE it reaches the sandbox. Blocked scripts are not executed; you receive the findings instead. Fix and resubmit.

| Rule | Severity | Triggers when | Typical fix |
|---|---|---|---|
| LK001 | block | the feature matrix still contains the label column | `y = df.pop("Churn")` before building `X` |
| LK002 | block | scaler/encoder/imputer/selector is fit before `train_test_split` | split first, `fit` on the train split only |
| LK003 | block, waivable | a feature column is derived from the label (e.g. full-frame target encoding) | compute from `y_train` only, or overwrite the target column in place |
| LK004 | block | test data is passed to `fit`/`fit_transform` or `eval_set` | use the validation split; test is for `predict` only |
| LK005 | warn, waivable | threshold or hyperparameters appear to be chosen with test labels | select on the validation split |
| LK006 | block, waivable | a column flagged in `eda_summary.json` `leakage_suspects` is never dropped | `df.drop(columns=[...])` or an explicit feature list |

Variable naming helps the screener: use `y`/`y_*` for labels, `X`/`X_*` for features, and `*_test`/`*_te` only for held-out data.
Waiver (verified false positives on waivable rules only): `# leakage-waive: LK006 <reason, 10+ characters>`. Waivers are logged to `artifacts/logs/leakage_audit.jsonl` and shown in the output. Never try to evade the guard.
