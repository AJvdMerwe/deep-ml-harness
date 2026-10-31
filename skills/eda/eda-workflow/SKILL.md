# Skill: Exploratory Data Analysis (EDA)

## Purpose
Run a consistent, auditable exploratory data analysis process for data science datasets and produce analysis artifacts, treatment recommendations, and validation findings before modeling.

## Parameters
- **dataset**: String path or accessible table reference to the source dataset.
- **target_feature**: String name of the column containing the target variable.
- **task_type**: (Optional) Classification, regression, survival, forecasting, or unspecified.
- **timestamp_column**: (Optional) Name of the column containing temporal tracking tokens.
- **entity_key**: (Optional) Grouping or unique entity identifier to detect clustered records.
- **ordinal_orders**: (Optional) Dictionary mapping ordinal column names to explicit level lists.
- **positive_class**: (Optional) The target category value to treat as positive for binary tasks.
- **output_directory**: (Optional) Base directory for artifacts. Defaults to current working directory.
- **thresholds**: (Optional) Set boundaries for rare_categories (default 0.05), skewness (default 1.5), and missingness (default 0.20).

## Instructions

### 1. Confirm Inputs and Analytical Intent
- Identify the dataset, target feature, prediction task, unit of observation, candidate time/index columns, and any known ordinal category orders.
- If any parameters are missing, infer cautiously from column names and value distributions. Label these inferences explicitly as assumptions.
- Ask for clarification only when the ambiguity would fundamentally alter validation strategy or treatment choices.

### 2. Classify Dataset Structure
- Determine whether rows represent independent tabular observations, time-aware observations with timestamps or grouped temporal ordering, or a true time series with serial dependence.
- Record the evidence driving this classification. Select downstream profiling methods that preserve and respect the identified ordering.

### 3. Audit and Convert Data Types
- Profile raw types, semantic types, missingness, uniqueness, parsing failures, and mixed-format columns.
- Convert numeric-looking strings, booleans, dates, datetimes, and categories only after measuring conversion loss (tracking dropped characters or unparseable rows).
- Preserve structural identifiers explicitly as strings or keys; do not treat them as numerical distributions.
- Distinguish nominal categories from ordinal categories. Use unordered categorical treatment for nominal features. Never impose lexical or alphabetic order as an ordinal ranking unless explicitly confirmed by parameters or metadata.

### 4. Profile the Target First
- Classify the target task as classification, regression, survival-like, forecasting, or unspecified.
- For classifiers, calculate target counts, class ratios, minority/majority imbalance indicators, missing target values, and rare classes before performing any data fitting or experimental splits.
- When recommending or implementing a classification split, mandate an explicit `stratify=target` protocol unless temporal ordering or grouped leakage makes stratification mathematically invalid. If invalid, explain the structural conflict and prescribe an explicit time-aware or grouped validation split strategy instead.

### 5. Examine Numeric Features Distribution-by-Distribution
- For each numeric variable, report robust and conventional summaries (mean, median, standard deviation, IQR, min, max), missingness rates, zero inflation, finite vs. non-finite counts, skewness, tail behavior, multimodality indicators, and plausible outliers.
- Recommend downstream treatments according to the observed distribution and modeling architecture: no change, robust scaling, non-linear transformation, winsorization/capping, domain-rule correction, or explicit missingness indicator tokens.
- Do not automatically remove outliers or transform values; quantify their footprint and preserve an untreated baseline for model comparison.

### 6. Analyze Categorical Features with Correct Semantics
- **Nominal Features:** Inspect cardinality, rare level frequencies based on the provided thresholds, missingness rates, target rates/distributions per category, and potential target leakage signs.
- **Ordinal Features:** Enforce the declared level order. Examine monotonic and non-monotonic relationships between the ordered steps and the target feature.
- Recommend encoding choices separately for nominal (e.g., one-hot, target encoding with smoothing) and ordinal features (e.g., explicit integer mapping).

### 7. Investigate Target Relationships & Screen for Leakage
- Use task- and type-appropriate summaries: grouped target rates and contingency measures for categorical classification; distribution comparisons and effect sizes for numeric classification; correlations and robust trends for regression; and lagged or rolling cross-correlations for time series.
- Flag target leakage risks including target-derived fields, post-outcome timestamps, identifier leakage, and sparse groups.
- Warn explicitly against multiple-testing risk when evaluating broad spaces (>50 features) where random associations appear statistically viable. Separate association from causal assumptions.

### 8. Assess Dependence Among Predictors
- Produce covariance and correlation analyses for suitable numeric variables, documenting Pearson (linear) and Spearman (rank-based) correlation coefficients.
- Document scale sensitivity constraints, nonlinearity limitations, missing-data handling method (e.g., pairwise vs. listwise deletion), sample size impacts, and multicollinearity signals (e.g., VIF or high-correlation clusters).
- For mixed feature types, supplement with suitable association measures (e.g., Cramér’s V, ANOVA F-test scores) rather than coercing nominal string categories to arbitrary pseudo-integers.

### 9. Apply Temporal Safeguards
- For time-aware or time-series data, sort rows chronologically by time within entity groups.
- Check for duplicate timestamps, irregular sampling intervals, data gaps, macro trends, seasonality, autocorrelation, lag structures, rolling stability, and data/concept drift indicators.
- Enforce strict chronological or rolling-origin validation splits. Prevent future information from entering past feature spaces. For independent tabular data, fallback to random or grouped cross-validation as justified by the entity key.

### 10. Generate Reproducible Outputs
- Build a deterministic baseline profile using a script framework (e.g., `scripts/main.py`).
- Save the final correlation plot exactly to `plots/correlation_matrix.png` with a resolution of `dpi=300`. Create directories dynamically if missing.
- **Visual Safeguard:** If the active feature space exceeds **30 features**, filter the visual correlation plot matrix to show the top 30 features with the strongest absolute correlation to the target, or slice them by logical feature blocks, to ensure visual legibility.
- Log all raw-to-semantic conversion steps, engineering assumptions, structural warnings, and active tuning parameters.

### 11. Validate Completion Checklist
Before finalizing the pipeline execution, verify the following properties:
- Every feature has been assigned an explicit semantic type.
- Every usable predictor has been assessed against the target feature.
- Nominal and ordinal variables are isolated under distinct treatment steps.
- Numeric transformations cite explicit distributional evidence (skewness, outliers).
- The temporal/independent structure of the dataset is explicitly classified.
- Covariance/correlation dependencies are documented.
- Classification split guidance enforces either target stratification or documents a temporal/group exception.

## Output Structure

Deliver a structured EDA report containing the following mandatory sections:
1. **Executive Summary:** High-level description of dataset volume, primary task, and core anomalies.
2. **Data-Quality Findings:** Missingness matrices, conversion logs, data loss metrics, and schema adjustments.
3. **Target Profile:** Task definition, baseline statistics, class balance ratios, and split validation mandates.
4. **Feature-by-Feature Findings:** Distribution breakdowns, categorical cardinality flags, and semantic annotations.
5. **Dependency Findings:** Covariance, correlation matrices, and mixed-type association metrics.
6. **Temporal Assessment:** (If applicable) Chronological consistency, intervals, gaps, and drift profiles.
7. **Treatment & Encoding Recommendations:** Mapping of feature types to suggested pre-processing actions.
8. **Leakage & Multiple-Testing Risks:** Explicit warnings regarding suspicious predictors or structural data bleed.
9. **Validation Strategy:** Precise recommendations for cross-validation vs. chronological train/test splits.
10. **Appendix:** Manifest of generated artifact file paths (`plots/correlation_matrix.png`, log files) and technical assumptions.
