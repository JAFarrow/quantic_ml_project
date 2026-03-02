# evaluation-and-design.md

## Overview

This document reports:
- 10-fold Stratified K-Fold cross-validation results for all evaluated models
- Final hold-out test set results for the selected production model
- The key design decisions, preprocessing, and feature engineering used to build a reproducible and deployable ML pipeline

---

## Cross-validation results

### CV setup
- Workflow: `ml.evaluate`
- Validation: 10-fold StratifiedKFold
- Metrics captured per model: ROC-AUC mean/std, accuracy mean/std, and total training time
- Seed associated with this representation: 17

Leaderboard (sorted by `cv_auc_mean`):

| model | cv_auc_mean | cv_auc_std | cv_acc_mean | cv_acc_std | total_time_sec |
| --- | ---: | ---: | ---: | ---: | ---: |
| xgboost | 0.999760 | 0.000123 | 0.995597 | 0.000994 | 9.65 |
| random_forest | 0.999647 | 0.000170 | 0.993160 | 0.001474 | 46.70 |
| adaboost | 0.997865 | 0.000214 | 0.979271 | 0.001805 | 215.41 |
| knn | 0.997703 | 0.000847 | 0.981184 | 0.002925 | 42.26 |
| decision_tree | 0.995133 | 0.001319 | 0.990042 | 0.001794 | 9.41 |
| logistic_regression | 0.963237 | 0.002960 | 0.910115 | 0.005042 | 206.67 |
| torch_mlp | 0.948576 | 0.003379 | 0.899266 | 0.009205 | 49.28 |

### Observations
- XGBoost achieved the best overall performance (highest mean AUC and accuracy) while also being among the fastest.
- Random Forest was extremely competitive in AUC but significantly slower.
- Logistic Regression repeatedly hit `max_iter=10000` (via `ConvergenceWarning`), suggesting the solver struggled to converge under this feature space / regularization setup.
- Tree-based models and KNN reached near-perfect separation quickly, indicating strong predictive signal in the engineered feature set.

---

## Final hold-out test evaluation

The hold-out test set was created via an 80/20 stratified split before any model selection or tuning, and was only used once for final evaluation.

### Selected production model
- Model: xgboost

### Test metrics
- Test ROC-AUC: 0.999584
- Test accuracy: 0.995493

### Confusion matrix
[[4197 27] [16 5301]]

Interpreting the matrix in the common sklearn layout `[[TN FP], [FN TP]]`:
- True Negative: 4197
- False Positive: 27
- False Negative: 16
- True Postive: 5301

Test set size: 9,541 samples.

This indicates a very low error rate overall, with slightly more false positives than false negatives at the chosen decision threshold.

---

## Design decisions, preprocessing, and feature engineering

### Pipeline-first design (reproducibility + no leakage)
All preprocessing is kept inside sklearn Pipelines, ensuring:
- Each CV fold learns transformations (imputation/scaling/vectorization) only from its training split
- The same transformations are reused consistently during inference
- The final model can be packaged as a single serialized pipeline artifact for deployment

### Numeric features
- Missing values: median imputation
- Scaling: StandardScaler

Rationale:
- Median imputation is robust to outliers.
- Scaling helps models sensitive to feature magnitudes (e.g., logistic regression, KNN, neural nets) and keeps numeric features comparable.

### Date feature engineering (`FirstSeenDate`)
- Extracted: year and month
- This replaces raw date strings with stable, low-dimensional temporal features.

### Identify feature handling
- Added a missingness flag for `Identify`
- EDA suggested the Identify feature as having some predictive ability, however samples were messy strings that would be difficult to encode.
- Captures predictive signal from presence/absence without relying on high-cardinality raw identifiers.

### Imported symbols summarization (ImportedSymbols)
- Summarized via token counts (aggregate signal)
- Followed by scaling
- This keeps useful “how much is imported” information without exploding dimensionality.

### DLL imports vectorization
- Tokenization: dll_tokenizer (case-folding + DLL regex)
- Vectorizer: CountVectorizer(binary=True) to treat each DLL as a binary indicator

Rationale:
- Presence/absence of a DLL import is often more important than frequency.
- Binary indicators simplify the representation and work well with tree/boosting models.

### Dropped columns
Dropped early so models only see engineered representations:
- `SHA1` (Unique to each record, provides no indication as to malware/goodware, memorization risk)
- `Magic` (No variability across records)
- `PE_TYPE` (No variability across records)
- `SizeOfOptionalHeader` (No variability across records)
- raw target label column

Rationale:
- Avoid leakage / memorization risks (e.g., hash-like identifiers).
- Remove fields that have no predictive ability.

---

## Model selection rationale

XGBoost was chosen for production because it:
- Ranked best in CV by AUC and accuracy
- Generalized strongly to the hold-out test set (AUC 0.999584, accuracy 0.995493)
- Was computationally efficient compared to similarly strong alternatives (notably Random Forest)

## Hyperparameter tuning

- Tuning workflow: `ml.train`, which rebuilds the XGBoost pipeline with the same preprocessing used in evaluation (`build_preprocessor`).
- Search strategy: `RandomizedSearchCV` over tuned ranges for `n_estimators`, `learning_rate`, `max_depth`, `min_child_weight`, `subsample`, `colsample_bytree`, `gamma`, and `reg_lambda`, using 5-fold `StratifiedKFold`, `n_iter=100`, `roc_auc` scoring, and `random_state=17`. The best configuration is refit on the whole dataset before serialization.
- Purpose: this CV-driven search identifies a strong default while keeping compute manageable, and the resulting metadata captures the best CV AUC and parameter values saved with the model artifact for traceability.

## Shipped artifacts

- `artifacts/model.joblib` is the serialized XGBoost pipeline, combining `build_preprocessor` features plus the best estimator refit on the entire dataset after tuning.
- `artifacts/model_metadata.json` captures provenance: `model_name` `xgboost`, `n_rows_total` 47,701, `training_mode` `full_dataset_refit_after_cv_tuning`, and the `tuning` block with RandomizedSearchCV details (`roc_auc`, 5-fold StratifiedKFold, `n_iter` 100) plus the chosen hyperparameters and best CV AUC 0.999772.
- Metadata also lists the expected input columns so downstream callers know which features must be present before calling the inference service.
