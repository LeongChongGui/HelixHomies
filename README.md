Here's the updated README with the Git LFS information added:

# HelixHomies — README

Documentation: how we predict m⁶A probabilities on test datasets (Tasks 1 & 2)

## Project Overview

This repository implements a reproducible pipeline that converts direct nanopore RNA-seq per-read signals into site-level m⁶A probability scores.

- **Task 1 (intermediate)**: XGBoost-based model (exploration + hyperparameter search)
- **Task 2 (final)**: Random Forest (RF) baseline selected for the final submission for reasons of interpretability and deployment

Final prediction outputs are CSVs with columns: `transcript_id`, `transcript_position`, `score`.

## Contents

- Requirements / environment
- Input data & file layout
- High-level pipeline summary
- Step-by-step instructions (parse → features → train → predict)
- Model specifics (XGBoost vs Random Forest)
- Evaluation, thresholds & saving outputs
- Post-processing & downstream analyses (e.g., DRACH motif check)
- Reproducibility, caveats and recommended improvements

## Requirements / Environment

Minimum Python packages (install with pip):

```bash
pip install numpy pandas matplotlib scikit-learn xgboost imbalanced-learn scipy joblib
```

*Prefer Python ≥ 3.8*

## Input Data & File Layout

### Expected Inputs

- `dataset0.json.gz` — training dataset (JSONL / gzipped JSON lines). Each line has structure mapping `transcript_id -> position -> sequence -> reads`
- `data.info.labelled` — CSV with labelled sites: `transcript_id`, `transcript_position`, `label` (0/1)
- `dataset1.json.gz`, `dataset2.json.gz` — additional test datasets

### Output Examples

Written by the pipeline:

- `helixhomies_dataset0_2.csv` (RF predictions on dataset0)
- `helixhomies_dataset1_2.csv` (RF on dataset1)  
- `helixhomies_dataset2_2.csv` (RF on dataset2)

*Intermediates for XGBoost: `helixhomies_dataset0_1.csv`, etc.*

### Persisted Artifacts

- `ohe.pkl` — fitted OneHotEncoder used for sequence features
- `random_forest_model.pkl` — final RF model (Task 2) - **Note: This file is stored using Git LFS due to its large size**
- `xgb_random_search_cv_results.csv` — XGBoost CV summary (if run)

## High-level Pipeline Summary

1. **Parse JSON**: read gzipped JSON lines; produce a flat table with columns `transcript_id`, `transcript_position`, `sequence`, `reads`
2. **Merge labels** (for dataset0) to produce training table
3. **Feature extraction per site**: aggregate per-read signals into fixed-length numeric descriptors (weighted means, weighted SD, standard summary stats, and difference features between central and flanking positions)
4. **Sequence features**: one-hot encode flanking bases (selected positions in the 5-mer / DRACH context)
5. **Resampling (SMOTE)**: synthetic oversampling applied to training numeric features for RF baseline
6. **Modeling**: 
   - XGBoost used in Task 1 (hyperparameter search & early stopping)
   - Random Forest chosen as final Task 2 model (trained on resampled data)
7. **Evaluation**: ROC-AUC and PR-AUC; decision threshold chosen by maximizing F1 from the precision–recall curve
8. **Save predictions**: CSVs with `transcript_id`, `transcript_position`, `score`

## Step-by-Step Instructions

### a) Parse JSON (`function: parse_json(path)`)

Reads gzipped JSON lines. For each site collects:

- `transcript_id` (string)
- `transcript_position` (int)
- `sequence` (string — window around the tested base)
- `reads` (list of per-read 9-d arrays: dwell time, mean, sd for prev/central/next)

Returns a `pandas.DataFrame` with an original index `orig_idx`.

### b) Merge Labels

Merge `json_df` with `data.info.labelled` on `transcript_id` and `transcript_position`. The merged table is used to get X (features) and y (labels).

### c) Feature Extraction (core functions)

#### `extract_features(reads_list)`:

- Assumes each `reads_list` is an array shape `(n_reads, 9)` with columns in this order: features for previous, central, next positions (dwell, mean, sd, ...)
- For each of the 9 feature columns, computes:
  - (1) dwell-weighted mean & sd when applicable (via `weight_map`)
  - (2) unweighted mean, std, median, min, max
- Computes diff features for (central − previous) and (next − central): mean, std, median, min, max (plus dwell-weighted mean & sd for some diffs)
- Returns a flattened numeric vector (fixed length) for each site

#### `extract_seq_features(df)`:

- Pulls characters at positions [0,1,2,5,6] from sequence to form categorical features (excludes invariant positions 3,4 — A and C of the DRACH motif)

#### `feature_extraction(df, encoder)`:

- Applies `extract_features` to reads column and OneHot encodes sequence positions with a `OneHotEncoder(handle_unknown='ignore')`
- Returns combined numeric + one-hot feature DataFrame

**Notes on the features:**

- Dwell-time weighting improves reliability of per-read means/SDs because reads with longer dwell times carry more signal
- Diff features capture local signal shifts that are informative for modifications

### d) SMOTE (only for Random Forest training)

After engineering numeric features, SMOTE (`imblearn.over_sampling.SMOTE`) is applied to balance the training set (used as RF baseline). The code records which rows are synthetic by adding a `synthetic` column.

### e) Test-time Feature Extraction (`test_feature_extraction(test_df, encoder)`)

Same pipeline as train but uses fitted `OneHotEncoder.transform()` rather than `fit_transform()`.

## Modeling Details

### XGBoost (Task 1 — intermediate submission)

**Purpose**: fast exploration, hyperparameter search and early stopping to get a competitive baseline.

Important config used in code:

```python
XGBClassifier(
    random_state=42,
    tree_method="hist",
    learning_rate=0.05,
    max_depth=4,
    n_estimators=2000,  # with early stopping
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=1,
    reg_lambda=1.0,
    reg_alpha=0.0,
    scale_pos_weight=neg/pos,  # from original labels
    eval_metric="aucpr"
)
```

A `RandomizedSearchCV` over a parameter grid was used in later experiments; final XGBoost was evaluated with ROC-AUC & PR-AUC.

Outputs saved for dataset0/1/2 as `helixhomies_dataset*_1.csv`.

*Note: XGBoost was used as the intermediate model for Task 1. It served as a performance and methodology benchmark.*

### Random Forest (Task 2 — final)

Final chosen model for Task 2:

```python
RandomForestClassifier(
    n_estimators=275,
    max_depth=35,
    class_weight='balanced_subsample',
    random_state=42,
    n_jobs=-1
)
```

Trained on the SMOTE-resampled numeric + one-hot features. Synthetic rows (from SMOTE) are dropped before final RF fit in the code path used for prediction.

RF outputs were saved as `helixhomies_dataset*_2.csv` (final submission).

## Evaluation & Choosing Thresholds

**Metrics used**: ROC-AUC and PR-AUC (average precision). PR-AUC is more informative under heavy class imbalance.

**Decision threshold**: chosen by maximizing F1 on precision_recall_curve from validation/test predictions.

Code example:

```python
prec, rec, thr = precision_recall_curve(y_test, probs)
f1s = 2 * (prec[:-1] * rec[:-1]) / (prec[:-1] + rec[:-1] + 1e-12)
best_idx = np.argmax(f1s)
best_thr = thr[best_idx]
```

PR / ROC curves are plotted for visual inspection. Also report the baseline random precision as `sum(y_test)/len(y_test)` for context.

## Producing Predictions & Saving Artifacts

After training the RF (final), predictions on test datasets are generated as:

```python
rf_pred_0 = json_df.copy()
rf_pred_0['score'] = rf_model.predict_proba(X_test_0)[:, 1]
rf_pred_0_final = rf_pred_0[['transcript_id','transcript_position','score']]
rf_pred_0_final.to_csv('helixhomies_dataset0_2.csv', index=False)
```

Persist models & encoders for reuse:

```python
import joblib
joblib.dump(enc, '/path/to/ohe.pkl')
joblib.dump(rf_model, '/path/to/random_forest_model.pkl')
```

These `.pkl` artifacts enable batch-scoring of new JSON files (the prediction script loads `ohe.pkl`, `random_forest_model.pkl`, runs `test_feature_extraction` and saves CSVs).

**Important Note**: The `random_forest_model.pkl` file is stored using Git LFS due to its large size. Make sure you have Git LFS installed and set up to properly download this file.

## Post-processing & Downstream Analyses

Common downstream checks and analyses (scripts available / recommended):

- **DRACH motif enrichment** among top predicted sites (Fisher's exact test). Confirm central 5-mer matches [D][R][A][C][H]
- **Shared vs sample-specific sites**: compute Jaccard similarity and counts of sites appearing in 1..N samples (threshold-based or top-K)
- **Score distributions**: histogram / boxplot per sample; check calibration differences
- **Heatmaps**: z-scored heatmap of union of top-K sites across samples to visualise shared vs specific sites
- **Per-gene site counts**: identify genes with many predicted sites as candidates for follow-up

Scripts in this repo include plotting functions for these analyses (see `analysis/`).

## Reproducibility & Recommended Best Practices

- **Gene/transcript-level splits**: to prevent information leakage, perform train/test splits by gene/transcript (not by random positions). The code shown used dataset0 for both training and (part of) test — treat `X_test_0` as in-repo validation. For final evaluation, use held-out genes or completely separate datasets (dataset1/2) as true test sets
- **Coverage checks**: predictions at low-read positions are less reliable. When possible, include read depth per-site and filter or downweight low-coverage sites
- **Calibration**: RF scores are not necessarily calibrated probabilities. If you need calibrated probabilities, consider `CalibratedClassifierCV` (Platt / isotonic)
- **Multiple thresholds & robustness**: report results for several thresholds (0.9, 0.7, 0.5) or top-K (top 100/500/1000) and show motif-enrichment robustness across these

## Limitations

- Labels are site-level (from orthogonal biochemical assay) but reads are unlabeled → label noise and MIL (multiple-instance) issues. This pipeline mitigates via per-site aggregation but does not implement read-level MIL or attention pooling. Future work: compare RF baseline with MIL/attention models (e.g., m6Anet-like architectures) to try to directly model per-read heterogeneity
- SMOTE synthetically balances classes but can introduce artifacts; we only used SMOTE for RF baseline. Alternative approaches include class weighting, focal loss (for deep models), and careful negative sampling
- Sequence window centring is assumed; ensure sequence strings in the JSON are centred around the candidate base

## Quick Examples / Commands

### a) Train RF (example)

```python
# after parsing & extracting features
X_train = feature_extraction(merged_data, enc)  # fit encoder inside
X_train_resampled, y_train_resampled = ...  # SMOTE step
rf_model.fit(X_train_resampled.drop(columns=['synthetic']), y_train_resampled)
joblib.dump(rf_model, 'random_forest_model.pkl')
joblib.dump(enc, 'ohe.pkl')
```

### b) Predict on a New JSON File

```python
# load artifacts
enc = joblib.load('ohe.pkl')
rf_model = joblib.load('random_forest_model.pkl')

# parse test json into df_test with same structure (transcript_id,pos,sequence,reads)
X_test = test_feature_extraction(df_test, enc)
scores = rf_model.predict_proba(X_test)[:,1]
df_test['score'] = scores
df_test[['transcript_id','transcript_position','score']].to_csv('predictions_new.csv', index=False)
```

**Note**: The `random_forest_model.pkl` file is stored with Git LFS. Ensure you have Git LFS installed and run `git lfs pull` to download the large model file if it doesn't download automatically.

## Where to Look in the Repo??????????????????????????????????????????

- `parse.py` — JSON parsing functions
- `features.py` — `extract_features`, `extract_seq_features`, `feature_extraction` & `test_feature_extraction`
- `train_xgb.py` — XGBoost training, randomized search scripts (Task 1)
- `train_rf.py` — Random Forest training & SMOTE path (Task 2, final)
- `predict.py` — load `.pkl` artifacts, extract features, save predictions CSVs
- `analysis/` — DRACH enrichment, plots (boxplots, heatmaps, correlation matrices)

## Git LFS Note

This repository uses Git LFS (Large File Storage) to handle large model files. To ensure you can properly download and use the `random_forest_model.pkl` file:

1. Install Git LFS on your system:
   ```bash
   git lfs install
   ```

2. Clone the repository as usual - Git LFS files should download automatically:
   ```bash
   git clone <repository-url>
   ```

3. If the large files don't download automatically, you can pull them manually:
   ```bash
   git lfs pull
   ```

## Final Notes on Innovation & Impact

**Innovation**: The pipeline combines dwell-time-weighted aggregation of per-read signal statistics with local difference features and contextual 5-mer encoding. This is a pragmatic, interpretable strategy that is fast to train and reproducible — and a sensible baseline before moving to MIL / attention models.

**Impact**: Persisting the encoder and RF model as `.pkl` artifacts allows rapid batch re-annotation of new direct-RNA datasets and comparative epitranscriptomics across cell lines. Results can accelerate biomarker discovery and candidate-site prioritisation for experimental validation, while keeping the codebase compact and deployable.
