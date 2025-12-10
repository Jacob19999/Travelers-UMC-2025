# TriGuard Insurance Subrogation Prediction Competition

## Overview
You work for **TriGuard Insurance Company** and are tasked with developing a predictive model to identify potential subrogation opportunities in auto insurance claims using historical claim data from 2020-2021.

**Subrogation** is an important concept in insurance claims. When a policyholder suffers a loss which is not their fault, the claim professional will pay for the loss and refer the case to a subrogation professional. The subrogation professional investigates the loss thoroughly to gather evidence to pursue the at-fault party and attempts to recover the money from the responsible third party.

### Example
Our claimant (Driver A) was backing out of a parking space at the same time as another driver (Driver B). Driver A stopped when they noticed Driver B, but Driver B continued to back out and impacted our claimant. TriGuard will pay Driver A to fix the car and will seek reimbursement from Driver B, who is accountable for the loss.

The goal is to determine which claims have a **high likelihood of successful subrogation**, understand key indicators, and recommend how TriGuard Insurance can use these insights to optimize recovery processes.

## Competition Goals
1. **Identify opportunities** for subrogation in first-party physical damage claims.
2. **Understand key indicators** of subrogation opportunity.
3. **Provide recommendations** on how this information could be leveraged at TriGuard Insurance Company.

## Key Concepts
*   **Claim**: Request made by a policyholder to the insurance company for payment or compensation for a covered loss or event.
*   **First-party physical damage**: Damages sustained by the insured's own property or vehicle.
*   **Subrogation Opportunity**: The possibility that a third party is responsible for the loss and recovery is possible.

## Modeling & Evaluation
Participants are expected to:
1.  **Build a model** to predict the subrogation indicator per claim (Binary Classification: 1 for subrogation opportunity, 0 for none).
2.  **Submit predictions** for the test data as a CSV file.
3.  **Prepare a presentation** for business partners summarizing analysis results and findings.

### Submission Format
The submission file should be a CSV with two columns: `claim_number` and `subrogation` (predicted indicator).

### Benchmark
The score from a simple **XGBoost** model serves as the competition benchmark.

## Dataset
The dataset contains historical claim data from 2020-2021.

*   **`Training_TriGuard.csv`**: Labeled training data including the target variable `subrogation`.
*   **`Testing_TriGuard.csv`**: Unlabeled test data for which predictions must be generated.
*   **`Column Definations.txt`**: Detailed descriptions of all variables in the dataset (e.g., driver demographics, vehicle details, accident specifics).

## Pipeline Portfolio (Directories `Pipeline0` – `Pipeline4`)

| Pipeline | Theme | Highlights |
| --- | --- | --- |
| **Pipeline0 – Target Encoding R&D** | Feature research lab modeled after Porto Seguro winners. Houses the `Pipeline_0.py` experiments, SHAP dashboards, and the `ROADMAP_TO_071.md` plan (RankGauss, advanced interactions, pseudo-labeling). |
| **Pipeline1 – Baseline GBMs** | Cleaned tabular dataset + LightGBM/XGBoost baselines (`Pipeline 1 Version 0.ipynb`, `Pipeline 1 Version 1.ipynb`, `Pipeline 1 Version 2.ipynb`). Establishes the production-ready data prep, feature lists, and benchmark submissions. |
| **Pipeline2 – Representation Learning (DAE)** | Current flagship. RankGauss + swap-noise Denoising Autoencoders build 256-dim features (3 variants concatenated). A 25-model ensemble (NNs, 10+ GBM/CatBoost flavors, scikit stacking heads) is weight-optimized for F1. See `pipeline2_dae.py`. |
| **Pipeline3 – Heavy Stacking Ensemble** | Model-zoo approach (Views A/B/C/D) with OOF predictions fed into Level-2 meta models. Mirrors BNP/Kaggle second-place stacking strategies. Documentation in `Pipeline3_Description.txt`. |
| **Pipeline4 – Actuarial / Explainable** | Regulator-friendly pipeline: missingness auditing, robust imputations, calibrated LightGBM, SHAP + PDP reporting for governance. Outputs probability files for compliance-friendly ensembles. |

Use these as mix-and-match modules: Pipeline0/1 provide vetted features, Pipeline2 supplies representation learning + ensembles, Pipeline3 contributes stacking infra, and Pipeline4 delivers explainable/calibrated scores for business stakeholders.

## Prerequisites

### Core Dependencies
All pipelines require Python 3.7+ and the following base packages:
```bash
pip install pandas numpy scikit-learn scipy matplotlib seaborn
```

### Pipeline-Specific Dependencies

**Pipeline0 (Target Encoding R&D):**
```bash
pip install xgboost lightgbm catboost tensorflow
```

**Pipeline1 (Baseline GBMs):**
```bash
pip install xgboost lightgbm jupyter
```

**Pipeline2 (Representation Learning DAE):**
```bash
pip install tensorflow lightgbm catboost optuna
```

**Pipeline3 (Heavy Stacking Ensemble):**
```bash
pip install xgboost lightgbm
# Optional: tensorflow (for MLP models)
pip install tensorflow
```

**Pipeline4 (Actuarial / Explainable):**
```bash
pip install lightgbm shap
```

### Quick Install (All Pipelines)
```bash
pip install pandas numpy scikit-learn scipy matplotlib seaborn xgboost lightgbm catboost tensorflow optuna shap jupyter
```

## Running the Pipelines

### Pipeline0 – Target Encoding R&D

**Main Script:** `Pipeline0/Pipeline_0.py`

**Run:**
```bash
cd Pipeline0
python Pipeline_0.py
```

**Outputs:**
- Submission CSV files in `submission csv/` folder (e.g., `target_encoding_submission_f1_0_*.csv`)
- Feature analysis results (`feature_analysis_results_*.csv`)
- Visualization plots (`*_plots.png`, `*_dashboard.png`)

**Notes:**
- Implements target encoding with cross-validation to prevent leakage
- Generates advanced features (time-based, risk scores, ratios, interactions)
- Includes threshold optimization for F1 score
- Script auto-installs missing packages if possible

### Pipeline1 – Baseline GBMs

**Main Scripts:** Jupyter Notebooks in `Pipeline1/`

**Run:**
```bash
cd Pipeline1
jupyter notebook
# Then open and run:
# - Pipeline 1 Version 0.ipynb
# - Pipeline 1 Version 1.ipynb
# - Pipeline 1 Version 2.ipynb
```

**Outputs:**
- Submission CSV files in `csv/` folder (e.g., `baseline_submission.csv`, `lightgbm_submission.csv`)
- Feature importance files in `csv/` folder (`*_feature_importance.csv`, `*_feature_importance.png`)
- Performance plots (`*_precision_recall_curve.png`)

**Notes:**
- Establishes production-ready data preprocessing
- Provides baseline LightGBM and XGBoost models
- Includes feature engineering and importance analysis

### Pipeline2 – Representation Learning (DAE)

**Main Script:** `Pipeline2/pipeline2_dae.py`

**Run:**
```bash
cd Pipeline2
python pipeline2_dae.py
```

**Outputs:**
- Submission CSV file (saved to `Data/` or `Output/submissions/` directory)
- Trained model artifacts (if saved)

**Notes:**
- Auto-installs missing packages on first run
- Uses RankGauss normalization + Denoising Autoencoders
- Builds 256-dim features from 3 DAE variants
- Trains 25-model ensemble with F1 weight optimization
- Can take significant time due to ensemble training

### Pipeline3 – Heavy Stacking Ensemble

**Main Script:** `Pipeline3/pipeline3_stacking.py`

**Run:**
```bash
cd Pipeline3
python pipeline3_stacking.py
```

**Outputs:**
- Submission CSV file (saved to `Data/` or `Output/submissions/` directory)
- OOF (Out-of-Fold) predictions for stacking

**Notes:**
- Implements multi-level stacking with Views A/B/C/D
- Uses OOF predictions to train Level-2 meta models
- TensorFlow is optional (for MLP models)
- Mirrors BNP/Kaggle second-place stacking strategies

### Pipeline4 – Actuarial / Explainable

**Main Script:** `Pipeline4/pipeline4_actuarial.py`

**Run:**
```bash
cd Pipeline4
python pipeline4_actuarial.py
```

**Outputs:**
- Submission CSV files (`pipeline4_submission_*.csv`)
- Governance artifacts:
  - `feature_importance_governance.csv`
  - `shap_summary.png`
  - `pdp_plots.png`

**Notes:**
- Regulator-friendly pipeline with missingness auditing
- Uses calibrated LightGBM for probability outputs
- Generates SHAP and Partial Dependence Plots for explainability
- Suitable for compliance and business stakeholder reporting

## Data Requirements

All pipelines expect the following data files in the `Data/` directory:
- `Training_TriGuard.csv` - Training data with target variable
- `Testing_TriGuard.csv` - Test data for predictions
- `Column Definations.txt` - Column descriptions (optional, for reference)

Ensure these files exist before running any pipeline.

## Running All Pipelines

A convenience script is available to run all pipelines sequentially:

**Script:** `run_all_pipelines.py`

**Run:**
```bash
python run_all_pipelines.py
```

**Note:** This will execute all pipeline scripts in order. Ensure all prerequisites are installed before running.

## Output Directory Structure

Pipeline outputs are organized as follows:
- **`Output/submissions/`** - Final submission CSV files from all pipelines
- **`Output/plots/`** - Comparison plots and metrics visualizations
- **`Pipeline0/submission csv/`** - Pipeline0 submission files
- **`Pipeline1/csv/`** - Pipeline1 submission and analysis files
- **`Pipeline4/`** - Pipeline4 governance artifacts (SHAP, PDP plots, feature importance)

## Leaderboard Probe & Dashboard Notes (Model 0.60780)

**Headline:** Public leaderboard F1 jumped from **0.60604 → 0.60780**. Gains are small but meaningful at this stage, and the diagnostic dashboard shows the model is production-ready.

### Key Metrics

- **Leaderboard Public F1:** `0.60780`
- **Validation CV F1 (10 folds):** `0.5974 ± 0.0202`
- **Best Val Threshold:** `0.5038` (F1 `0.5973`) → probabilities are nearly perfectly calibrated.
- **AUC-ROC:** `0.8412`
- **Average Precision:** `0.6026`

### Confusion Matrix @ Threshold 0.5038

| | Pred 0 | Pred 1 |
| --- | --- | --- |
| Actual 0 | **11,181 (TN)** | **2,703 (FP)** |
| Actual 1 | **1,212 (FN)** | **2,903 (TP)** |

- **Precision:** 51.8%
- **Recall:** 70.6%
- **F1:** 59.7%
- Validation prevalence ≈ **22.86%** (matches public probe prevalence 23.156%).

### Plot Interpretation

1. **Precision–Recall (AP 0.6026):** Curve hugs the top-left corner until recall ≈0.7 – elite for this prevalence.
2. **Probability Distribution:** TNs pile below 0.3, TPs peak above 0.7, leaving little ambiguity near 0.5 → explains why threshold ≈0.5 is optimal.
3. **Threshold-vs-F1 Curve:** Classic convex shape peaking at 0.597 F1 near 0.5, confirming strong calibration.
4. **Fold Stability:** F1 varies ±0.02, thresholds 0.48–0.55. No leak/instability detected.
5. 
## Reference
Porto Seguro 1st place: https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/44629
DAE Tutorial: https://www.kaggle.com/code/sishihara/keras-autoencoder
Threshold Optimization: https://www.kaggle.com/code/ragnar123/optimizing-f1-score-using-threshold

## Disclaimer
TriGuard Insurance Company and the data are fictitious examples used for the purpose of this competition only.
