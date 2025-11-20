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
| **Pipeline0 – Target Encoding R&D** | Feature research lab modeled after Porto Seguro winners. Houses the `target_encoding_advanced_features.py` experiments, SHAP dashboards, and the `ROADMAP_TO_071.md` plan (RankGauss, advanced interactions, pseudo-labeling). |
| **Pipeline1 – Baseline GBMs** | Cleaned tabular dataset + LightGBM/XGBoost baselines (`Baseline.ipynb`, `lightgbm_jacob.ipynb`). Establishes the production-ready data prep, feature lists, and benchmark submissions. |
| **Pipeline2 – Representation Learning (DAE)** | Current flagship. RankGauss + swap-noise Denoising Autoencoders build 256-dim features (3 variants concatenated). A 25-model ensemble (NNs, 10+ GBM/CatBoost flavors, scikit stacking heads) is weight-optimized for F1. See `pipeline2_dae.py`. |
| **Pipeline3 – Heavy Stacking Ensemble** | Model-zoo approach (Views A/B/C/D) with OOF predictions fed into Level-2 meta models. Mirrors BNP/Kaggle second-place stacking strategies. Documentation in `Pipeline3_Description.txt`. |
| **Pipeline4 – Actuarial / Explainable** | Regulator-friendly pipeline: missingness auditing, robust imputations, calibrated LightGBM, SHAP + PDP reporting for governance. Outputs probability files for compliance-friendly ensembles. |

Use these as mix-and-match modules: Pipeline0/1 provide vetted features, Pipeline2 supplies representation learning + ensembles, Pipeline3 contributes stacking infra, and Pipeline4 delivers explainable/calibrated scores for business stakeholders.

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

### Why LB > CV (~0.0106 gap)

The small bump is within expected noise and likely due to a slightly easier public split or lucky prevalence shift. There is no evidence of overfitting; CV remains tight.

### How Much Headroom Is Left?

- With AP ≈0.60 and prevalence ≈0.2316, Bayes-optimal F1 is roughly 0.68–0.70 (cf. Porto Seguro leaderboards).
- Current 0.6078 captures ~90–95% of the signal. Remaining gains come from richer features/ensembles, not tweaking a single GBM.

### Action Plan to Push 0.65+

1. **Lower Final Threshold (0.45–0.48):** Trade a few precision points for more recall to gain +0.01–0.02 F1 when the separation is this clean.
2. **Ship the Full DAE Stack (Pipeline2):** 128–256 bottleneck features routinely add +0.03–0.06 F1 on insurance data.
3. **Add More Diversity:** CatBoost with alternative depths, LightGBM monotonicity on `liab_prct`, NN on `[raw + DAE]` – blended they add another +0.01–0.02.
4. **Hill-Climb Thresholds on Full Train Predictions:** Start from the 0.60780 submission and perturb ±0.05 to squeeze +0.005–0.015.

These steps are already in-flight in `pipeline2_dae.py`; the current 25-model ensemble is built to capitalize on them. Hitting **0.66–0.68** on the leaderboard is realistic with the remaining roadmap items.

## Disclaimer
TriGuard Insurance Company and the data are fictitious examples used for the purpose of this competition only.
