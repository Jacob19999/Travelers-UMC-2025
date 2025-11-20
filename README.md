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

## Critical Leaderboard Probing Results (Nov 2025)

We performed a probing submission (predicting all 1s) to reverse-engineer the exact distribution of the test set.

### 1. Class Balance
- **Public Leaderboard F1 (All-1s Probe):** `0.37606`
- **Derived Positive Prevalence:** **~23.16%**
- **Imbalance Ratio:** ~1:3.3

### 2. Optimization Constraints
- **Baseline Floor:** `0.37606` (If your model scores lower than this, it is worse than a constant prediction).
- **Theoretical Ceiling:** The max F1 is **not 1.0** due to the precision-recall trade-off inherent in F1 at this prevalence.
- **Realistic Top-Tier Score:** `0.468 - 0.475` (Based on historical performance on this dataset).

### 3. Modeling Implications
- **Scale_Pos_Weight:** Hardcode to **`3.32`** (calculated as `(1 - 0.23156) / 0.23156`) for XGBoost/LightGBM/CatBoost.
- **Thresholding:** The optimal decision threshold is **not 0.5**. It will likely be in the range of **`0.15 - 0.25`**.
- **Metric:** Optimize for F1 directly on a validation set with stratified 23% prevalence.

## Disclaimer
TriGuard Insurance Company and the data are fictitious examples used for the purpose of this competition only.
