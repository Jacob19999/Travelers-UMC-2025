# Current Methods Analysis & Improvement Opportunities

## Executive Summary

**Current Baseline Performance:**
- **Pipeline 1 Best:** XGBoost with F1 = 0.59272, AUC ≈ 0.837
- **Pipeline 2:** Not yet implemented/run

**Goal:** Beat F1 > 0.59 and AUC > 0.837

## FORMULA SUMMARY TABLE

| Formula Type | Formula | Parameters |
|-------------|---------|------------|
| **Target Encoding Smoothing** | `smoothed_mean = global_mean × (1 - α) + category_mean × α` | `α = 1/(1+exp(-(count-1)/1))` |
| **Feature Importance** | `score = 0.65×tree_norm + 0.35×mi_norm` | Normalized to [0,1] |
| **Ensemble Blending** | `y = Σ(w_i × pred_i)` | `Σw_i = 1, w_i ≥ 0` |
| **F1 Score** | `F1 = 2PR/(P+R)` | P=precision, R=recall |
| **Optimal Threshold** | `argmax(F1(threshold))` | From PR curve |
| **Scale Pos Weight** | `(1-p)/p` | p = positive rate |
| **Recoverable Amount** | `(100-liab)/100 × payout` | Domain-specific |
| **Fault Clarity** | `|liab-50|/50` | Range [0,1] |
| **Evidence Score** | `witness×2 + police` | Binary indicators |
| **IQR Outlier Capping** | `clip(x, Q1-1.5IQR, Q3+1.5IQR)` | Q1/Q3 = quartiles |
| **RankGauss** | `QuantileTransformer(output='normal')` | n_quantiles=1000 |

---
