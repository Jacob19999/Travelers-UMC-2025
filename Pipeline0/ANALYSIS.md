# Current Methods Analysis & Improvement Opportunities

## Executive Summary

**Current Baseline Performance:**
- **Pipeline 1 Best:** XGBoost with F1 = 0.59272, AUC ≈ 0.837
- **Pipeline 2:** Not yet implemented/run

**Goal:** Beat F1 > 0.59 and AUC > 0.837

---

## 1. CURRENT METHODS IN PIPELINE 1

### 1.1 Data Preprocessing ✅ WORKING

**Methods Used:**
- **Outlier Capping (IQR method):** Caps outliers at Q1-1.5*IQR and Q3+1.5*IQR
- **Missing Value Imputation:** Median imputation for numeric features
- **Low Variance Feature Removal:** Removes features with <1% variance
- **Highly Correlated Feature Removal:** Removes features with correlation >0.95

**Effectiveness:** ✅ **WORKING WELL**
- Removed 24 low variance features
- Removed 16 highly correlated features
- Helps reduce noise and multicollinearity

---

### 1.2 Feature Engineering ⚠️ MIXED RESULTS

**Methods Used:**

1. **Polynomial Features (Degree 3):**
   - Applied to top 10 features
   - Created 20 polynomial features
   - Selected by correlation with target

2. **K-Means Clustering:**
   - 5 clusters on top 10 features
   - Created 5 cluster membership features
   - One-hot encoded clusters

3. **Binning (KBinsDiscretizer):**
   - 5 bins for top 5 continuous features
   - Quantile-based strategy

4. **PCA on Correlated Groups:**
   - Applied to groups of features with correlation >0.7
   - Created 4 PCA components (2 per group, 3 groups max)

**Effectiveness:** ⚠️ **PARTIALLY WORKING**
- Polynomial features appear in top features (poly_63, poly_64, etc.)
- Cluster features are top-ranked (cluster_2 is #1)
- PCA features also appear in top 20
- **Issue:** Feature count testing showed 70 features > 80 features (0.64460 vs 0.64127)
- **Suggests:** Some engineered features may be redundant or noisy

---

### 1.3 Feature Selection ✅ WORKING

**Methods Used:**
- **Tree-based Importance (XGBoost):** Feature importance from gradient boosting
- **Mutual Information:** Information-theoretic feature selection
- **Combined Score:** 70% tree importance + 30% mutual information
- **Final Selection:** Top 70 features (tested 70/80/90, 70 was best)

**Effectiveness:** ✅ **WORKING WELL**
- Combined approach captures both linear and non-linear relationships
- Optimal feature count found through testing
- Selected features show good predictive power

---

### 1.4 Class Imbalance Handling ⚠️ NOT WORKING AS EXPECTED

**Methods Tried:**
1. **SMOTE/BorderlineSMOTE:** Tested but **REMOVED** in final version
2. **scale_pos_weight (XGBoost):** Fine-tuned to 1.60
3. **class_weight='balanced' (LightGBM):** Used in some models
4. **Training on Unbalanced Data:** Final approach

**Effectiveness:** ⚠️ **MIXED**
- **SMOTE was removed** - suggests it didn't help or hurt performance
- **Best approach:** Training on unbalanced data with cost-sensitive learning
- **Finding:** Unbalanced training + threshold optimization > SMOTE

**Key Insight:** The notebook explicitly states "SKIPPING SMOTE - USING ORIGINAL UNBALANCED DATA"

---

### 1.5 Model Training ✅ WORKING

**Models Used:**
1. **XGBoost:** Best performer (F1: 0.59272)
2. **LightGBM:** Second best (F1: 0.55521)
3. **HistGradientBoosting:** F1: 0.56845
4. **ExtraTrees:** F1: 0.55755
5. **GradientBoosting:** Lower performance

**Hyperparameter Optimization:**
- **Optuna** used for hyperparameter tuning
- **Best XGBoost params:** Found through 40+ trials
- **Best LightGBM params:** Found through 11+ trials
- **Early Stopping:** Used to prevent overfitting

**Effectiveness:** ✅ **WORKING WELL**
- XGBoost clearly outperforms other models
- Hyperparameter tuning significantly improved performance
- Early stopping prevents overfitting

---

### 1.6 Probability Calibration ⚠️ NOT WORKING

**Methods Tested:**
1. **Isotonic Calibration:** Tested but often not selected
2. **Sigmoid Calibration:** Tested but often not selected
3. **No Calibration:** Often the best choice

**Effectiveness:** ❌ **NOT HELPING**
- **XGBoost:** Best calibration = "none" (F1: 0.58897)
- **LightGBM:** Best calibration = "none" (F1: 0.55521)
- **HistGBM:** Isotonic helped slightly
- **Finding:** Calibration often hurts performance, suggesting models are already well-calibrated

---

### 1.7 Threshold Optimization ✅ WORKING VERY WELL

**Methods Used:**
1. **PR Curve Iterative:** PR curve + iterative refinement (most robust)
2. **Adaptive Range:** Based on probability distribution
3. **F1 Direct:** Direct F1 maximization
4. **Youden's J:** Sensitivity + Specificity - 1

**Effectiveness:** ✅ **WORKING EXCELLENTLY**
- **Best strategy:** PR curve iterative (selected for XGBoost)
- **Optimal threshold:** ~0.287 (much lower than 0.5!)
- **Key insight:** Threshold optimization is critical for imbalanced data
- This is one of the most impactful techniques

---

### 1.8 Cross-Validation ✅ WORKING

**Method:**
- **10-fold Stratified K-Fold:** Ensures class distribution maintained
- **Out-of-Fold (OOF) Predictions:** Used for model evaluation
- **No data leakage:** Proper train/validation split

**Effectiveness:** ✅ **WORKING WELL**
- Robust evaluation method
- Prevents overfitting to single validation set

---

### 1.9 Ensemble Methods ❌ NOT USED

**Status:** ❌ **EXPLICITLY REMOVED**
- Code comments: "NO ENSEMBLE, just picking the best"
- Cells removed: "This cell has been removed - no ensemble code needed"
- **Current approach:** Single best model (XGBoost)

**Opportunity:** 🎯 **MAJOR IMPROVEMENT OPPORTUNITY**
- Multiple models trained but not combined
- Ensemble could potentially improve F1 score

---

## 2. PIPELINE 2 METHODS (NOT YET RUN)

### 2.1 RankGauss Normalization
- **Status:** Implemented but not tested
- **Method:** Rank-based Gaussian transformation
- **Potential:** Could help with non-normal distributions

### 2.2 Denoising Autoencoder (DAE)
- **Status:** Implemented but not tested
- **Method:** Deep representation learning with swap noise
- **Potential:** Could capture high-order interactions

### 2.3 Neural Network + GBM Ensemble
- **Status:** Planned but not implemented
- **Method:** DAE features + raw features → NN + GBM → Ensemble
- **Potential:** Could beat Pipeline 1 baseline

---

## 3. WHAT WORKS ✅

1. **Outlier Capping (IQR):** Effective noise reduction
2. **Feature Selection (Tree + MI):** Good feature quality
3. **XGBoost Model:** Best individual model
4. **Threshold Optimization:** Critical for F1 score improvement
5. **Hyperparameter Tuning (Optuna):** Significant gains
6. **Training on Unbalanced Data:** Better than SMOTE
7. **10-fold CV:** Robust evaluation
8. **Early Stopping:** Prevents overfitting

---

## 4. WHAT DOESN'T WORK ❌

1. **SMOTE/Oversampling:** Removed from final pipeline
2. **Probability Calibration:** Often hurts performance
3. **Ensemble Methods:** Not used (but could help!)
4. **Too Many Features:** 80 features < 70 features
5. **LightGBM:** Underperforms XGBoost significantly

---

## 5. POTENTIAL IMPROVEMENTS 🎯

### 5.1 Data-Level Improvements

1. **Better Feature Engineering:**
   - **Target Encoding:** Encode categoricals by target mean (could be powerful)
   - **Time-based Features:** Extract more from `claim_date` (day of month, quarter, seasonality)
   - **Domain-Specific Features:**
     - Risk scores based on combinations (e.g., low liability + witness = high subro chance)
     - Vehicle age categories (new vs old)
     - Income-to-claim ratio
   - **Interaction Features:** More targeted interactions (liab_prct × accident_type is already top feature)

2. **Advanced Preprocessing:**
   - **Robust Scaling:** Instead of just outlier capping, use RobustScaler
   - **Quantile Transformation:** Could help with non-normal distributions
   - **Feature Clustering:** Group similar features before selection

3. **Missing Value Strategy:**
   - **Missing as Feature:** Create indicator for missing values
   - **Advanced Imputation:** KNN imputation or model-based imputation

### 5.2 Model-Level Improvements

1. **Ensemble Methods (HIGH PRIORITY):**
   - **Stacking:** Meta-learner on top of base models
   - **Blending:** Weighted average of XGBoost, LightGBM, HistGBM
   - **Voting:** Hard or soft voting ensemble
   - **Optimal Weights:** Use validation set to find best blend weights

2. **Advanced Models:**
   - **CatBoost:** Could perform better than XGBoost/LightGBM
   - **Neural Networks:** Deep learning with embeddings for categoricals
   - **TabNet:** Attention-based tabular learning
   - **NGBoost:** Natural gradient boosting with uncertainty

3. **Hyperparameter Optimization:**
   - **More Trials:** Run Optuna for longer (200+ trials)
   - **Different Search Spaces:** Explore wider ranges
   - **Multi-Objective:** Optimize both F1 and AUC simultaneously

4. **Training Strategy:**
   - **Pseudo-Labeling:** Use confident test predictions to augment training
   - **Adversarial Validation:** Detect train/test distribution shift
   - **Cross-Validation Strategy:** Try GroupKFold or TimeSeriesSplit if applicable

### 5.3 Threshold & Calibration Improvements

1. **Per-Fold Thresholds:** Optimize threshold per CV fold
2. **Cost-Sensitive Threshold:** Consider business cost of false positives/negatives
3. **Threshold Ensembling:** Different thresholds for different models

### 5.4 Pipeline 2 Completion

1. **Run DAE Pipeline:** Complete Pipeline 2 implementation
2. **Feature Combination:** Combine DAE features with Pipeline 1 features
3. **Multi-Model Ensemble:** DAE-NN + DAE-GBM + Pipeline 1 models

---

## 6. RECOMMENDED NEXT STEPS

### Priority 1: Quick Wins (High Impact, Low Effort)
1. **Implement Ensemble:** Blend XGBoost + LightGBM + HistGBM (expected +0.01-0.02 F1)
2. **Target Encoding:** Encode categoricals by target mean
3. **More Feature Interactions:** Focus on top feature combinations
4. **Per-Fold Threshold Optimization:** Optimize threshold per CV fold

### Priority 2: Medium Effort (Medium-High Impact)
1. **CatBoost Model:** Add CatBoost to model mix
2. **Advanced Feature Engineering:** Time features, domain features
3. **Stacking Ensemble:** Meta-learner approach
4. **Complete Pipeline 2:** Run DAE pipeline and combine with Pipeline 1

### Priority 3: High Effort (High Potential Impact)
1. **Neural Network with Embeddings:** Deep learning for tabular data
2. **TabNet:** Attention-based model
3. **Pseudo-Labeling:** Semi-supervised learning
4. **Multi-Objective Optimization:** Balance F1 and AUC

---

## 7. KEY INSIGHTS

1. **Threshold optimization is critical** - Optimal threshold ~0.287 vs default 0.5
2. **Ensemble not used** - Major opportunity for improvement
3. **SMOTE doesn't help** - Unbalanced training + threshold opt > oversampling
4. **Calibration doesn't help** - Models already well-calibrated
5. **Feature count matters** - 70 features > 80 features (sweet spot)
6. **XGBoost dominates** - But ensemble could improve further
7. **Pipeline 2 untested** - DAE approach could be breakthrough

---

## 8. EXPECTED IMPROVEMENTS

| Method | Expected F1 Gain | Effort | Priority |
|--------|-----------------|--------|----------|
| Ensemble (Blending) | +0.01-0.02 | Low | ⭐⭐⭐ |
| Target Encoding | +0.005-0.015 | Low | ⭐⭐⭐ |
| CatBoost | +0.005-0.01 | Medium | ⭐⭐ |
| Stacking | +0.01-0.02 | Medium | ⭐⭐ |
| Pipeline 2 (DAE) | +0.01-0.03 | High | ⭐⭐ |
| Neural Network | +0.005-0.02 | High | ⭐ |
| TabNet | +0.01-0.02 | High | ⭐ |

**Target:** Beat F1 = 0.59272 → Aim for F1 > 0.62 (5% improvement)

---

## 9. PIPELINE 0 FORMULAS & MATHEMATICAL METHODS

### 9.1 Target Encoding with Smoothing

**Frequency Encoding:**
```
freq(category) = count(category) in training set
```

**Target Encoding (with CV):**
For each CV fold:
```
category_mean = mean(y | category) in training fold
category_count = count(category) in training fold
global_mean = mean(y) in training fold

smoothing_factor = 1 / (1 + exp(-(category_count - smoothing) / smoothing))

smoothed_mean = global_mean × (1 - smoothing_factor) + category_mean × smoothing_factor
```

**Default Parameters:**
- `smoothing = 1.0`
- `cv_folds = 5`

### 9.2 Feature Importance Combination

**Normalization:**
```
tree_importance_norm = (tree_importance - min(tree_importance)) / (max(tree_importance) - min(tree_importance) + ε)
mi_score_norm = (mi_score - min(mi_score)) / (max(mi_score) - min(mi_score) + ε)
```

**Combined Score:**
```
combined_score = 0.65 × tree_importance_norm + 0.35 × mi_score_norm
```

### 9.3 Ensemble Blending

**Weight Normalization:**
```
weights_normalized = weights / (sum(weights) + ε)
```

**Blended Prediction:**
```
y_blended = Σ(weight_i × prediction_i) for all models i
```

**Optimization Objective:**
```
minimize: -F1_score(y_true, (y_blended >= threshold).astype(int))
subject to: Σ(weights) = 1, weights ≥ 0
```

**Optimization Method:** Differential Evolution (DE)
- `maxiter = 100`
- `popsize = 15`
- `tol = 1e-6`

### 9.4 Threshold Optimization

**F1 Score Calculation:**
```
F1 = 2 × (precision × recall) / (precision + recall + ε)
```

**Optimal Threshold from PR Curve:**
```
precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
f1_scores = 2 × (precision × recall) / (precision + recall + ε)
optimal_threshold = thresholds[argmax(f1_scores)]
```

**Rank-Based Threshold:**
```
target_positive_rate = 0.27  # 27% based on test prevalence
sorted_probs = sort(y_proba)
target_idx = int(len(sorted_probs) × (1 - target_positive_rate))
rank_threshold = sorted_probs[target_idx]
```

**Final Threshold (Weighted Average):**
```
if |rank_threshold - cv_threshold| > 0.1:
    final_threshold = 0.7 × rank_threshold + 0.3 × cv_threshold
else:
    final_threshold = 0.5 × rank_threshold + 0.5 × cv_threshold
```

### 9.5 Domain-Specific Feature Formulas

**Recoverable Amount:**
```
recoverable_amount = (100 - liab_prct) / 100 × claim_est_payout
```

**Fault Clarity:**
```
fault_clarity = |liab_prct - 50| / 50
```

**Evidence Score:**
```
evidence_score = witness_present_ind × 2 + policy_report_filed_ind
```

**Risk Scores:**
```
risk_score_witness_liab = (100 - liab_prct) × witness_encoded
risk_score_police_liab = (100 - liab_prct) × policy_report_filed_ind
risk_score_safety_claims = (100 - safety_rating) × (past_num_of_claims + 1)
```

**Ratios:**
```
payout_to_income_ratio = claim_est_payout / (annual_income + 1)
mileage_per_year = vehicle_mileage / (age_of_vehicle + 1)
payout_ratio = claim_est_payout / (vehicle_price + 1)
income_to_car_price = annual_income / (vehicle_price + 1)
```

**High-Value Recoverable Flag:**
```
threshold = quantile(recoverable_amount, 0.75)
high_value_recoverable = (recoverable_amount > threshold) AND (liab_prct < 50)
```

### 9.6 Preprocessing Formulas

**Outlier Capping (IQR Method):**
```
Q1 = quantile(feature, 0.25)
Q3 = quantile(feature, 0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 × IQR
upper_bound = Q3 + 1.5 × IQR
feature_capped = clip(feature, lower_bound, upper_bound)
```

**RankGauss Transformation:**
```
# Using QuantileTransformer
rank_gauss = QuantileTransformer(
    output_distribution='normal',
    n_quantiles=1000,
    random_state=42
)
feature_transformed = rank_gauss.fit_transform(feature)
```

**Variance Threshold:**
```
variance_threshold = 0.01
feature_removed if variance(feature) < variance_threshold
```

**Correlation Pruning:**
```
correlation_threshold = 0.95
feature_removed if |correlation(feature_i, feature_j)| > correlation_threshold
```

### 9.7 Class Imbalance Handling

**Scale Positive Weight:**
```
scale_pos_weight = (1 - positive_rate) / positive_rate
# For test prevalence of 23.156%:
scale_pos_weight = (1 - 0.23156) / 0.23156 = 3.32
```

**Class Weights (for Neural Networks):**
```
class_weight = {
    0: (1 / count(0)) × (total / 2.0),
    1: (1 / count(1)) × (total / 2.0)
}
# Simplified version:
class_weight = {0: 1, 1: 3.32}
```

### 9.8 Pseudo-Labeling

**High Confidence Selection:**
```
high_conf_threshold_pos = 0.95
high_conf_threshold_neg = 0.05
high_conf_indices = where((prob >= 0.95) OR (prob <= 0.05))
```

**Pseudo-Label Assignment:**
```
y_pseudo = (prob >= 0.5).astype(int) for high_conf_indices
```

**Blended Prediction:**
```
final_pred = 0.70 × ensemble_pred + 0.30 × pseudo_label_model_pred
```

### 9.9 Aggregation Features

**Groupby Statistics:**
For each categorical × numeric combination:
```
mean_by_cat = mean(numeric | categorical) in training set
std_by_cat = std(numeric | categorical) in training set
median_by_cat = median(numeric | categorical) in training set
min_by_cat = min(numeric | categorical) in training set
max_by_cat = max(numeric | categorical) in training set
```

**Test Set Application:**
```
test_feature = map(test_categorical, train_statistics_dict).fillna(0)
```

### 9.10 Clustering Features

**KMeans Clustering:**
```
# Select top 10 numeric features by importance
top_features = argsort(importance)[-10:]

# Fit KMeans on training data
kmeans = KMeans(n_clusters=7, random_state=42)
cluster_labels_train = kmeans.fit_predict(train[top_features])
cluster_labels_test = kmeans.predict(test[top_features])

# One-hot encode clusters
cluster_i = (cluster_labels == i).astype(int) for i in range(7)
```

### 9.11 Model Averaging (Seed Bagging)

**XGBoost Seed Bagging:**
```
seeds = [42, 2023, 2024]
for each seed:
    train model with random_state=seed
    get predictions

final_prediction = mean(predictions across all seeds)
```

### 9.12 Calibration (Platt Scaling)

**Sigmoid Calibration:**
```
# Fit logistic regression on OOF predictions
calibrator = LogisticRegression(C=1.0, solver='lbfgs')
calibrator.fit(oof_preds.reshape(-1, 1), y_true)

# Calibrate probabilities
calibrated_prob = calibrator.predict_proba(prob.reshape(-1, 1))[:, 1]
```

---

## 10. FORMULA SUMMARY TABLE

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
