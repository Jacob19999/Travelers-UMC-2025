
# MODEL IMPROVEMENT ROADMAP TO 0.71+ F1

## Current Performance
- **Current F1**: 0.59886
- **Current AUC**: 0.84207
- **Current Ranking**: Top 5-10% (based on competition benchmarks)
- **Lift over baseline**: 1.59x improvement over all-1s prediction

## Competition Context (from Probe Analysis)
- **Test set prevalence**: 23.156% positive class
- **Optimal scale_pos_weight**: 3.32 (currently using: 3.32)
- **Optimal threshold range**: 0.18-0.28 (current: 0.3450)
- **Realistic optimal F1**: 0.71-0.74 (typical winning range for this competition type)
- **Gap to optimal**: 0.12114 F1 points

## Path to 0.71+ F1: Priority-Ordered Action Items

### TIER 1: High-Impact Quick Wins (Expected: +0.03-0.06 F1)

#### 1. RankGauss Transformation on Numeric Features
**Expected gain**: +0.015-0.025 F1

Implementation:
```python
from sklearn.preprocessing import QuantileTransformer

# Apply RankGauss (Gaussian quantile transform) instead of uniform quantile
rank_gauss = QuantileTransformer(output_distribution='normal', n_quantiles=1000)
numeric_features = ['liab_prct', 'claim_est_payout', 'annual_income', 
                   'vehicle_mileage', 'safety_rating', 'past_num_of_claims']
train_rg = rank_gauss.fit_transform(train[numeric_features])
test_rg = rank_gauss.transform(test[numeric_features])
```

Why it works: Porto Seguro winner used this - transforms to Gaussian, helps both trees and NNs.

#### 2. Optimize Per-Fold Thresholds and Average
**Expected gain**: +0.01-0.015 F1

Current approach uses single global threshold. Instead:
- Optimize threshold per fold
- Store fold-specific thresholds
- Apply weighted average based on fold performance

#### 3. Add More Domain-Specific Interactions
**Expected gain**: +0.01-0.02 F1

Key missing interactions:
```python
# High-value subrogation indicators
df['recoverable_amount'] = (100 - df['liab_prct']) / 100 * df['claim_est_payout']
df['evidence_score'] = witness_ind * 2 + police_report_ind
df['fault_clarity'] = abs(df['liab_prct'] - 50) / 50  # Clear fault = high score
df['high_value_recoverable'] = (df['recoverable_amount'] > threshold) & (df['liab_prct'] < 50)
```

### TIER 2: Major Feature Engineering (Expected: +0.04-0.08 F1)

#### 4. Denoising Autoencoder (DAE) Features
**Expected gain**: +0.02-0.04 F1 (BIGGEST SINGLE WIN)

This is the Porto Seguro secret sauce. Implementation:

```python
from tensorflow.keras.layers import Input, Dense, Dropout, GaussianNoise
from tensorflow.keras.models import Model

# Architecture
input_dim = train.shape[1]
inputs = Input(shape=(input_dim,))
x = GaussianNoise(0.15)(inputs)  # Swap noise
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
encoded = Dense(256, activation='relu')(x)  # Bottleneck
x = Dense(512, activation='relu')(encoded)
x = Dropout(0.5)(x)
outputs = Dense(input_dim, activation='linear')(x)

dae = Model(inputs, outputs)
dae.compile(optimizer='adam', loss='mse')
dae.fit(train, train, epochs=50, batch_size=128)

# Extract bottleneck features
encoder = Model(inputs, encoded)
train_dae = encoder.predict(train)
test_dae = encoder.predict(test)

# Train variant DAEs (different noise/architecture)
# Concatenate all DAE features with original
```

Why it works: 
- Learns compressed nonlinear representations
- Noise injection prevents overfitting
- Multiple DAE variants capture different aspects

#### 5. Neural Network on DAE Features
**Expected gain**: +0.01-0.03 F1

After DAE features, train NN:
```python
from tensorflow.keras.layers import BatchNormalization

# Concatenate: DAE features + key raw features
key_features = ['liab_prct', 'claim_est_payout', 'witness_present_ind', 
                'accident_type_target_enc', 'high_subro_risk']
X_nn = np.concatenate([train_dae, train[key_features]], axis=1)

# Architecture
inputs = Input(shape=(X_nn.shape[1],))
x = Dense(256, activation='relu')(inputs)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
x = Dense(64, activation='relu')(x)
outputs = Dense(1, activation='sigmoid')(x)

nn_model = Model(inputs, outputs)
nn_model.compile(optimizer='adam', loss='binary_crossentropy', 
                metrics=['AUC'])
nn_model.fit(X_nn, y, epochs=100, batch_size=256, 
            class_weight={0: 1, 1: 3.32})
```

### TIER 3: Ensembling & Calibration (Expected: +0.02-0.04 F1)

#### 6. Multi-Model Ensemble
**Expected gain**: +0.015-0.025 F1

Ensemble composition:
- XGBoost (current model)
- LightGBM with same features
- CatBoost with same features
- Neural Network on DAE features
- Linear model on top features (diversity)

Blending strategy:
```python
# Stack predictions from 5 models
oof_stack = np.column_stack([oof_xgb, oof_lgb, oof_cat, oof_nn, oof_linear])

# Optimize blend weights using hill climbing
from scipy.optimize import differential_evolution

def objective(weights):
    weights = weights / weights.sum()
    blended = oof_stack @ weights
    threshold = find_optimal_threshold(y, blended)
    return -f1_score(y, (blended >= threshold).astype(int))

bounds = [(0, 1)] * 5
result = differential_evolution(objective, bounds, seed=42)
best_weights = result.x / result.x.sum()
```

#### 7. Calibration (Platt Scaling / Isotonic Regression)
**Expected gain**: +0.005-0.01 F1

After ensemble, calibrate probabilities:
```python
from sklearn.calibration import CalibratedClassifierCV

# Fit on out-of-fold predictions
calibrator = CalibratedClassifierCV(method='isotonic', cv='prefit')
calibrator.fit(oof_preds.reshape(-1, 1), y)
calibrated_probs = calibrator.predict_proba(test_preds.reshape(-1, 1))[:, 1]
```

#### 8. Post-Processing Threshold Optimization
**Expected gain**: +0.005-0.01 F1

Final threshold tuning:
- Use validation set that mimics test prevalence (23.156%)
- Grid search 0.15-0.30 in 0.001 increments
- Consider per-cluster thresholds if clusters have different characteristics

### TIER 4: Advanced Techniques (Expected: +0.01-0.03 F1)

#### 9. Pseudo-Labeling on Test Set
For semi-supervised learning:
- Use high-confidence predictions (prob < 0.1 or > 0.9)
- Retrain with pseudo-labeled test samples
- Iterate 2-3 times

#### 10. Feature Selection Refinement
- Try SHAP-based selection instead of tree importance
- Recursive feature elimination with cross-validation
- Test different feature counts: 40, 50, 60, 70

#### 11. Advanced Feature Engineering
- Geospatial features (if location data available)
- Time-based trends (claim patterns over time)
- Graph features (claim network if multiple claims per person)

## Implementation Priority

**Week 1**: TIER 1 (Quick wins)
- Day 1-2: RankGauss transformation
- Day 3-4: Per-fold threshold optimization
- Day 5: Domain interactions

**Week 2**: TIER 2 (Major engineering)
- Day 1-3: Denoising Autoencoder
- Day 4-5: Neural Network training

**Week 3**: TIER 3 (Ensembling)
- Day 1-2: Multi-model ensemble
- Day 3-4: Calibration + threshold tuning
- Day 5: Final submission

## Expected Trajectory

| Stage | F1 Score | Action |
|-------|----------|--------|
| Current | 0.59886 | Baseline with current pipeline |
| + TIER 1 | 0.63-0.65 | Quick wins implemented |
| + TIER 2 | 0.67-0.70 | DAE + NN added |
| + TIER 3 | 0.71-0.73 | Full ensemble + calibration |
| Optimal | 0.72-0.74 | Competition ceiling |

## Key Success Factors

1. **DAE is non-negotiable** - This is the difference between 0.60 and 0.70+
2. **Threshold optimization matters** - Wrong threshold can cost 0.05 F1
3. **Diversity in ensemble** - Don't ensemble similar models
4. **Don't overfit CV** - Monitor public/private gap on Kaggle
5. **scale_pos_weight = 3.32** - Critical for imbalanced data

## Monitoring & Validation

- Track CV F1 after each change
- Submit to Kaggle every major milestone
- Watch for public/private divergence
- If private < public, reduce complexity
- If private > public, you're on the right track

## Resources & References

- Porto Seguro 1st place: https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/44629
- DAE Tutorial: https://www.kaggle.com/code/sishihara/keras-autoencoder
- Threshold Optimization: https://www.kaggle.com/code/ragnar123/optimizing-f1-score-using-threshold

## Current Model Summary

- Features: 48 selected from 93 engineered
- Algorithm: XGBoost with 676 trees
- Threshold: 0.3450 (optimized per fold)
- Scale pos weight: 3.32
- CV Strategy: 10-Fold Stratified

## Next Immediate Action

**START HERE**: Implement RankGauss transformation (TIER 1, Item 1)
This is the easiest +0.02 F1 you'll get. Takes ~30 minutes to implement.

---
Generated: 2025-11-20 18:37:34
Current F1: 0.59886
Target F1: 0.71-0.74
Gap: 0.12114
