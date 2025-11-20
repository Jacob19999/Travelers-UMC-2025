"""
Target Encoding + Advanced Feature Engineering Pipeline
Recommendation #1 Implementation

This script implements:
1. Target Encoding with Cross-Validation (prevents leakage)
2. Advanced Feature Engineering:
   - Time-based features from claim_date
   - Domain-specific risk scores
   - Ratio features
   - Binned interactions
3. Preprocessing pipeline
4. Model training with threshold optimization
5. Submission generation
"""

import os
import sys
import subprocess
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.feature_selection import SelectFromModel, VarianceThreshold, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.metrics import (roc_auc_score, f1_score, precision_score, recall_score,
                            precision_recall_curve, average_precision_score)
import xgboost as xgb

print("="*80)
print("TARGET ENCODING + ADVANCED FEATURE ENGINEERING PIPELINE")
print("="*80)
print("\n[CHECKPOINT] Pipeline initialization starting...")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, 'Data')

TRAIN_FILE = os.path.join(DATA_DIR, 'Training_TriGuard.csv')
TEST_FILE = os.path.join(DATA_DIR, 'Testing_TriGuard.csv')

TARGET_COL = 'subrogation'
ID_COL = 'claim_number'

print(f"\n[CHECKPOINT] Configuration loaded:")
print(f"  Data directory: {DATA_DIR}")
print(f"  Train file: {TRAIN_FILE}")
print(f"  Test file: {TEST_FILE}")
print("[CHECKPOINT] Ready to begin data loading...")

# ============================================================================
# STEP 1: Load Raw Data
# ============================================================================
print("\n" + "="*80)
print("STEP 1: LOADING RAW DATA")
print("="*80)
print("[CHECKPOINT] Starting Step 1: Loading raw data files...")

print("[CHECKPOINT] Loading training data...")
train_df = pd.read_csv(TRAIN_FILE)
print("[CHECKPOINT] ✓ Training data loaded")

print("[CHECKPOINT] Loading test data...")
test_df = pd.read_csv(TEST_FILE)
test_original = test_df.copy()
print("[CHECKPOINT] ✓ Test data loaded")

print(f"\n[CHECKPOINT] Data shapes:")
print(f"  Train shape: {train_df.shape}")
print(f"  Test shape: {test_df.shape}")
print(f"  Train columns: {len(train_df.columns)} columns")

print("[CHECKPOINT] Extracting target variable...")
y = train_df[TARGET_COL].astype(int)
train_df = train_df.drop(columns=[TARGET_COL, ID_COL], errors='ignore')
test_df = test_df.drop(columns=[ID_COL], errors='ignore')
print("[CHECKPOINT] ✓ Target extracted and ID columns removed")

print(f"\n[CHECKPOINT] Class distribution:")
print(y.value_counts(normalize=True))
print("[CHECKPOINT] ✓ Step 1 complete: Data loaded successfully")

# ============================================================================
# STEP 2: Target Encoding with Cross-Validation
# ============================================================================
print("\n" + "="*80)
print("STEP 2: TARGET ENCODING WITH CROSS-VALIDATION")
print("="*80)
print("[CHECKPOINT] Starting Step 2: Target encoding with cross-validation...")
print("[CHECKPOINT] This step may take a few minutes (5-fold CV for each categorical column)...")

def target_encode_cv(train_df, test_df, y, categorical_cols, cv_folds=5, smoothing=1.0, random_state=42):
    """
    Target encoding with cross-validation to prevent leakage.
    
    Args:
        train_df: Training dataframe
        test_df: Test dataframe
        y: Target variable
        categorical_cols: List of categorical column names
        cv_folds: Number of CV folds
        smoothing: Smoothing parameter (higher = more regularization)
        random_state: Random seed
    """
    train_encoded = train_df.copy()
    test_encoded = test_df.copy()
    
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    for col_idx, col in enumerate(categorical_cols, 1):
        if col not in train_df.columns:
            continue
            
        print(f"[CHECKPOINT]  Encoding column {col_idx}/{len(categorical_cols)}: {col}...")
        
        train_encoded[f'{col}_target_enc'] = 0.0
        test_encoded[f'{col}_target_enc'] = 0.0
        
        # Frequency encoding (count of each category)
        train_freq = train_df[col].value_counts().to_dict()
        train_encoded[f'{col}_freq'] = train_df[col].map(train_freq).fillna(0)
        test_encoded[f'{col}_freq'] = test_df[col].map(train_freq).fillna(0)
        
        global_mean = y.mean()
        
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(train_df, y)):
            if fold_idx == 0:
                print(f"[CHECKPOINT]    Processing CV fold {fold_idx + 1}/{cv_folds}...")
            train_fold = train_df.iloc[train_idx]
            val_fold = train_df.iloc[val_idx]
            y_train_fold = y.iloc[train_idx]
            
            category_means = y_train_fold.groupby(train_fold[col]).mean()
            category_counts = y_train_fold.groupby(train_fold[col]).count()
            
            smoothing_factor = 1 / (1 + np.exp(-(category_counts - smoothing) / smoothing))
            smoothed_means = global_mean * (1 - smoothing_factor) + category_means * smoothing_factor
            
            val_values = val_fold[col].map(smoothed_means).fillna(global_mean)
            train_encoded.loc[val_idx, f'{col}_target_enc'] = val_values
        
        test_category_means = y.groupby(train_df[col]).mean()
        test_category_counts = y.groupby(train_df[col]).count()
        test_smoothing_factor = 1 / (1 + np.exp(-(test_category_counts - smoothing) / smoothing))
        test_smoothed_means = global_mean * (1 - test_smoothing_factor) + test_category_means * test_smoothing_factor
        
        test_values = test_df[col].map(test_smoothed_means).fillna(global_mean)
        test_encoded[f'{col}_target_enc'] = test_values
        
        print(f"[CHECKPOINT]    ✓ {col} encoding complete (target + frequency)")
    
    return train_encoded, test_encoded

print("[CHECKPOINT] Identifying categorical columns...")
categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()
print(f"[CHECKPOINT] Found {len(categorical_cols)} categorical columns to encode: {categorical_cols}")

print("[CHECKPOINT] Beginning target encoding process...")
train_encoded, test_encoded = target_encode_cv(
    train_df, test_df, y, categorical_cols, cv_folds=5, smoothing=1.0
)

print(f"\n[CHECKPOINT] ✓ Step 2 complete: Target encoding finished")
print(f"[CHECKPOINT] Added {len(categorical_cols)} target-encoded features.")

# ============================================================================
# STEP 3: Advanced Feature Engineering
# ============================================================================
print("\n" + "="*80)
print("STEP 3: ADVANCED FEATURE ENGINEERING")
print("="*80)
print("[CHECKPOINT] Starting Step 3: Advanced feature engineering...")

def create_time_features(df, date_col='claim_date'):
    """Extract time-based features from claim_date"""
    if date_col not in df.columns:
        return df
    
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], format='%m/%d/%Y', errors='coerce')
    
    df[f'{date_col}_year'] = df[date_col].dt.year
    df[f'{date_col}_month'] = df[date_col].dt.month
    df[f'{date_col}_day'] = df[date_col].dt.day
    df[f'{date_col}_dayofweek'] = df[date_col].dt.dayofweek
    df[f'{date_col}_quarter'] = df[date_col].dt.quarter
    df[f'{date_col}_weekofyear'] = df[date_col].dt.isocalendar().week
    df[f'{date_col}_is_weekend'] = (df[date_col].dt.dayofweek >= 5).astype(int)
    
    df[f'{date_col}_days_since_epoch'] = (df[date_col] - pd.Timestamp('2020-01-01')).dt.days
    
    return df

def create_domain_features(df):
    """Create domain-specific features for insurance claims"""
    df = df.copy()
    
    if 'liab_prct' in df.columns and 'witness_present_ind' in df.columns:
        witness_encoded = df['witness_present_ind'].map({'Y': 1, 'N': 0}).fillna(0)
        df['risk_score_witness_liab'] = (100 - df['liab_prct']) * witness_encoded
        df['high_subro_risk'] = ((df['liab_prct'] < 50) & (witness_encoded == 1)).astype(int)
    
    if 'liab_prct' in df.columns and 'policy_report_filed_ind' in df.columns:
        df['risk_score_police_liab'] = (100 - df['liab_prct']) * df['policy_report_filed_ind']
    
    if 'claim_est_payout' in df.columns and 'annual_income' in df.columns:
        df['payout_to_income_ratio'] = df['claim_est_payout'] / (df['annual_income'] + 1)
        df['high_payout_ratio'] = (df['payout_to_income_ratio'] > 0.1).astype(int)
    
    if 'vehicle_mileage' in df.columns and 'age_of_vehicle' in df.columns:
        df['mileage_per_year'] = df['vehicle_mileage'] / (df['age_of_vehicle'] + 1)
        df['high_mileage_vehicle'] = (df['mileage_per_year'] > 15000).astype(int)
    
    if 'safety_rating' in df.columns and 'past_num_of_claims' in df.columns:
        df['risk_score_safety_claims'] = (100 - df['safety_rating']) * (df['past_num_of_claims'] + 1)
        df['high_risk_driver'] = ((df['safety_rating'] < 50) & (df['past_num_of_claims'] > 2)).astype(int)
    
    if 'age_of_vehicle' in df.columns:
        df['vehicle_age_category'] = pd.cut(
            df['age_of_vehicle'],
            bins=[-1, 2, 5, 10, 100],
            labels=[0, 1, 2, 3]
        ).astype(int)
        df['is_new_vehicle'] = (df['age_of_vehicle'] <= 2).astype(int)
        df['is_old_vehicle'] = (df['age_of_vehicle'] > 10).astype(int)
    
    if 'liab_prct' in df.columns:
        df['liab_category'] = pd.cut(
            df['liab_prct'],
            bins=[-1, 25, 50, 75, 101],
            labels=[0, 1, 2, 3]
        ).astype(int)
        df['low_liability'] = (df['liab_prct'] < 25).astype(int)
        df['high_liability'] = (df['liab_prct'] > 75).astype(int)
    
    if 'claim_est_payout' in df.columns:
        df['payout_log'] = np.log1p(df['claim_est_payout'])
        try:
            df['payout_category'] = pd.qcut(
                df['claim_est_payout'],
                q=5,
                duplicates='drop',
                labels=False
            )
            df['payout_category'] = df['payout_category'].fillna(-1).astype(int)
        except (ValueError, TypeError):
            df['payout_category'] = 0
    
    if 'annual_income' in df.columns:
        df['income_log'] = np.log1p(df['annual_income'])
        try:
            df['income_category'] = pd.qcut(
                df['annual_income'],
                q=5,
                duplicates='drop',
                labels=False
            )
            df['income_category'] = df['income_category'].fillna(-1).astype(int)
        except (ValueError, TypeError):
            df['income_category'] = 0
    
    return df

def create_interaction_features(df):
    """Create interaction features between important variables"""
    df = df.copy()
    
    if 'liab_prct' in df.columns and 'accident_type' in df.columns:
        if df['accident_type'].dtype == 'object':
            accident_encoded = pd.get_dummies(df['accident_type'], prefix='accident_type', drop_first=True)
            for col in accident_encoded.columns:
                df[f'liab_x_{col}'] = df['liab_prct'] * accident_encoded[col]
    
    if 'liab_prct' in df.columns and 'accident_site' in df.columns:
        if df['accident_site'].dtype == 'object':
            site_encoded = pd.get_dummies(df['accident_site'], prefix='accident_site', drop_first=True)
            for col in site_encoded.columns:
                df[f'liab_x_{col}'] = df['liab_prct'] * site_encoded[col]
    
    if 'safety_rating' in df.columns and 'past_num_of_claims' in df.columns:
        df['safety_x_claims'] = df['safety_rating'] * df['past_num_of_claims']
        df['safety_plus_claims'] = df['safety_rating'] + df['past_num_of_claims']
    
    if 'vehicle_price' in df.columns and 'vehicle_mileage' in df.columns:
        df['price_x_mileage'] = df['vehicle_price'] * df['vehicle_mileage']
        df['price_per_mile'] = df['vehicle_price'] / (df['vehicle_mileage'] + 1)
    
    return df

print("\n[CHECKPOINT] 3.1: Creating time-based features from claim_date...")
train_encoded = create_time_features(train_encoded, 'claim_date')
test_encoded = create_time_features(test_encoded, 'claim_date')
print("[CHECKPOINT]    ✓ Time features created")

print("\n[CHECKPOINT] 3.2: Creating domain-specific features (risk scores, ratios, categories)...")
train_encoded = create_domain_features(train_encoded)
test_encoded = create_domain_features(test_encoded)
print("[CHECKPOINT]    ✓ Domain features created")

print("\n[CHECKPOINT] 3.3: Creating interaction features...")
train_encoded = create_interaction_features(train_encoded)
test_encoded = create_interaction_features(test_encoded)
print("[CHECKPOINT]    ✓ Interaction features created")

print("\n[CHECKPOINT] 3.4: Creating clustering features...")
# Identify top numeric features for clustering
numeric_cols = train_encoded.select_dtypes(include=[np.number]).columns.tolist()
if len(numeric_cols) > 0:
    # Use top 10 numeric features for clustering
    temp_model = xgb.XGBClassifier(
        n_estimators=50,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss',
        tree_method='hist'
    )
    temp_model.fit(train_encoded[numeric_cols], y)
    importances = temp_model.feature_importances_
    top_features_for_cluster = [numeric_cols[i] for i in np.argsort(importances)[-10:]]
    
    # Create 5 clusters
    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    train_cluster_labels = kmeans.fit_predict(train_encoded[top_features_for_cluster])
    test_cluster_labels = kmeans.predict(test_encoded[top_features_for_cluster])
    
    # Add cluster features
    train_encoded['cluster_id'] = train_cluster_labels
    test_encoded['cluster_id'] = test_cluster_labels
    
    # One-hot encode clusters
    for i in range(n_clusters):
        train_encoded[f'cluster_{i}'] = (train_cluster_labels == i).astype(int)
        test_encoded[f'cluster_{i}'] = (test_cluster_labels == i).astype(int)
    
    print(f"[CHECKPOINT]    ✓ Created {n_clusters} cluster membership features")
else:
    print("[CHECKPOINT]    ⚠️  No numeric features available for clustering")

print("\n[CHECKPOINT] 3.5: Creating aggregation features...")
# Create groupby statistics for numeric features grouped by original categoricals
# Use original categorical columns (they still exist at this point, dropped in Step 4)
original_cat_cols = [col for col in categorical_cols if col in train_encoded.columns]
numeric_cols_for_agg = train_encoded.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols_for_agg = [col for col in numeric_cols_for_agg 
                        if col not in ['cluster_id'] 
                        and not col.startswith('cluster_')
                        and not col.endswith('_target_enc')
                        and not col.endswith('_freq')]

if len(original_cat_cols) > 0 and len(numeric_cols_for_agg) > 0:
    # Use top 3 categoricals and top 5 numeric features for aggregation
    top_cats = original_cat_cols[:3]
    top_nums = numeric_cols_for_agg[:5]
    
    for cat_col in top_cats:
        for num_col in top_nums:
            if cat_col in train_encoded.columns and num_col in train_encoded.columns:
                try:
                    # Mean aggregation
                    train_agg = train_encoded.groupby(cat_col)[num_col].mean().to_dict()
                    train_encoded[f'{num_col}_mean_by_{cat_col}'] = train_encoded[cat_col].map(train_agg).fillna(0)
                    test_encoded[f'{num_col}_mean_by_{cat_col}'] = test_encoded[cat_col].map(train_agg).fillna(0)
                    
                    # Std aggregation
                    train_std = train_encoded.groupby(cat_col)[num_col].std().fillna(0).to_dict()
                    train_encoded[f'{num_col}_std_by_{cat_col}'] = train_encoded[cat_col].map(train_std).fillna(0)
                    test_encoded[f'{num_col}_std_by_{cat_col}'] = test_encoded[cat_col].map(train_std).fillna(0)
                except:
                    continue
    
    print(f"[CHECKPOINT]    ✓ Created aggregation features (mean/std) for {len(top_cats)} categoricals × {len(top_nums)} numerics")
else:
    print("[CHECKPOINT]    ⚠️  Insufficient features for aggregation")

print(f"\n[CHECKPOINT] ✓ Step 3 complete: Feature engineering finished")
print(f"[CHECKPOINT] Feature counts:")
print(f"  Train features: {train_encoded.shape[1]}")
print(f"  Test features: {test_encoded.shape[1]}")

# ============================================================================
# STEP 4: Handle Categorical Columns
# ============================================================================
print("\n" + "="*80)
print("STEP 4: HANDLING CATEGORICAL COLUMNS")
print("="*80)
print("[CHECKPOINT] Starting Step 4: Removing original categorical columns...")
print("[CHECKPOINT] (Keeping target-encoded versions)")

cols_to_drop = categorical_cols.copy()
if 'claim_date' in train_encoded.columns:
    cols_to_drop.append('claim_date')

print(f"[CHECKPOINT] Dropping {len(cols_to_drop)} original categorical columns...")
for col in cols_to_drop:
    if col in train_encoded.columns:
        train_encoded = train_encoded.drop(columns=[col])
        if col in test_encoded.columns:
            test_encoded = test_encoded.drop(columns=[col])

print(f"[CHECKPOINT] ✓ Step 4 complete: Original categorical columns removed")
print(f"[CHECKPOINT] Remaining features: {train_encoded.shape[1]}")

# ============================================================================
# STEP 5: Preprocessing Pipeline
# ============================================================================
print("\n" + "="*80)
print("STEP 5: PREPROCESSING PIPELINE")
print("="*80)
print("[CHECKPOINT] Starting Step 5: Preprocessing pipeline...")

def cap_outliers_iqr(df, factor=1.5):
    """Cap outliers using IQR method"""
    df_capped = df.copy()
    capped_count = 0
    
    for col in df.columns:
        if df[col].dtype in [np.int64, np.float64]:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR > 0:
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                
                outliers_before = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                df_capped[col] = df_capped[col].clip(lower=lower_bound, upper=upper_bound)
                
                if outliers_before > 0:
                    capped_count += outliers_before
    
    return df_capped, capped_count

print("\n[CHECKPOINT] 5.1: Outlier capping (IQR method)...")
train_processed, outliers_capped = cap_outliers_iqr(train_encoded, factor=1.5)
test_processed, outliers_capped_test = cap_outliers_iqr(test_encoded, factor=1.5)
print(f"[CHECKPOINT]    ✓ Capped {outliers_capped} outliers in training data")
print(f"[CHECKPOINT]    ✓ Capped {outliers_capped_test} outliers in test data")

print("\n[CHECKPOINT] 5.2: Missing value imputation (median strategy)...")
numeric_cols = train_processed.select_dtypes(include=[np.number]).columns
imputer = SimpleImputer(strategy='median')
train_processed = pd.DataFrame(
    imputer.fit_transform(train_processed),
    columns=train_processed.columns,
    index=train_processed.index
)
test_processed = pd.DataFrame(
    imputer.transform(test_processed),
    columns=test_processed.columns,
    index=test_processed.index
)
print("[CHECKPOINT]    ✓ Missing values imputed")

print("\n[CHECKPOINT] 5.3: Removing low variance features...")
variance_selector = VarianceThreshold(threshold=0.01)
train_var_filtered = variance_selector.fit_transform(train_processed)
test_var_filtered = variance_selector.transform(test_processed)
low_var_features = train_processed.columns[~variance_selector.get_support()].tolist()
if len(low_var_features) > 0:
    print(f"[CHECKPOINT]    ✓ Removed {len(low_var_features)} low variance features")
    train_processed = pd.DataFrame(
        train_var_filtered,
        columns=train_processed.columns[variance_selector.get_support()]
    )
    test_processed = pd.DataFrame(
        test_var_filtered,
        columns=test_processed.columns[variance_selector.get_support()]
    )
else:
    print("[CHECKPOINT]    ✓ No low variance features to remove")

print("\n[CHECKPOINT] 5.4: Removing highly correlated features...")
corr_matrix = train_processed.corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr_features = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
if len(high_corr_features) > 0:
    print(f"[CHECKPOINT]    ✓ Removed {len(high_corr_features)} highly correlated features")
    train_processed = train_processed.drop(columns=high_corr_features)
    test_processed = test_processed.drop(columns=[col for col in high_corr_features if col in test_processed.columns])
else:
    print("[CHECKPOINT]    ✓ No highly correlated features to remove")

print(f"\n[CHECKPOINT] ✓ Step 5 complete: Preprocessing finished")
print(f"[CHECKPOINT] Final feature count: {train_processed.shape[1]}")

# Align test columns with train
test_processed = test_processed.reindex(columns=train_processed.columns, fill_value=0)

# Handle any remaining NaN/inf
train_processed = train_processed.replace([np.inf, -np.inf], np.nan).fillna(0)
test_processed = test_processed.replace([np.inf, -np.inf], np.nan).fillna(0)

# ============================================================================
# STEP 6: Feature Selection
# ============================================================================
print("\n" + "="*80)
print("STEP 6: FEATURE SELECTION")
print("="*80)
print("[CHECKPOINT] Starting Step 6: Feature selection...")
print("[CHECKPOINT] This step may take a minute...")

print("\n[CHECKPOINT] 6.1: Computing tree-based feature importance...")
selector_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss',
    tree_method='hist'
)
selector_model.fit(train_processed, y)
print("[CHECKPOINT]    ✓ Tree importance computed")

print("\n[CHECKPOINT] 6.2: Computing mutual information scores...")
mi_scores = mutual_info_classif(train_processed, y, random_state=42, n_neighbors=5)
print("[CHECKPOINT]    ✓ Mutual information computed")

feature_importance_df = pd.DataFrame({
    'feature': train_processed.columns,
    'tree_importance': selector_model.feature_importances_,
    'mi_score': mi_scores
})

feature_importance_df['tree_importance_norm'] = (
    (feature_importance_df['tree_importance'] - feature_importance_df['tree_importance'].min()) /
    (feature_importance_df['tree_importance'].max() - feature_importance_df['tree_importance'].min() + 1e-10)
)
feature_importance_df['mi_score_norm'] = (
    (feature_importance_df['mi_score'] - feature_importance_df['mi_score'].min()) /
    (feature_importance_df['mi_score'].max() - feature_importance_df['mi_score'].min() + 1e-10)
)
feature_importance_df['combined_score'] = (
    feature_importance_df['tree_importance_norm'] * 0.7 +
    feature_importance_df['mi_score_norm'] * 0.3
)
feature_importance_df = feature_importance_df.sort_values('combined_score', ascending=False)

print(f"\n[CHECKPOINT] 6.3: Combining importance scores and selecting top features...")
print(f"\n[CHECKPOINT] Top 20 features:")
print(feature_importance_df.head(20)[['feature', 'combined_score']].to_string(index=False))

desired_feature_count = 50
available_feature_count = len(feature_importance_df)
best_feature_count = min(desired_feature_count, available_feature_count)

print(f"[CHECKPOINT]    Available features: {available_feature_count}")
print(f"[CHECKPOINT]    Desired features: {desired_feature_count}")
print(f"[CHECKPOINT]    Selecting top {best_feature_count} features...")

top_features = feature_importance_df.head(best_feature_count)['feature'].tolist()

selector = SelectFromModel(selector_model, max_features=best_feature_count, threshold=-np.inf)
train_selected = selector.fit_transform(train_processed, y)
test_selected = selector.transform(test_processed)

train_selected = pd.DataFrame(train_selected, columns=top_features, index=train_processed.index)
test_selected = pd.DataFrame(test_selected, columns=top_features, index=test_processed.index)

print(f"[CHECKPOINT] ✓ Step 6 complete: Selected top {best_feature_count} features")

# ============================================================================
# STEP 7: Load Best Hyperparameters (Hard-coded from 1500-trial optimization)
# ============================================================================
print("\n" + "="*80)
print("STEP 7: LOADING BEST HYPERPARAMETERS")
print("="*80)
print("[CHECKPOINT] Starting Step 7: Loading best hyperparameters from previous optimization...")
print("[CHECKPOINT] Using hard-coded best parameters from 1500-trial Optuna optimization")

# Best parameters found from 1500-trial Optuna optimization
# These parameters achieved F1: 0.59354 (final evaluation with 10-fold CV)
best_params = {
    'n_estimators': 676,
    'max_depth': 3,
    'learning_rate': 0.012270473488372841,
    'subsample': 0.700861220955029,
    'colsample_bytree': 0.6090367723004863,
    'gamma': 4.140593823608315,
    'min_child_weight': 7,
    'reg_alpha': 1.5561708103069546e-06,
    'reg_lambda': 6.035110051538266e-05,
    'scale_pos_weight': 2.3317808523652226,
    'random_state': 42,
    'eval_metric': 'logloss',
    'tree_method': 'hist'
}

print(f"\n[CHECKPOINT] ✓ Step 7 complete: Best hyperparameters loaded")
print(f"[CHECKPOINT] Best parameters: {best_params}")

# ============================================================================
# STEP 8: Train Final Model with Threshold Optimization
# ============================================================================
print("\n" + "="*80)
print("STEP 8: TRAIN FINAL MODEL")
print("="*80)
print("[CHECKPOINT] Starting Step 8: Training final model with threshold optimization...")
print("[CHECKPOINT] This step will take a few minutes (10-fold CV)...")

print("\n[CHECKPOINT] 8.1: Computing out-of-fold predictions (10-fold CV)...")
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
oof_preds = np.zeros(len(y))

for fold_idx, (train_idx, val_idx) in enumerate(cv.split(train_selected, y)):
    print(f"[CHECKPOINT]    Training fold {fold_idx + 1}/10...")
    X_train_fold = train_selected.iloc[train_idx]
    X_val_fold = train_selected.iloc[val_idx]
    y_train_fold = y.iloc[train_idx]
    y_val_fold = y.iloc[val_idx]
    
    model = xgb.XGBClassifier(**best_params)
    model.fit(
        X_train_fold, y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        verbose=False
    )
    
    fold_probs = model.predict_proba(X_val_fold)[:, 1]
    oof_preds[val_idx] = fold_probs
    print(f"[CHECKPOINT]    ✓ Fold {fold_idx + 1}/10 complete")

print("\n[CHECKPOINT] ✓ Out-of-fold predictions computed")
print("\n[CHECKPOINT] 8.2: Optimizing threshold for F1 score...")
precision, recall, pr_thresholds = precision_recall_curve(y, oof_preds)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
f1_scores = np.nan_to_num(f1_scores, nan=0.0)

best_idx = np.argmax(f1_scores)
optimal_threshold = pr_thresholds[best_idx] if best_idx < len(pr_thresholds) else 0.3

prob_lower = np.percentile(oof_preds, 5)
prob_upper = np.percentile(oof_preds, 95)
adaptive_min = max(0.1, min(prob_lower, optimal_threshold - 0.15))
adaptive_max = min(0.9, max(prob_upper, optimal_threshold + 0.15))

fine_thresholds = np.linspace(adaptive_min, adaptive_max, 201)
best_threshold = optimal_threshold
best_f1 = f1_score(y, (oof_preds >= optimal_threshold).astype(int))

for thresh in fine_thresholds:
    f1 = f1_score(y, (oof_preds >= thresh).astype(int))
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = thresh

print("[CHECKPOINT]    ✓ Threshold optimization complete")

final_f1 = f1_score(y, (oof_preds >= best_threshold).astype(int))
final_auc = roc_auc_score(y, oof_preds)
final_precision = precision_score(y, (oof_preds >= best_threshold).astype(int))
final_recall = recall_score(y, (oof_preds >= best_threshold).astype(int))

print(f"\n[CHECKPOINT] ✓ Step 8.2 complete: Threshold optimized")
print(f"[CHECKPOINT] Performance metrics:")
print(f"  Optimal threshold: {best_threshold:.4f}")
print(f"  F1 Score: {final_f1:.5f}")
print(f"  AUC: {final_auc:.5f}")
print(f"  Precision: {final_precision:.5f}")
print(f"  Recall: {final_recall:.5f}")

print("\n[CHECKPOINT] 8.3: Training final model on full training data...")
final_model = xgb.XGBClassifier(**best_params)
final_model.fit(train_selected, y, verbose=False)
print("[CHECKPOINT]    ✓ Final model trained")

print("\n[CHECKPOINT] 8.4: Generating predictions on test set...")
test_preds_proba = final_model.predict_proba(test_selected)[:, 1]
print("[CHECKPOINT]    ✓ Test probability predictions generated")

# Apply optimal threshold to convert probabilities to binary predictions
test_preds_binary = (test_preds_proba >= best_threshold).astype(int)
print(f"[CHECKPOINT]    ✓ Applied threshold {best_threshold:.4f} to convert to binary predictions")
print(f"[CHECKPOINT]    Binary predictions: {test_preds_binary.sum()} positives ({test_preds_binary.sum()/len(test_preds_binary)*100:.2f}%)")

print(f"\n[CHECKPOINT] ✓ Step 8 complete: Final model training finished")

# ============================================================================
# STEP 9: Generate Submission
# ============================================================================
print("\n" + "="*80)
print("STEP 9: GENERATING SUBMISSION")
print("="*80)
print("[CHECKPOINT] Starting Step 9: Generating submission file...")

print("[CHECKPOINT] Creating submission dataframe with binary predictions (0 or 1)...")
submission = pd.DataFrame({
    'claim_number': test_original[ID_COL],
    'subrogation': test_preds_binary
})

output_file = os.path.join(SCRIPT_DIR, 'target_encoding_submission.csv')
print(f"[CHECKPOINT] Saving submission to: {output_file}")
submission.to_csv(output_file, index=False)

print(f"\n[CHECKPOINT] ✓ Step 9 complete: Submission file saved")
print(f"[CHECKPOINT] Submission file: {output_file}")
print(f"\n[CHECKPOINT] Submission statistics:")
print(f"  Shape: {submission.shape}")
print(f"  Prediction range: [{submission['subrogation'].min()}, {submission['subrogation'].max()}]")
print(f"  Predicted positives (1): {submission['subrogation'].sum()} ({submission['subrogation'].sum()/len(submission)*100:.2f}%)")
print(f"  Predicted negatives (0): {(submission['subrogation'] == 0).sum()} ({(submission['subrogation'] == 0).sum()/len(submission)*100:.2f}%)")

print("\n" + "="*80)
print("🎉 PIPELINE COMPLETE! 🎉")
print("="*80)
print("[CHECKPOINT] All steps completed successfully!")
print(f"\n[CHECKPOINT] Final Performance Summary:")
print(f"  F1 Score: {final_f1:.5f}")
print(f"  AUC: {final_auc:.5f}")
print(f"  Optimal Threshold: {best_threshold:.4f}")
print(f"  Precision: {final_precision:.5f}")
print(f"  Recall: {final_recall:.5f}")
print(f"\n[CHECKPOINT] Baseline Comparison:")
print(f"  Baseline F1: 0.59272")
print(f"  Your F1: {final_f1:.5f}")
print(f"  Improvement: {final_f1 - 0.59272:+.5f} ({((final_f1 - 0.59272) / 0.59272 * 100):+.2f}%)")
print("="*80)
print("[CHECKPOINT] Pipeline execution finished. Check the submission file for results!")

