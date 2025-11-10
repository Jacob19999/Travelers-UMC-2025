'''
Advanced Subrogation Prediction with Pattern Discovery
Better captures liability patterns through exploratory analysis and multiple transformations.
Uses precision-recall optimization and weighted ensemble with confidence-based rules.
'''

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, precision_recall_curve
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)


def engineer_features_v2(df):
    """Advanced feature engineering based on pattern analysis"""
    df = df.copy()
    
    # === 1. LIABILITY TRANSFORMATION ===
    # The relationship might be non-linear
    if 'liab_prct' in df.columns:
        df['liab_prct_squared'] = df['liab_prct'] ** 2
        df['liab_prct_cubed'] = df['liab_prct'] ** 3
        df['liab_prct_sqrt'] = np.sqrt(df['liab_prct'])
        df['liab_prct_log'] = np.log1p(df['liab_prct'])
        
        # Inverse liability (other party's fault)
        df['other_fault'] = 100 - df['liab_prct']
        df['other_fault_squared'] = df['other_fault'] ** 2
        df['other_fault_log'] = np.log1p(df['other_fault'])
        
        # Binary indicators for key thresholds
        df['zero_liability'] = (df['liab_prct'] == 0).astype(int)
        df['liability_under_10'] = (df['liab_prct'] < 10).astype(int)
        df['liability_under_25'] = (df['liab_prct'] < 25).astype(int)
        df['liability_under_50'] = (df['liab_prct'] < 50).astype(int)
        df['liability_over_75'] = (df['liab_prct'] > 75).astype(int)
        df['liability_over_90'] = (df['liab_prct'] > 90).astype(int)
        df['full_liability'] = (df['liab_prct'] == 100).astype(int)
        
        # Liability ranges
        df['liability_0_10'] = ((df['liab_prct'] >= 0) & (df['liab_prct'] <= 10)).astype(int)
        df['liability_10_30'] = ((df['liab_prct'] > 10) & (df['liab_prct'] <= 30)).astype(int)
        df['liability_30_50'] = ((df['liab_prct'] > 30) & (df['liab_prct'] <= 50)).astype(int)
        df['liability_50_70'] = ((df['liab_prct'] > 50) & (df['liab_prct'] <= 70)).astype(int)
        df['liability_70_90'] = ((df['liab_prct'] > 70) & (df['liab_prct'] <= 90)).astype(int)
        df['liability_90_100'] = ((df['liab_prct'] > 90) & (df['liab_prct'] <= 100)).astype(int)
    else:
        # Create dummy columns if liability doesn't exist
        df['other_fault'] = 0
        df['zero_liability'] = 0
        df['liability_under_25'] = 0
    
    # === 2. EVIDENCE FEATURES ===
    if 'witness_present_ind' in df.columns:
        df['witness_yes'] = (df['witness_present_ind'] == 'Y').astype(int)
        df['witness_no'] = (df['witness_present_ind'] == 'N').astype(int)
    else:
        df['witness_yes'] = 0
        df['witness_no'] = 0
    
    if 'policy_report_filed_ind' in df.columns:
        df['police_report'] = df['policy_report_filed_ind'].astype(int)
    else:
        df['police_report'] = 0
    
    df['evidence_score'] = df['witness_yes'] + df['police_report']
    df['perfect_evidence'] = (df['evidence_score'] == 2).astype(int)
    df['no_evidence'] = (df['evidence_score'] == 0).astype(int)
    
    # === 3. ACCIDENT TYPE ENCODING ===
    if 'accident_type' in df.columns:
        df['multi_clear'] = (df['accident_type'] == 'multi_vehicle_clear').astype(int)
        df['multi_unclear'] = (df['accident_type'] == 'multi_vehicle_unclear').astype(int)
        df['single_car'] = (df['accident_type'] == 'single_car').astype(int)
        df['any_multi'] = df['accident_type'].astype(str).str.contains('multi', na=False).astype(int)
    else:
        df['multi_clear'] = 0
        df['multi_unclear'] = 0
        df['single_car'] = 0
        df['any_multi'] = 0
    
    # === 4. CRITICAL INTERACTIONS ===
    # These are the most important for subrogation
    if 'liability_under_25' in df.columns and 'multi_clear' in df.columns:
        df['low_liab_multi_clear'] = df['liability_under_25'] * df['multi_clear']
    
    if 'liability_under_25' in df.columns and 'evidence_score' in df.columns:
        df['low_liab_with_evidence'] = df['liability_under_25'] * df['evidence_score']
        df['low_liab_perfect_evidence'] = df['liability_under_25'] * df['perfect_evidence']
    
    if 'zero_liability' in df.columns and 'any_multi' in df.columns:
        df['zero_liab_multi'] = df['zero_liability'] * df['any_multi']
    
    if 'other_fault' in df.columns and 'evidence_score' in df.columns:
        df['other_fault_evidence'] = df['other_fault'] * df['evidence_score'] / 100
    
    # Polynomial interactions with liability
    if 'liab_prct' in df.columns and 'evidence_score' in df.columns:
        df['liab_evidence_product'] = df['liab_prct'] * df['evidence_score']
        df['liab_evidence_ratio'] = df['evidence_score'] / (df['liab_prct'] + 1)
    
    if 'other_fault' in df.columns and 'any_multi' in df.columns:
        df['other_fault_multi'] = df['other_fault'] * df['any_multi'] / 100
    
    # === 5. ACCIDENT LOCATION ===
    if 'accident_site' in df.columns:
        df['parking'] = (df['accident_site'] == 'Parking Area').astype(int)
        df['highway'] = df['accident_site'].astype(str).str.contains('Highway', na=False).astype(int)
        df['intersection'] = df['accident_site'].astype(str).str.contains('Intersection', na=False).astype(int)
        df['local'] = (df['accident_site'] == 'Local').astype(int)
        df['unknown_site'] = (df['accident_site'] == 'Unknown').astype(int)
        
        # Location-liability interactions
        if 'liability_under_25' in df.columns:
            df['parking_low_liab'] = df['parking'] * df['liability_under_25']
            df['highway_low_liab'] = df['highway'] * df['liability_under_25']
    else:
        df['parking'] = 0
        df['highway'] = 0
        df['intersection'] = 0
        df['local'] = 0
        df['unknown_site'] = 0
    
    # === 6. CLAIM VALUE FEATURES ===
    if 'claim_est_payout' in df.columns:
        df['claim_log'] = np.log1p(df['claim_est_payout'])
        df['claim_sqrt'] = np.sqrt(df['claim_est_payout'])
        
        # Quantile-based features
        q25 = df['claim_est_payout'].quantile(0.25)
        q50 = df['claim_est_payout'].quantile(0.50)
        q75 = df['claim_est_payout'].quantile(0.75)
        
        df['claim_q1'] = (df['claim_est_payout'] <= q25).astype(int)
        df['claim_q2'] = ((df['claim_est_payout'] > q25) & (df['claim_est_payout'] <= q50)).astype(int)
        df['claim_q3'] = ((df['claim_est_payout'] > q50) & (df['claim_est_payout'] <= q75)).astype(int)
        df['claim_q4'] = (df['claim_est_payout'] > q75).astype(int)
        
        # Value-liability interaction
        if 'liability_under_25' in df.columns:
            df['high_value_low_liab'] = df['claim_q4'] * df['liability_under_25']
        if 'other_fault' in df.columns:
            df['claim_times_other_fault'] = df['claim_log'] * df['other_fault'] / 100
    else:
        df['claim_log'] = 0
        df['claim_sqrt'] = 0
        df['claim_q1'] = 0
        df['claim_q2'] = 0
        df['claim_q3'] = 0
        df['claim_q4'] = 0
    
    # === 7. DRIVER FEATURES ===
    if 'year_of_born' in df.columns:
        df['age'] = 2024 - df['year_of_born']
        if 'age_of_DL' in df.columns:
            df['experience'] = df['age'] - df['age_of_DL']
            df['inexperienced'] = (df['experience'] < 5).astype(int)
        else:
            df['experience'] = 0
            df['inexperienced'] = 0
        df['young'] = (df['age'] < 25).astype(int)
        df['senior'] = (df['age'] > 65).astype(int)
    else:
        df['age'] = 0
        df['experience'] = 0
        df['young'] = 0
        df['senior'] = 0
        df['inexperienced'] = 0
    
    # === 8. SAFETY AND HISTORY ===
    if 'safety_rating' in df.columns:
        df['high_safety'] = (df['safety_rating'] >= 80).astype(int)
        df['low_safety'] = (df['safety_rating'] < 50).astype(int)
    else:
        df['high_safety'] = 0
        df['low_safety'] = 0
    
    if 'past_num_of_claims' in df.columns:
        df['no_claims'] = (df['past_num_of_claims'] == 0).astype(int)
        df['many_claims'] = (df['past_num_of_claims'] >= 3).astype(int)
    else:
        df['no_claims'] = 0
        df['many_claims'] = 0
    
    # === 9. VEHICLE FEATURES ===
    if 'age_of_vehicle' in df.columns:
        df['new_vehicle'] = (df['age_of_vehicle'] <= 3).astype(int)
        df['old_vehicle'] = (df['age_of_vehicle'] >= 10).astype(int)
        if 'vehicle_mileage' in df.columns:
            df['mileage_per_year'] = df['vehicle_mileage'] / (df['age_of_vehicle'] + 1)
        else:
            df['mileage_per_year'] = 0
    else:
        df['new_vehicle'] = 0
        df['old_vehicle'] = 0
        df['mileage_per_year'] = 0
    
    # === 10. TIME FEATURES ===
    if 'claim_date' in df.columns:
        try:
            df['claim_date'] = pd.to_datetime(df['claim_date'], errors='coerce')
            df['month'] = df['claim_date'].dt.month.fillna(0)
            df['quarter'] = df['claim_date'].dt.quarter.fillna(0)
        except:
            df['month'] = 0
            df['quarter'] = 0
    else:
        df['month'] = 0
        df['quarter'] = 0
    
    if 'claim_day_of_week' in df.columns:
        weekday_map = {
            'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4,
            'Friday': 5, 'Saturday': 6, 'Sunday': 7
        }
        df['weekday'] = df['claim_day_of_week'].map(weekday_map).fillna(0)
        df['weekend'] = df['weekday'].isin([6, 7]).astype(int)
    else:
        df['weekday'] = 0
        df['weekend'] = 0
    
    # === 11. COMPOSITE SCORES ===
    # Weighted score based on key factors
    if all(col in df.columns for col in ['liab_prct', 'evidence_score', 'multi_clear', 'any_multi', 'single_car']):
        df['subrogation_potential'] = (
            (100 - df['liab_prct']) * 0.5 +  # Other party's fault is most important
            df['evidence_score'] * 15 +
            df['multi_clear'] * 20 +
            df['any_multi'] * 10 -
            df['single_car'] * 20
        )
        
        # Risk-adjusted potential
        if 'claim_log' in df.columns:
            df['risk_adjusted_potential'] = df['subrogation_potential'] * (1 + df['claim_log'] / 20)
        else:
            df['risk_adjusted_potential'] = df['subrogation_potential']
    else:
        df['subrogation_potential'] = 0
        df['risk_adjusted_potential'] = 0
    
    # === 12. ADDITIONAL CATEGORICAL ENCODING ===
    if 'channel' in df.columns:
        df['channel_broker'] = (df['channel'] == 'Broker').astype(int)
        df['channel_online'] = (df['channel'] == 'Online').astype(int)
        df['channel_phone'] = (df['channel'] == 'Phone').astype(int)
    else:
        df['channel_broker'] = 0
        df['channel_online'] = 0
        df['channel_phone'] = 0
    
    if 'gender' in df.columns:
        df['gender_m'] = (df['gender'] == 'M').astype(int)
        df['gender_f'] = (df['gender'] == 'F').astype(int)
    else:
        df['gender_m'] = 0
        df['gender_f'] = 0
    
    if 'living_status' in df.columns:
        df['living_own'] = (df['living_status'] == 'Own').astype(int)
        df['living_rent'] = (df['living_status'] == 'Rent').astype(int)
    else:
        df['living_own'] = 0
        df['living_rent'] = 0
    
    if 'in_network_bodyshop' in df.columns:
        df['network_yes'] = (df['in_network_bodyshop'] == 'yes').astype(int)
    else:
        df['network_yes'] = 0
    
    if 'vehicle_category' in df.columns:
        df['vehicle_compact'] = (df['vehicle_category'] == 'Compact').astype(int)
        df['vehicle_medium'] = (df['vehicle_category'] == 'Medium').astype(int)
        df['vehicle_large'] = (df['vehicle_category'] == 'Large').astype(int)
    else:
        df['vehicle_compact'] = 0
        df['vehicle_medium'] = 0
        df['vehicle_large'] = 0
    
    return df


def main():
    """Main execution function"""
    print("="*80)
    print("ADVANCED SUBROGATION PREDICTION WITH PATTERN DISCOVERY")
    print("="*80)
    
    # Set up paths
    base_path = Path(__file__).parent.parent
    train_path = base_path / 'Data' / 'Training_TriGuard.csv'
    test_path = base_path / 'Data' / 'Testing_TriGuard.csv'
    
    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    test_ids = test_df['claim_number'].copy()
    
    print(f"\nData shapes - Train: {train_df.shape}, Test: {test_df.shape}")
    
    # ==================== EXPLORATORY ANALYSIS ====================
    print("\n" + "="*80)
    print("LIABILITY PATTERN ANALYSIS")
    print("="*80)
    
    # Analyze liability distribution by subrogation
    if 'liab_prct' in train_df.columns:
        for sub_val in [0, 1]:
            liability_stats = train_df[train_df['subrogation'] == sub_val]['liab_prct'].describe()
            print(f"\nSubrogation={sub_val} - Liability % statistics:")
            print(f"  Mean: {liability_stats['mean']:.1f}, Median: {liability_stats['50%']:.1f}")
            print(f"  25th percentile: {liability_stats['25%']:.1f}, 75th percentile: {liability_stats['75%']:.1f}")
        
        # Key insight: Create liability-based segments
        liability_bins = [0, 1, 10, 25, 50, 75, 90, 99, 100]
        train_df['liability_segment'] = pd.cut(train_df['liab_prct'], bins=liability_bins, 
                                                labels=['0', '1-10', '10-25', '25-50', '50-75', '75-90', '90-99', '100'])
        
        print("\nSubrogation rate by liability segment:")
        subrogation_by_liability = train_df.groupby('liability_segment')['subrogation'].agg(['mean', 'count'])
        print(subrogation_by_liability)
    
    # ==================== ADVANCED FEATURE ENGINEERING ====================
    print("\n" + "="*80)
    print("ADVANCED FEATURE ENGINEERING")
    print("="*80)
    
    train_df = engineer_features_v2(train_df)
    test_df = engineer_features_v2(test_df)
    
    print(f"Features after engineering - Train: {train_df.shape[1]}, Test: {test_df.shape[1]}")
    
    # ==================== FEATURE SELECTION ====================
    print("\n" + "="*80)
    print("FEATURE SELECTION")
    print("="*80)
    
    # Select numeric features
    exclude_cols = ['subrogation', 'claim_number', 'claim_date', 'zip_code', 'vehicle_color',
                    'accident_site', 'claim_day_of_week', 'gender', 'living_status', 
                    'channel', 'vehicle_category', 'accident_type', 'witness_present_ind', 
                    'in_network_bodyshop', 'liability_segment', 'year_of_born']
    
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]
    feature_cols = [col for col in feature_cols if train_df[col].dtype in ['int64', 'float64', 'int32', 'float32']]
    
    print(f"Selected {len(feature_cols)} features")
    
    X_train = train_df[feature_cols].fillna(-1)
    y_train = train_df['subrogation']
    X_test = test_df[feature_cols].fillna(-1)
    
    # ==================== MODELS WITH DIFFERENT STRATEGIES ====================
    print("\n" + "="*80)
    print("MODEL TRAINING WITH MULTIPLE STRATEGIES")
    print("="*80)
    
    # Strategy 1: Focus on minority class (subrogation=1)
    lgb_params_focused = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'num_leaves': 31,
        'max_depth': 6,
        'learning_rate': 0.03,
        'n_estimators': 400,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.5,
        'reg_lambda': 0.5,
        'min_child_samples': 20,
        'is_unbalance': True,  # Let LightGBM handle imbalance
        'random_state': 42,
        'verbosity': -1
    }
    
    # Strategy 2: Custom objective for F1 optimization
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb_params_f1 = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 5,
        'learning_rate': 0.03,
        'n_estimators': 400,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'gamma': 0.1,
        'scale_pos_weight': pos_weight,
        'random_state': 42,
        'verbosity': 0
    }
    
    # Strategy 3: CatBoost with auto-balancing
    cat_params_balanced = {
        'iterations': 400,
        'depth': 6,
        'learning_rate': 0.03,
        'l2_leaf_reg': 3,
        'auto_class_weights': 'Balanced',
        'random_seed': 42,
        'verbose': False,
        'eval_metric': 'F1'
    }
    
    # ==================== CROSS-VALIDATION ====================
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    oof_preds = np.zeros((len(X_train), 3))  # Store predictions from 3 models
    test_preds = np.zeros((len(X_test), 3))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"\nFold {fold + 1}/{n_splits}")
        
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # Model 1: LightGBM
        lgb_model = lgb.LGBMClassifier(**lgb_params_focused)
        lgb_model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
        )
        oof_preds[val_idx, 0] = lgb_model.predict_proba(X_val)[:, 1]
        test_preds[:, 0] += lgb_model.predict_proba(X_test)[:, 1] / n_splits
        
        # Model 2: XGBoost
        xgb_model = xgb.XGBClassifier(**xgb_params_f1)
        xgb_model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=False
        )
        oof_preds[val_idx, 1] = xgb_model.predict_proba(X_val)[:, 1]
        test_preds[:, 1] += xgb_model.predict_proba(X_test)[:, 1] / n_splits
        
        # Model 3: CatBoost
        cat_model = CatBoostClassifier(**cat_params_balanced)
        cat_model.fit(
            X_tr, y_tr,
            eval_set=(X_val, y_val),
            early_stopping_rounds=50
        )
        oof_preds[val_idx, 2] = cat_model.predict_proba(X_val)[:, 1]
        test_preds[:, 2] += cat_model.predict_proba(X_test)[:, 1] / n_splits
        
        # Evaluate fold
        fold_blend = oof_preds[val_idx, 0] * 0.4 + oof_preds[val_idx, 1] * 0.3 + oof_preds[val_idx, 2] * 0.3
        fold_pred = (fold_blend > 0.5).astype(int)
        fold_f1 = f1_score(y_val, fold_pred)
        print(f"  Fold {fold + 1} F1: {fold_f1:.4f}")
    
    # ==================== ENSEMBLE AND THRESHOLD OPTIMIZATION ====================
    print("\n" + "="*80)
    print("ENSEMBLE OPTIMIZATION")
    print("="*80)
    
    # Find optimal weights for ensemble
    best_weights = None
    best_threshold = 0.5
    best_f1 = 0
    
    # Grid search for weights
    for w1 in np.arange(0.2, 0.6, 0.1):
        for w2 in np.arange(0.2, 0.6, 0.1):
            w3 = 1 - w1 - w2
            if w3 <= 0 or w3 >= 1:
                continue
            
            # Weighted ensemble
            oof_ensemble = w1 * oof_preds[:, 0] + w2 * oof_preds[:, 1] + w3 * oof_preds[:, 2]
            
            # Find best threshold for this ensemble using precision-recall curve
            precision, recall, thresholds = precision_recall_curve(y_train, oof_ensemble)
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
            best_idx = np.argmax(f1_scores[:-1])  # Last value is undefined
            threshold = thresholds[best_idx]
            f1 = f1_scores[best_idx]
            
            if f1 > best_f1:
                best_f1 = f1
                best_weights = (w1, w2, w3)
                best_threshold = threshold
    
    print(f"Best weights: LGB={best_weights[0]:.2f}, XGB={best_weights[1]:.2f}, CAT={best_weights[2]:.2f}")
    print(f"Best threshold: {best_threshold:.3f}")
    print(f"Best CV F1: {best_f1:.4f}")
    
    # ==================== FINAL PREDICTIONS ====================
    print("\n" + "="*80)
    print("GENERATING FINAL PREDICTIONS")
    print("="*80)
    
    # Apply best ensemble
    test_ensemble = (best_weights[0] * test_preds[:, 0] + 
                    best_weights[1] * test_preds[:, 1] + 
                    best_weights[2] * test_preds[:, 2])
    
    # Apply threshold
    predictions = (test_ensemble > best_threshold).astype(int)
    
    # ==================== POST-PROCESSING WITH CONFIDENCE ====================
    print("\n" + "="*80)
    print("APPLYING CONFIDENCE-BASED RULES")
    print("="*80)
    
    # High confidence positive predictions
    if all(col in test_df.columns for col in ['liab_prct', 'witness_present_ind', 'accident_type']):
        high_conf_positive = (
            (test_df['liab_prct'] <= 10) & 
            (test_df['witness_present_ind'] == 'Y') & 
            (test_df['accident_type'].astype(str).str.contains('multi_vehicle', na=False))
        )
        print(f"High confidence positive: {high_conf_positive.sum()} cases")
        predictions[high_conf_positive] = 1
    
    # High confidence negative predictions
    if 'liab_prct' in test_df.columns:
        high_conf_negative = (test_df['liab_prct'] >= 90)
        if 'accident_type' in test_df.columns and 'witness_present_ind' in test_df.columns:
            high_conf_negative = high_conf_negative | (
                (test_df['accident_type'] == 'single_car') & (test_df['witness_present_ind'] == 'N')
            )
        print(f"High confidence negative: {high_conf_negative.sum()} cases")
        predictions[high_conf_negative] = 0
    
    # Edge cases based on probability
    edge_cases = (test_ensemble > 0.45) & (test_ensemble < 0.55)
    print(f"Edge cases: {edge_cases.sum()} cases")
    
    # For edge cases, use simple rules
    if all(col in test_df.columns for col in ['liab_prct', 'evidence_score']):
        edge_positive = edge_cases & (test_df['liab_prct'] < 40) & (test_df['evidence_score'] > 0)
        predictions[edge_positive] = 1
    
    print(f"\nFinal positive predictions: {predictions.sum()} ({predictions.mean():.3%})")
    
    # ==================== SAVE SUBMISSION ====================
    print("\n" + "="*80)
    print("SAVING SUBMISSION")
    print("="*80)
    
    submission = pd.DataFrame({
        'claim_number': test_ids,
        'subrogation': predictions
    })
    
    # Ensure correct data types
    submission['claim_number'] = submission['claim_number'].astype(int)
    submission['subrogation'] = submission['subrogation'].astype(int)
    
    output_path = Path(__file__).parent / 'submission_advanced_v2.csv'
    submission.to_csv(output_path, index=False)
    
    print(f"\nSubmission saved to: {output_path}")
    print(f"Shape: {submission.shape}")
    print(f"Distribution: {predictions.mean():.3%} positive rate")
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nKey Features of This Approach:")
    print("- Pattern discovery through liability segment analysis")
    print("- Multiple liability transformations (squared, cubed, log, sqrt)")
    print("- Precision-recall curve for optimal threshold")
    print("- Grid search for ensemble weights")
    print("- Confidence-based post-processing rules")
    print("- Edge case handling")
    print("="*80)
    
    return submission


if __name__ == '__main__':
    submission = main()
