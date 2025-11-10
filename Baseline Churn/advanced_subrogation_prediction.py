'''
Robust Subrogation Prediction Solution
Focuses on core subrogation logic with liability percentage as the primary driver.
Uses simpler feature engineering, stronger regularization, and business rules.
'''

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)


def create_core_features(df):
    """Create only the most important features for subrogation"""
    df = df.copy()
    
    # 1. LIABILITY - THE MOST CRITICAL FEATURE
    # Subrogation occurs when the other party is at fault
    if 'liab_prct' in df.columns:
        df['other_party_liability'] = 100 - df['liab_prct']  # Other party's fault percentage
        df['no_fault'] = (df['liab_prct'] == 0).astype(int)  # We have no fault
        df['low_fault'] = (df['liab_prct'] <= 25).astype(int)  # We have minimal fault
        df['majority_other_fault'] = (df['liab_prct'] < 50).astype(int)  # Other party majority at fault
        
        # Liability buckets - more granular
        df['liability_0'] = (df['liab_prct'] == 0).astype(int)
        df['liability_1_25'] = ((df['liab_prct'] > 0) & (df['liab_prct'] <= 25)).astype(int)
        df['liability_26_50'] = ((df['liab_prct'] > 25) & (df['liab_prct'] <= 50)).astype(int)
        df['liability_51_75'] = ((df['liab_prct'] > 50) & (df['liab_prct'] <= 75)).astype(int)
        df['liability_76_99'] = ((df['liab_prct'] > 75) & (df['liab_prct'] < 100)).astype(int)
        df['liability_100'] = (df['liab_prct'] == 100).astype(int)
    
    # 2. EVIDENCE - Critical for proving the other party's fault
    if 'witness_present_ind' in df.columns:
        df['has_witness'] = (df['witness_present_ind'] == 'Y').astype(int)
    else:
        df['has_witness'] = 0
    
    if 'policy_report_filed_ind' in df.columns:
        df['has_police_report'] = df['policy_report_filed_ind'].astype(int)
    else:
        df['has_police_report'] = 0
    
    df['strong_evidence'] = ((df['has_witness'] == 1) & (df['has_police_report'] == 1)).astype(int)
    df['any_evidence'] = ((df['has_witness'] == 1) | (df['has_police_report'] == 1)).astype(int)
    df['evidence_count'] = df['has_witness'] + df['has_police_report']
    
    # 3. ACCIDENT TYPE - Multi-vehicle with clear fault is best for subrogation
    if 'accident_type' in df.columns:
        df['is_multi_vehicle_clear'] = (df['accident_type'] == 'multi_vehicle_clear').astype(int)
        df['is_multi_vehicle'] = df['accident_type'].astype(str).str.contains('multi_vehicle', na=False).astype(int)
        df['is_single_vehicle'] = (df['accident_type'] == 'single_car').astype(int)
    else:
        df['is_multi_vehicle_clear'] = 0
        df['is_multi_vehicle'] = 0
        df['is_single_vehicle'] = 0
    
    # 4. KEY INTERACTIONS - Most important for subrogation
    if 'no_fault' in df.columns and 'is_multi_vehicle_clear' in df.columns:
        df['perfect_subrogation'] = df['no_fault'] * df['is_multi_vehicle_clear']
    
    if all(col in df.columns for col in ['low_fault', 'is_multi_vehicle', 'any_evidence']):
        df['strong_subrogation'] = df['low_fault'] * df['is_multi_vehicle'] * df['any_evidence']
    
    if 'majority_other_fault' in df.columns and 'is_multi_vehicle' in df.columns:
        df['good_subrogation'] = df['majority_other_fault'] * df['is_multi_vehicle']
    
    # Liability and evidence interaction
    if 'no_fault' in df.columns and 'any_evidence' in df.columns:
        df['no_fault_with_evidence'] = df['no_fault'] * df['any_evidence']
    
    if 'low_fault' in df.columns and 'has_witness' in df.columns:
        df['low_fault_with_witness'] = df['low_fault'] * df['has_witness']
    
    if 'low_fault' in df.columns and 'has_police_report' in df.columns:
        df['low_fault_with_police'] = df['low_fault'] * df['has_police_report']
    
    # 5. ACCIDENT LOCATION - Some locations have clearer fault determination
    if 'accident_site' in df.columns:
        df['is_parking'] = (df['accident_site'] == 'Parking Area').astype(int)
        df['is_highway'] = df['accident_site'].astype(str).str.contains('Highway', na=False).astype(int)
        df['is_intersection'] = df['accident_site'].astype(str).str.contains('Intersection', na=False).astype(int)
        
        # Location and liability interaction
        if 'no_fault' in df.columns:
            df['parking_no_fault'] = df['is_parking'] * df['no_fault']
        if 'low_fault' in df.columns:
            df['highway_low_fault'] = df['is_highway'] * df['low_fault']
    else:
        df['is_parking'] = 0
        df['is_highway'] = 0
        df['is_intersection'] = 0
    
    # 6. CLAIM VALUE - Higher values justify subrogation effort
    if 'claim_est_payout' in df.columns:
        df['claim_amount_log'] = np.log1p(df['claim_est_payout'])
        median_payout = df['claim_est_payout'].median()
        q75_payout = df['claim_est_payout'].quantile(0.75)
        df['claim_above_median'] = (df['claim_est_payout'] > median_payout).astype(int)
        df['claim_above_q75'] = (df['claim_est_payout'] > q75_payout).astype(int)
        
        # Value and fault interaction
        if 'no_fault' in df.columns:
            df['high_value_no_fault'] = df['claim_above_q75'] * df['no_fault']
        if 'low_fault' in df.columns:
            df['high_value_low_fault'] = df['claim_above_q75'] * df['low_fault']
    else:
        df['claim_amount_log'] = 0
        df['claim_above_median'] = 0
        df['claim_above_q75'] = 0
    
    # 7. DRIVER CHARACTERISTICS
    if 'year_of_born' in df.columns:
        current_year = 2024
        df['driver_age'] = current_year - df['year_of_born']
        if 'age_of_DL' in df.columns:
            df['years_driving'] = df['driver_age'] - df['age_of_DL']
            df['experienced_driver'] = (df['years_driving'] > 10).astype(int)
        else:
            df['years_driving'] = 0
            df['experienced_driver'] = 0
    else:
        df['driver_age'] = 0
        df['years_driving'] = 0
        df['experienced_driver'] = 0
    
    # 8. CLAIM HISTORY AND SAFETY
    if 'past_num_of_claims' in df.columns:
        df['no_prior_claims'] = (df['past_num_of_claims'] == 0).astype(int)
        df['multiple_prior_claims'] = (df['past_num_of_claims'] >= 2).astype(int)
    else:
        df['no_prior_claims'] = 0
        df['multiple_prior_claims'] = 0
    
    if 'safety_rating' in df.columns:
        df['good_safety_rating'] = (df['safety_rating'] >= 75).astype(int)
    else:
        df['good_safety_rating'] = 0
    
    # 9. VEHICLE CHARACTERISTICS
    if 'age_of_vehicle' in df.columns:
        df['vehicle_age'] = df['age_of_vehicle']
        df['newer_vehicle'] = (df['vehicle_age'] <= 5).astype(int)
    else:
        df['vehicle_age'] = 0
        df['newer_vehicle'] = 0
    
    # 10. COMPOSITE SCORES
    # Subrogation likelihood score based on domain knowledge
    if all(col in df.columns for col in ['other_party_liability', 'evidence_count', 'is_multi_vehicle', 'is_multi_vehicle_clear']):
        df['subrogation_score'] = (
            df['other_party_liability'] * 0.5 +  # Weight liability highest
            df['evidence_count'] * 15 +  # Evidence is crucial
            df['is_multi_vehicle'] * 20 +  # Multi-vehicle important
            df['is_multi_vehicle_clear'] * 30  # Clear fault multi-vehicle is best
        )
        
        # Risk-adjusted score
        if 'claim_amount_log' in df.columns:
            df['risk_adjusted_score'] = df['subrogation_score'] * (1 + df['claim_amount_log'] / 10)
        else:
            df['risk_adjusted_score'] = df['subrogation_score']
    else:
        df['subrogation_score'] = 0
        df['risk_adjusted_score'] = 0
    
    # 11. TEMPORAL FEATURES
    if 'claim_date' in df.columns:
        try:
            df['claim_date'] = pd.to_datetime(df['claim_date'], errors='coerce')
            df['claim_month'] = df['claim_date'].dt.month.fillna(0)
        except:
            df['claim_month'] = 0
    else:
        df['claim_month'] = 0
    
    if 'claim_day_of_week' in df.columns:
        weekday_map = {
            'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4,
            'Friday': 5, 'Saturday': 6, 'Sunday': 7
        }
        df['claim_weekday'] = df['claim_day_of_week'].map(weekday_map).fillna(0)
        df['is_weekend'] = df['claim_weekday'].isin([6, 7]).astype(int)
    else:
        df['claim_weekday'] = 0
        df['is_weekend'] = 0
    
    # 12. SIMPLE RATIOS
    if 'annual_income' in df.columns and 'claim_est_payout' in df.columns:
        df['income_to_claim_ratio'] = df['annual_income'] / (df['claim_est_payout'] + 1)
    else:
        df['income_to_claim_ratio'] = 0
    
    if 'vehicle_price' in df.columns and 'claim_est_payout' in df.columns:
        df['vehicle_value_to_claim_ratio'] = df['vehicle_price'] / (df['claim_est_payout'] + 1)
    else:
        df['vehicle_value_to_claim_ratio'] = 0
    
    return df


def main():
    """Main execution function"""
    print('='*80)
    print('ROBUST SUBROGATION PREDICTION PIPELINE')
    print('='*80)
    
    # Set up paths
    base_path = Path(__file__).parent.parent
    train_path = base_path / 'Data' / 'Training_TriGuard.csv'
    test_path = base_path / 'Data' / 'Testing_TriGuard.csv'
    
    # Load data
    print('\nLoading data...')
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Store IDs
    test_ids = test_df['claim_number'].copy()
    
    print(f"\nTrain shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print(f"Target distribution:\n{train_df['subrogation'].value_counts(normalize=True)}")
    
    # ==================== FOCUSED FEATURE ENGINEERING ====================
    print('\n' + '='*80)
    print('FOCUSED FEATURE ENGINEERING')
    print('='*80)
    
    train_df = create_core_features(train_df)
    test_df = create_core_features(test_df)
    
    print(f"Features after engineering - Train: {train_df.shape[1]}, Test: {test_df.shape[1]}")
    
    # ==================== SIMPLE BUT EFFECTIVE ENCODING ====================
    print('\n' + '='*80)
    print('ENCODING CATEGORICAL VARIABLES')
    print('='*80)
    
    # Handle categorical variables simply
    cat_columns = ['gender', 'living_status', 'channel', 'vehicle_category', 'witness_present_ind']
    
    # Label encoding
    for col in cat_columns:
        if col in train_df.columns:
            le = LabelEncoder()
            # Combine train and test to handle all categories
            combined = pd.concat([train_df[col].fillna('Unknown'), test_df[col].fillna('Unknown')])
            le.fit(combined)
            train_df[col + '_encoded'] = le.transform(train_df[col].fillna('Unknown'))
            test_df[col + '_encoded'] = le.transform(test_df[col].fillna('Unknown'))
            print(f"  Encoded: {col}")
    
    # For accident_site - create dummy variables for important categories
    if 'accident_site' in train_df.columns:
        train_df['site_parking'] = (train_df['accident_site'] == 'Parking Area').astype(int)
        train_df['site_highway'] = train_df['accident_site'].astype(str).str.contains('Highway', na=False).astype(int)
        train_df['site_local'] = (train_df['accident_site'] == 'Local').astype(int)
        
        test_df['site_parking'] = (test_df['accident_site'] == 'Parking Area').astype(int)
        test_df['site_highway'] = test_df['accident_site'].astype(str).str.contains('Highway', na=False).astype(int)
        test_df['site_local'] = (test_df['accident_site'] == 'Local').astype(int)
        print("  Created accident site dummies")
    
    # ==================== FEATURE SELECTION ====================
    print('\n' + '='*80)
    print('FEATURE SELECTION')
    print('='*80)
    
    # Select features - focus on numeric and encoded features
    exclude_cols = ['subrogation', 'claim_number', 'claim_date', 'year_of_born', 
                    'zip_code', 'vehicle_color', 'accident_site', 'claim_day_of_week',
                    'gender', 'living_status', 'channel', 'vehicle_category', 
                    'accident_type', 'witness_present_ind', 'in_network_bodyshop']
    
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]
    feature_cols = [col for col in feature_cols if train_df[col].dtype in ['int64', 'float64', 'int32', 'float32']]
    
    print(f"Using {len(feature_cols)} features")
    
    X_train = train_df[feature_cols].fillna(-1)
    y_train = train_df['subrogation']
    X_test = test_df[feature_cols].fillna(-1)
    
    # ==================== ROBUST MODEL TRAINING ====================
    print('\n' + '='*80)
    print('ROBUST MODEL TRAINING')
    print('='*80)
    
    # Calculate class weight
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"Positive class weight: {pos_weight:.2f}")
    
    # Model 1: LightGBM with careful tuning
    lgb_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 20,  # Reduced to prevent overfitting
        'max_depth': 5,  # Limit depth
        'learning_rate': 0.05,
        'n_estimators': 300,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 1.0,  # Increased regularization
        'reg_lambda': 1.0,
        'min_child_samples': 30,  # Increased to prevent overfitting
        'scale_pos_weight': pos_weight,
        'random_state': 42,
        'verbosity': -1
    }
    
    # Model 2: XGBoost
    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 4,  # Shallow trees
        'learning_rate': 0.05,
        'n_estimators': 300,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'min_child_weight': 5,
        'gamma': 0.5,  # Minimum loss reduction
        'reg_alpha': 1.0,
        'reg_lambda': 1.0,
        'scale_pos_weight': pos_weight,
        'random_state': 42,
        'verbosity': 0
    }
    
    # Model 3: CatBoost
    cat_params = {
        'iterations': 300,
        'depth': 5,
        'learning_rate': 0.05,
        'l2_leaf_reg': 5,
        'border_count': 32,
        'auto_class_weights': 'Balanced',
        'random_seed': 42,
        'verbose': False
    }
    
    # ==================== CROSS-VALIDATION WITH ENSEMBLE ====================
    print('\n' + '='*80)
    print('CROSS-VALIDATION WITH ENSEMBLE')
    print('='*80)
    
    n_splits = 5
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Store predictions
    oof_lgb = np.zeros(len(X_train))
    oof_xgb = np.zeros(len(X_train))
    oof_cat = np.zeros(len(X_train))
    
    test_lgb = np.zeros(len(X_test))
    test_xgb = np.zeros(len(X_test))
    test_cat = np.zeros(len(X_test))
    
    # Store models for feature importance
    models_lgb = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train, y_train)):
        print(f"\nFold {fold + 1}/{n_splits}")
        
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # LightGBM
        lgb_model = lgb.LGBMClassifier(**lgb_params)
        lgb_model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)]
        )
        models_lgb.append(lgb_model)
        
        oof_lgb[val_idx] = lgb_model.predict_proba(X_val)[:, 1]
        test_lgb += lgb_model.predict_proba(X_test)[:, 1] / n_splits
        
        # XGBoost
        xgb_model = xgb.XGBClassifier(**xgb_params)
        xgb_model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=30,
            verbose=False
        )
        
        oof_xgb[val_idx] = xgb_model.predict_proba(X_val)[:, 1]
        test_xgb += xgb_model.predict_proba(X_test)[:, 1] / n_splits
        
        # CatBoost
        cat_model = CatBoostClassifier(**cat_params)
        cat_model.fit(
            X_tr, y_tr,
            eval_set=(X_val, y_val),
            early_stopping_rounds=30,
            use_best_model=True
        )
        
        oof_cat[val_idx] = cat_model.predict_proba(X_val)[:, 1]
        test_cat += cat_model.predict_proba(X_test)[:, 1] / n_splits
        
        # Evaluate fold
        fold_blend = oof_lgb[val_idx] * 0.4 + oof_xgb[val_idx] * 0.3 + oof_cat[val_idx] * 0.3
        fold_pred = (fold_blend > 0.5).astype(int)
        fold_f1 = f1_score(y_val, fold_pred)
        print(f"  Fold {fold + 1} F1: {fold_f1:.4f}")
    
    # ==================== OPTIMIZE THRESHOLD ====================
    print('\n' + '='*80)
    print('THRESHOLD OPTIMIZATION')
    print('='*80)
    
    # Blend predictions
    oof_blend = oof_lgb * 0.4 + oof_xgb * 0.3 + oof_cat * 0.3
    
    # Find best threshold
    best_threshold = 0.5
    best_f1 = 0
    
    for threshold in np.arange(0.3, 0.7, 0.01):
        pred = (oof_blend > threshold).astype(int)
        score = f1_score(y_train, pred)
        if score > best_f1:
            best_f1 = score
            best_threshold = threshold
    
    print(f"\nBest threshold: {best_threshold:.3f}")
    print(f"Best CV F1: {best_f1:.4f}")
    
    # ==================== RULE-BASED ADJUSTMENTS ====================
    print('\n' + '='*80)
    print('BUSINESS RULE POST-PROCESSING')
    print('='*80)
    
    test_blend = test_lgb * 0.4 + test_xgb * 0.3 + test_cat * 0.3
    
    # Apply threshold
    predictions = (test_blend > best_threshold).astype(int)
    
    # Apply business rules based on domain knowledge
    # Rule 1: If our liability is 0% and it's multi-vehicle, very likely subrogation
    if 'liab_prct' in test_df.columns and 'accident_type' in test_df.columns:
        rule1_mask = (test_df['liab_prct'] == 0) & (test_df['accident_type'].astype(str).str.contains('multi_vehicle', na=False))
        predictions[rule1_mask] = 1
        print(f"  Rule 1 applied: {rule1_mask.sum()} cases")
    
    # Rule 2: If our liability is 100%, no subrogation possible
    if 'liab_prct' in test_df.columns:
        rule2_mask = test_df['liab_prct'] == 100
        predictions[rule2_mask] = 0
        print(f"  Rule 2 applied: {rule2_mask.sum()} cases")
    
    # Rule 3: Single vehicle accidents rarely have subrogation (unless hit by external object)
    if 'accident_type' in test_df.columns and 'liab_prct' in test_df.columns:
        rule3_mask = (test_df['accident_type'] == 'single_car') & (test_df['liab_prct'] > 50)
        predictions[rule3_mask] = 0
        print(f"  Rule 3 applied: {rule3_mask.sum()} cases")
    
    # Rule 4: Strong evidence with low liability
    if all(col in test_df.columns for col in ['liab_prct', 'witness_present_ind', 'policy_report_filed_ind']):
        rule4_mask = (test_df['liab_prct'] <= 25) & (test_df['witness_present_ind'] == 'Y') & (test_df['policy_report_filed_ind'] == 1)
        predictions[rule4_mask] = 1
        print(f"  Rule 4 applied: {rule4_mask.sum()} cases")
    
    print(f"\nFinal predictions: {predictions.sum()} positive out of {len(predictions)}")
    
    # ==================== FEATURE IMPORTANCE ====================
    print('\n' + '='*80)
    print('FEATURE IMPORTANCE')
    print('='*80)
    
    # Average feature importance across folds
    feature_importance = pd.DataFrame({'feature': feature_cols})
    feature_importance['importance'] = 0
    
    for model in models_lgb:
        feature_importance['importance'] += model.feature_importances_ / len(models_lgb)
    
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    print("\nTop 15 Features:")
    print(feature_importance.head(15).to_string())
    
    # Save feature importance
    importance_path = Path(__file__).parent / 'feature_importance_robust.csv'
    feature_importance.to_csv(importance_path, index=False)
    print(f"\nFeature importance saved to: {importance_path}")
    
    # ==================== SAVE SUBMISSION ====================
    print('\n' + '='*80)
    print('SAVING SUBMISSION')
    print('='*80)
    
    submission = pd.DataFrame({
        'claim_number': test_ids,
        'subrogation': predictions
    })
    
    # Ensure correct data types
    submission['claim_number'] = submission['claim_number'].astype(int)
    submission['subrogation'] = submission['subrogation'].astype(int)
    
    output_path = Path(__file__).parent / 'submission_robust.csv'
    submission.to_csv(output_path, index=False)
    
    print(f"\nSubmission saved to: {output_path}")
    print(f"Distribution: {predictions.mean():.3f} positive rate")
    print(f"Shape: {submission.shape}")
    
    # ==================== ANALYSIS OF KEY PATTERNS ====================
    print('\n' + '='*80)
    print('KEY PATTERNS IN TRAINING DATA')
    print('='*80)
    
    # Analyze liability distribution for subrogation cases
    if 'liab_prct' in train_df.columns:
        print("\nLiability % for Subrogation=1 cases:")
        print(train_df[train_df['subrogation'] == 1]['liab_prct'].describe())
        print("\nLiability % for Subrogation=0 cases:")
        print(train_df[train_df['subrogation'] == 0]['liab_prct'].describe())
    
    # Analyze accident type distribution
    if 'accident_type' in train_df.columns:
        print("\nAccident type distribution for Subrogation=1:")
        print(train_df[train_df['subrogation'] == 1]['accident_type'].value_counts(normalize=True))
    
    # Evidence patterns
    if all(col in train_df.columns for col in ['witness_present_ind', 'policy_report_filed_ind']):
        print("\nEvidence patterns for Subrogation=1:")
        evidence_sub1 = train_df[train_df['subrogation'] == 1][['witness_present_ind', 'policy_report_filed_ind']].value_counts(normalize=True)
        print(evidence_sub1)
    
    print('\n' + '='*80)
    print('PIPELINE COMPLETED SUCCESSFULLY!')
    print('='*80)
    print("\nKey Features of This Approach:")
    print("- Focuses on liability percentage as primary driver")
    print("- Simpler feature engineering (only subrogation-relevant features)")
    print("- Stronger regularization to prevent overfitting")
    print("- Balanced ensemble of 3 models (LightGBM, XGBoost, CatBoost)")
    print("- Business rule post-processing for domain knowledge")
    print("- Proper stratified cross-validation")
    print('='*80)
    
    return submission, feature_importance


if __name__ == '__main__':
    submission, feature_importance = main()
