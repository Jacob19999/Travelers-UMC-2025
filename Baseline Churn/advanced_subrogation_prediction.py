'''
Advanced Subrogation Prediction Solution
Targets 0.7-0.8 F1 score using ensemble of LightGBM, XGBoost, and CatBoost
with comprehensive feature engineering and threshold optimization.
'''

# ============================================================================
# Library Installation and Import Check
# ============================================================================
import subprocess
import sys

def install_package(package, import_name=None):
    """Install a package if it's not already installed"""
    if import_name is None:
        import_name = package
    
    try:
        __import__(import_name)
        print(f"✓ {package} is already installed")
        return True
    except ImportError:
        print(f"⚠ {package} not found. Installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--quiet"])
            print(f"✓ {package} installed successfully!")
            return True
        except Exception as e:
            print(f"✗ Failed to install {package}: {e}")
            return False

# List of required packages and their import names
required_packages = {
    'pandas': 'pandas',
    'numpy': 'numpy',
    'scikit-learn': 'sklearn',
    'xgboost': 'xgboost',
    'lightgbm': 'lightgbm',
    'catboost': 'catboost'
}

print("="*80)
print("CHECKING AND INSTALLING REQUIRED LIBRARIES")
print("="*80)

# Check and install packages
all_installed = True
for pip_name, import_name in required_packages.items():
    if not install_package(pip_name, import_name):
        all_installed = False

if not all_installed:
    print("\n⚠ Some packages failed to install. Please install them manually.")
    sys.exit(1)

print("\n" + "="*80)
print("IMPORTING LIBRARIES")
print("="*80)

# Import all required libraries
try:
    import pandas as pd
    print("✓ pandas imported successfully")
    
    import numpy as np
    print("✓ numpy imported successfully")
    
    from pathlib import Path
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import f1_score, classification_report, confusion_matrix
    print("✓ scikit-learn imported successfully")
    
    import xgboost as xgb
    print("✓ xgboost imported successfully")
    
    import lightgbm as lgb
    print("✓ lightgbm imported successfully")
    
    from catboost import CatBoostClassifier
    print("✓ catboost imported successfully")
    
    import warnings
    warnings.filterwarnings('ignore')
    print("✓ warnings configured")
    
    print("\n" + "="*80)
    print("ALL LIBRARIES SUCCESSFULLY IMPORTED!")
    print("="*80)
    print(f"\nLibrary Versions:")
    print(f"  pandas: {pd.__version__}")
    print(f"  numpy: {np.__version__}")
    print(f"  xgboost: {xgb.__version__}")
    print(f"  lightgbm: {lgb.__version__}")
    print(f"  scikit-learn: {__import__('sklearn').__version__}")
    try:
        import catboost
        print(f"  catboost: {catboost.__version__}")
    except:
        pass
    
except ImportError as e:
    print(f"\n✗ Error importing libraries: {e}")
    print("Please ensure all packages are installed correctly.")
    sys.exit(1)


def create_advanced_features(df):
    """Create powerful features for subrogation prediction"""
    df = df.copy()
    
    # 1. Liability-based features (KEY for subrogation)
    if 'liab_prct' in df.columns:
        df['low_liability'] = (df['liab_prct'] <= 30).astype(int)
        df['no_liability'] = (df['liab_prct'] == 0).astype(int)
        df['partial_liability'] = ((df['liab_prct'] > 0) & (df['liab_prct'] < 100)).astype(int)
        df['liability_bins'] = pd.cut(df['liab_prct'], bins=[-1, 0, 25, 50, 75, 100], 
                                       labels=['none', 'low', 'medium', 'high', 'full'])
    
    # 2. Evidence strength features
    if 'witness_present_ind' in df.columns and 'policy_report_filed_ind' in df.columns:
        witness_bool = (df['witness_present_ind'] == 'Y') | (df['witness_present_ind'] == 1)
        policy_bool = (df['policy_report_filed_ind'] == 1) | (df['policy_report_filed_ind'] == 'Y')
        df['strong_evidence'] = (witness_bool | policy_bool).astype(int)
        df['evidence_score'] = witness_bool.astype(int) + policy_bool.astype(int)
    
    # 3. Accident type features (multi-vehicle clear = high subrogation potential)
    if 'accident_type' in df.columns:
        df['clear_fault_accident'] = (df['accident_type'] == 'multi_vehicle_clear').astype(int)
        df['multi_vehicle'] = df['accident_type'].astype(str).str.contains('multi_vehicle', na=False).astype(int)
    
    # 4. Claim value features
    if 'claim_est_payout' in df.columns:
        df['high_value_claim'] = (df['claim_est_payout'] > df['claim_est_payout'].quantile(0.75)).astype(int)
        df['log_claim_payout'] = np.log1p(df['claim_est_payout'])
        if 'vehicle_price' in df.columns:
            df['payout_to_vehicle_price_ratio'] = df['claim_est_payout'] / (df['vehicle_price'] + 1)
    
    # 5. Driver features
    if 'year_of_born' in df.columns:
        df['age'] = 2024 - df['year_of_born']
        df['young_driver'] = (df['age'] < 25).astype(int)
        df['senior_driver'] = (df['age'] > 65).astype(int)
        if 'age_of_DL' in df.columns:
            df['driving_experience'] = df['age'] - df['age_of_DL']
            df['inexperienced_driver'] = (df['driving_experience'] < 5).astype(int)
    
    # 6. Vehicle features
    if 'claim_date' in df.columns and 'vehicle_made_year' in df.columns:
        try:
            df['claim_date'] = pd.to_datetime(df['claim_date'], errors='coerce')
            claim_year = df['claim_date'].dt.year
            df['age_of_vehicle'] = claim_year - df['vehicle_made_year']
            df['age_of_vehicle'] = df['age_of_vehicle'].clip(lower=0, upper=50)
        except:
            pass
    
    if 'age_of_vehicle' in df.columns:
        df['new_vehicle'] = (df['age_of_vehicle'] <= 3).astype(int)
        df['old_vehicle'] = (df['age_of_vehicle'] > 10).astype(int)
        if 'vehicle_mileage' in df.columns:
            df['high_mileage'] = (df['vehicle_mileage'] > df['vehicle_mileage'].quantile(0.75)).astype(int)
            df['mileage_per_year'] = df['vehicle_mileage'] / (df['age_of_vehicle'] + 1)
    
    # 7. Location-based risk features
    if 'accident_site' in df.columns:
        df['highway_accident'] = df['accident_site'].astype(str).str.contains('Highway', na=False).astype(int)
        df['parking_accident'] = (df['accident_site'] == 'Parking Area').astype(int)
        df['intersection_accident'] = df['accident_site'].astype(str).str.contains('Intersection', na=False).astype(int)
    
    # 8. Claim timing features
    if 'claim_date' in df.columns:
        try:
            df['claim_date'] = pd.to_datetime(df['claim_date'], errors='coerce')
            df['claim_month'] = df['claim_date'].dt.month
            df['claim_quarter'] = df['claim_date'].dt.quarter
        except:
            pass
    
    if 'claim_day_of_week' in df.columns:
        df['is_weekend'] = df['claim_day_of_week'].isin(['Saturday', 'Sunday']).astype(int)
        df['is_monday'] = (df['claim_day_of_week'] == 'Monday').astype(int)
        df['is_friday'] = (df['claim_day_of_week'] == 'Friday').astype(int)
    
    # 9. Risk profile features
    if 'past_num_of_claims' in df.columns and 'safety_rating' in df.columns:
        high_risk_conditions = (df['past_num_of_claims'] > 2) | (df['safety_rating'] < 50)
        if 'young_driver' in df.columns:
            high_risk_conditions = high_risk_conditions | (df['young_driver'] == 1)
        df['high_risk_profile'] = high_risk_conditions.astype(int)
        
        low_risk_conditions = (df['past_num_of_claims'] == 0) & (df['safety_rating'] > 75)
        if 'high_education_ind' in df.columns:
            low_risk_conditions = low_risk_conditions & (df['high_education_ind'] == 1)
        df['low_risk_profile'] = low_risk_conditions.astype(int)
    
    # 10. Interaction features
    if 'low_liability' in df.columns and 'strong_evidence' in df.columns:
        df['liability_x_evidence'] = df['low_liability'] * df['strong_evidence']
    
    if 'clear_fault_accident' in df.columns and 'high_value_claim' in df.columns:
        df['clear_fault_x_high_value'] = df['clear_fault_accident'] * df['high_value_claim']
    
    if 'multi_vehicle' in df.columns and 'witness_present_ind' in df.columns:
        witness_bool = (df['witness_present_ind'] == 'Y') | (df['witness_present_ind'] == 1)
        df['multi_vehicle_x_witness'] = df['multi_vehicle'] * witness_bool.astype(int)
    
    if 'low_liability' in df.columns and 'policy_report_filed_ind' in df.columns:
        policy_bool = (df['policy_report_filed_ind'] == 1) | (df['policy_report_filed_ind'] == 'Y')
        df['low_liability_x_police_report'] = df['low_liability'] * policy_bool.astype(int)
    
    # 11. Financial features
    if 'annual_income' in df.columns and 'claim_est_payout' in df.columns:
        df['income_to_payout_ratio'] = df['annual_income'] / (df['claim_est_payout'] + 1)
        df['can_afford_deductible'] = (df['annual_income'] > df['claim_est_payout'] * 10).astype(int)
    
    # 12. Network features
    if 'in_network_bodyshop' in df.columns:
        df['using_network_shop'] = ((df['in_network_bodyshop'] == 'yes') | 
                                    (df['in_network_bodyshop'] == 1)).astype(int)
    
    return df


def target_encode(train_df, test_df, column, target='subrogation'):
    """Target encoding with smoothing to prevent overfitting"""
    global_mean = train_df[target].mean()
    
    # Calculate statistics for each category
    stats = train_df.groupby(column)[target].agg(['mean', 'count'])
    
    # Apply smoothing (more samples = more trust in the category mean)
    smoothing = 10
    stats['smooth_mean'] = (stats['mean'] * stats['count'] + global_mean * smoothing) / (stats['count'] + smoothing)
    
    # Create encoding dictionary
    encoding_dict = stats['smooth_mean'].to_dict()
    
    # Apply encoding
    train_df[f'{column}_target_enc'] = train_df[column].map(encoding_dict).fillna(global_mean)
    test_df[f'{column}_target_enc'] = test_df[column].map(encoding_dict).fillna(global_mean)
    
    return train_df, test_df


def find_optimal_threshold(y_true, y_prob):
    """Find threshold that maximizes F1 score"""
    thresholds = np.linspace(0.3, 0.7, 41)
    f1_scores = []
    
    for threshold in thresholds:
        y_pred = (y_prob > threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        f1_scores.append(f1)
    
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_f1 = f1_scores[optimal_idx]
    
    return optimal_threshold, optimal_f1


def main():
    """Main execution function"""
    print('='*80)
    print('ADVANCED SUBROGATION PREDICTION PIPELINE')
    print('='*80)
    
    # Set up paths
    base_path = Path(__file__).parent.parent
    train_path = base_path / 'Data' / 'Training_TriGuard.csv'
    test_path = base_path / 'Data' / 'Testing_TriGuard.csv'
    
    # Load data
    print('\nLoading data...')
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Store claim numbers for submission
    test_claim_numbers = test_df['claim_number'].copy()
    
    # Initial exploration
    print(f"\nTrain shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print(f"Target distribution:\n{train_df['subrogation'].value_counts(normalize=True)}")
    
    # ==================== FEATURE ENGINEERING ====================
    print('\n' + '='*80)
    print('FEATURE ENGINEERING')
    print('='*80)
    
    train_df = create_advanced_features(train_df)
    test_df = create_advanced_features(test_df)
    
    print(f"Features after engineering - Train: {train_df.shape[1]}, Test: {test_df.shape[1]}")
    
    # ==================== ENCODING CATEGORICAL VARIABLES ====================
    print('\n' + '='*80)
    print('ENCODING CATEGORICAL VARIABLES')
    print('='*80)
    
    # Identify categorical columns
    categorical_cols = ['gender', 'living_status', 'accident_site', 'channel', 
                       'vehicle_category', 'vehicle_color', 'accident_type', 
                       'in_network_bodyshop', 'witness_present_ind', 'liability_bins',
                       'claim_day_of_week']
    
    # Filter to only existing columns
    categorical_cols = [col for col in categorical_cols if col in train_df.columns]
    
    # Apply target encoding to high-cardinality features
    high_cardinality_cols = ['zip_code', 'vehicle_color']
    for col in high_cardinality_cols:
        if col in train_df.columns:
            train_df, test_df = target_encode(train_df, test_df, col)
            print(f"  Applied target encoding to: {col}")
    
    # Label encode other categorical variables
    label_encoders = {}
    for col in categorical_cols:
        if col in train_df.columns:
            le = LabelEncoder()
            # Convert to string first to handle categorical columns, then fillna
            train_col_str = train_df[col].astype(str).replace('nan', 'missing')
            test_col_str = test_df[col].astype(str).replace('nan', 'missing')
            # Fit on combined data to handle unseen categories
            combined = pd.concat([train_col_str, test_col_str])
            le.fit(combined)
            train_df[f'{col}_encoded'] = le.transform(train_col_str)
            test_df[f'{col}_encoded'] = le.transform(test_col_str)
            label_encoders[col] = le
            print(f"  Applied label encoding to: {col}")
    
    # ==================== FEATURE SELECTION ====================
    print('\n' + '='*80)
    print('FEATURE SELECTION')
    print('='*80)
    
    # Select features for modeling
    exclude_cols = ['subrogation', 'claim_number', 'claim_date', 'zip_code', 'vehicle_color'] + categorical_cols
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]
    
    # Remove any remaining object type columns
    feature_cols = [col for col in feature_cols if train_df[col].dtype != 'object']
    
    X_train = train_df[feature_cols].fillna(-999)
    y_train = train_df['subrogation']
    X_test = test_df[feature_cols].fillna(-999)
    
    print(f"Number of features: {len(feature_cols)}")
    print(f"Feature columns sample: {feature_cols[:10]}")
    
    # ==================== ADVANCED MODELING ====================
    print('\n' + '='*80)
    print('MODEL TRAINING')
    print('='*80)
    
    # 1. LightGBM with optimized parameters for subrogation
    lgb_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.03,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'n_estimators': 500,
        'random_state': 42,
        'class_weight': 'balanced',  # Handle imbalance
        'importance_type': 'gain',
        'verbosity': -1
    }
    
    # 2. XGBoost with scale_pos_weight for imbalanced data
    scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 6,
        'learning_rate': 0.03,
        'n_estimators': 500,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'scale_pos_weight': scale_pos_weight,
        'min_child_weight': 3,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1,
        'random_state': 42,
        'verbosity': 0
    }
    
    # 3. CatBoost with auto class weights
    cat_params = {
        'iterations': 500,
        'learning_rate': 0.03,
        'depth': 6,
        'l2_leaf_reg': 3,
        'auto_class_weights': 'Balanced',
        'random_seed': 42,
        'verbose': False
    }
    
    # ==================== STACKING ENSEMBLE ====================
    print('\n' + '='*80)
    print('STACKING ENSEMBLE WITH CROSS-VALIDATION')
    print('='*80)
    
    # Initialize models
    lgb_model = lgb.LGBMClassifier(**lgb_params)
    xgb_model = xgb.XGBClassifier(**xgb_params)
    cat_model = CatBoostClassifier(**cat_params)
    
    # Create out-of-fold predictions for stacking
    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Arrays to store out-of-fold predictions
    oof_lgb = np.zeros((len(X_train), 2))
    oof_xgb = np.zeros((len(X_train), 2))
    oof_cat = np.zeros((len(X_train), 2))
    
    # Arrays to store test predictions
    test_lgb = np.zeros((len(X_test), 2))
    test_xgb = np.zeros((len(X_test), 2))
    test_cat = np.zeros((len(X_test), 2))
    
    # Track F1 scores
    f1_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"\nFold {fold + 1}/{n_folds}")
        
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # Train LightGBM
        lgb_model.fit(X_fold_train, y_fold_train, 
                      eval_set=[(X_fold_val, y_fold_val)], 
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        oof_lgb[val_idx] = lgb_model.predict_proba(X_fold_val)
        test_lgb += lgb_model.predict_proba(X_test) / n_folds
        
        # Train XGBoost
        xgb_model.fit(X_fold_train, y_fold_train, 
                      eval_set=[(X_fold_val, y_fold_val)], 
                      early_stopping_rounds=50, 
                      verbose=False)
        oof_xgb[val_idx] = xgb_model.predict_proba(X_fold_val)
        test_xgb += xgb_model.predict_proba(X_test) / n_folds
        
        # Train CatBoost
        cat_model.fit(X_fold_train, y_fold_train, 
                      eval_set=(X_fold_val, y_fold_val),
                      early_stopping_rounds=50)
        oof_cat[val_idx] = cat_model.predict_proba(X_fold_val)
        test_cat += cat_model.predict_proba(X_test) / n_folds
        
        # Calculate fold F1 score
        fold_pred = (oof_lgb[val_idx, 1] + oof_xgb[val_idx, 1] + oof_cat[val_idx, 1]) / 3
        fold_f1 = f1_score(y_fold_val, (fold_pred > 0.5).astype(int))
        f1_scores.append(fold_f1)
        print(f"  Fold {fold + 1} F1 Score: {fold_f1:.4f}")
    
    print(f"\nMean CV F1 Score: {np.mean(f1_scores):.4f} (+/- {np.std(f1_scores):.4f})")
    
    # ==================== OPTIMIZATION OF THRESHOLD ====================
    print('\n' + '='*80)
    print('THRESHOLD OPTIMIZATION')
    print('='*80)
    
    # Create ensemble predictions
    ensemble_oof = (oof_lgb[:, 1] * 0.4 + oof_xgb[:, 1] * 0.35 + oof_cat[:, 1] * 0.25)
    ensemble_test = (test_lgb[:, 1] * 0.4 + test_xgb[:, 1] * 0.35 + test_cat[:, 1] * 0.25)
    
    # Find optimal threshold
    optimal_threshold, optimal_f1 = find_optimal_threshold(y_train, ensemble_oof)
    print(f"\nOptimal Threshold: {optimal_threshold:.3f}")
    print(f"Optimal F1 Score: {optimal_f1:.4f}")
    
    # ==================== FINAL PREDICTIONS ====================
    print('\n' + '='*80)
    print('FINAL PREDICTIONS')
    print('='*80)
    
    # Apply optimal threshold
    final_predictions = (ensemble_test > optimal_threshold).astype(int)
    
    # Create submission
    submission = pd.DataFrame({
        'claim_number': test_claim_numbers,
        'subrogation': final_predictions
    })
    
    # Ensure correct data types
    submission['claim_number'] = submission['claim_number'].astype(int)
    submission['subrogation'] = submission['subrogation'].astype(int)
    
    # Save submission
    output_path = Path(__file__).parent / 'submission_advanced.csv'
    submission.to_csv(output_path, index=False)
    
    print(f"\nSubmission saved to: {output_path}")
    print(f"Shape: {submission.shape}")
    print(f"Prediction distribution:\n{submission['subrogation'].value_counts(normalize=True)}")
    
    # ==================== FEATURE IMPORTANCE ====================
    print('\n' + '='*80)
    print('FEATURE IMPORTANCE')
    print('='*80)
    
    # Get feature importance from LightGBM (using the last trained model)
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': lgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 15 Most Important Features:")
    print(feature_importance.head(15))
    
    # Save feature importance
    importance_path = Path(__file__).parent / 'feature_importance_advanced.csv'
    feature_importance.to_csv(importance_path, index=False)
    print(f"\nFeature importance saved to: {importance_path}")
    
    print('\n' + '='*80)
    print('PIPELINE COMPLETED SUCCESSFULLY!')
    print('='*80)
    
    return submission, feature_importance


if __name__ == '__main__':
    submission, feature_importance = main()

