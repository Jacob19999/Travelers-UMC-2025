import pandas as pd
import numpy as np
import os
import sys
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import shap
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, precision_recall_curve, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import PartialDependenceDisplay

# Suppress warnings
warnings.filterwarnings('ignore')

# --- Configuration ---
SEED = 42
N_FOLDS = 5
np.random.seed(SEED)

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'Data')
TRAIN_PATH = os.path.join(DATA_DIR, 'Training_TriGuard.csv')
TEST_PATH = os.path.join(DATA_DIR, 'Testing_TriGuard.csv')
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__)) # Save artifacts here

def load_data():
    """Loads data and performs basic splitting."""
    print("Loading data...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    
    train_df['is_train'] = 1
    test_df['is_train'] = 0
    
    # Store target and IDs
    target = train_df['subrogation']
    train_ids = train_df['claim_number']
    test_ids = test_df['claim_number']
    
    # Concatenate for consistent processing
    df = pd.concat([train_df.drop(['subrogation'], axis=1), test_df], axis=0).reset_index(drop=True)
    
    return df, target, train_ids, test_ids

def audit_missingness(df):
    """
    Step 0: Missingness & Data Quality Audit
    - Computes % missing per feature.
    - Creates row-level missingness features.
    - Identifies informative missingness.
    """
    print("\n--- Step 0: Missingness Audit ---")
    
    # Feature-level missingness
    missing_percent = df.isnull().mean()
    high_missing_cols = missing_percent[missing_percent > 0.6].index.tolist()
    print(f"Features with >60% missing (candidates to drop): {high_missing_cols}")
    
    # Create missing indicators for informative missingness (e.g. > 5% missing)
    informative_missing_cols = missing_percent[(missing_percent > 0.05) & (missing_percent <= 0.6)].index.tolist()
    print(f"Creating missing indicators for: {informative_missing_cols}")
    
    for col in informative_missing_cols:
        df[f'{col}_is_missing'] = df[col].isnull().astype(int)
        
    # Row-level missingness
    df['missing_count'] = df.isnull().sum(axis=1)
    df['missing_fraction'] = df.isnull().mean(axis=1)
    
    # Drop high missingness columns (optional, but good for "Clean Data" goal)
    # df.drop(columns=high_missing_cols, inplace=True) 
    # Keeping them for now as tree models handle them, but user mentioned dropping candidates.
    # Let's strictly follow: "If a feature has >50–60% missing... candidates to drop."
    # We will drop them to be "Regulator Friendly" (less noise).
    if high_missing_cols:
        print(f"Dropping high missingness columns: {high_missing_cols}")
        df.drop(columns=high_missing_cols, inplace=True)
        
    return df

def robust_imputation(df):
    """
    Step 1: Simple but Robust Imputation
    - Numeric: Median + is_missing flag (already done for informative ones, but do for all remaining NA).
    - Categorical: Replace with "MISSING".
    """
    print("\n--- Step 1: Robust Imputation ---")
    
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    
    # Numeric Imputation
    for col in numeric_cols:
        if df[col].isnull().any():
            # Create flag if not already created in Step 0
            if f'{col}_is_missing' not in df.columns:
                df[f'{col}_is_missing'] = df[col].isnull().astype(int)
            
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            
    # Categorical Imputation
    for col in cat_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna("MISSING")
            
    return df

def feature_selection(df, target, train_ids):
    """
    Step 2: Feature Selection & Reduction
    - Low variance filter.
    - Correlation-based pruning.
    - Importance-based selection using a quick LightGBM.
    """
    print("\n--- Step 2: Feature Selection ---")
    
    # 1. Low Variance Filter
    # Variance threshold: drop if >99% of values are the same
    cols_to_drop = []
    for col in df.columns:
        if df[col].nunique() <= 1:
            cols_to_drop.append(col)
        else:
            # Check frequency of most common value
            top_val_freq = df[col].value_counts(normalize=True).iloc[0]
            if top_val_freq > 0.99:
                cols_to_drop.append(col)
                
    print(f"Dropping low variance features: {cols_to_drop}")
    df.drop(columns=cols_to_drop, inplace=True)
    
    # Prepare for numeric analysis (Correlation)
    numeric_df = df.select_dtypes(include=['number'])
    
    # 2. Correlation Pruning (>0.95)
    corr_matrix = numeric_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop_corr = [column for column in upper.columns if any(upper[column] > 0.95)]
    
    print(f"Dropping highly correlated features (>0.95): {to_drop_corr}")
    df.drop(columns=to_drop_corr, inplace=True)
    
    # 3. Importance Pre-selection
    # Need to encode categoricals first for LightGBM
    # (LightGBM handles cats, but we need to set type to 'category')
    
    # Separate Train for importance check
    df_train = df[df['is_train'] == 1].drop('is_train', axis=1)
    # Ensure alignment
    df_train = df_train.iloc[:len(target)] 
    
    cat_cols = df_train.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df_train[col] = df_train[col].astype('category')
        df[col] = df[col].astype('category') # Apply to whole df for consistency
        
    print("Running quick LightGBM for feature importance...")
    lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
        'n_estimators': 100,
        'learning_rate': 0.1,
        'random_state': SEED,
        'verbose': -1,
        'n_jobs': -1
    }
    
    # Drop IDs if present
    features = [c for c in df_train.columns if c not in ['claim_number', 'is_train']]
    
    clf = lgb.LGBMClassifier(**lgb_params)
    clf.fit(df_train[features], target)
    
    importances = pd.DataFrame({
        'feature': features,
        'gain': clf.booster_.feature_importance(importance_type='gain')
    }).sort_values('gain', ascending=False)
    
    # Threshold: Drop features with 0 or very low gain relative to top
    # Let's keep top 95% of cumulative gain or simply drop 0 importance
    zero_importance = importances[importances['gain'] == 0]['feature'].tolist()
    print(f"Dropping {len(zero_importance)} features with 0 importance.")
    df.drop(columns=zero_importance, inplace=True)
    
    return df

def train_explainable_gbm(X, y, X_test):
    """
    Step 3: Explainable GBM
    - Train LightGBM with CV.
    - Calibrate probabilities.
    """
    print("\n--- Step 3: Training Explainable GBM ---")
    
    kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    models = []
    
    # Features
    features = [c for c in X.columns if c not in ['claim_number', 'is_train']]
    X = X[features]
    X_test = X_test[features]
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Base Model
        clf = lgb.LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=SEED,
            n_jobs=-1,
            verbose=-1
        )
        
        clf.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        # Calibration (Platt Scaling)
        # Note: CalibratedClassifierCV with 'prefit' requires the model to be already fitted.
        # We use the validation set to calibrate.
        calibrator = CalibratedClassifierCV(clf, method='sigmoid', cv='prefit')
        calibrator.fit(X_val, y_val)
        
        # Predictions (Calibrated)
        val_pred = calibrator.predict_proba(X_val)[:, 1]
        test_pred = calibrator.predict_proba(X_test)[:, 1]
        
        oof_preds[val_idx] = val_pred
        test_preds += test_pred / N_FOLDS
        
        models.append(clf) # Save base model for SHAP (Calibrated wrapper obscures SHAP)
        
        auc = roc_auc_score(y_val, val_pred)
        print(f"Fold {fold+1} AUC: {auc:.4f}")
        
    print(f"Overall CV AUC: {roc_auc_score(y, oof_preds):.4f}")
    return oof_preds, test_preds, models, features

def run_xai(models, X_train, features, output_dir):
    """
    Step 4: Explainability (XAI)
    - SHAP Summary Plot
    - PDP Plots
    """
    print("\n--- Step 4: Generating XAI Artifacts ---")
    
    # Use the first model for explanation (representative)
    model = models[0]
    
    # SHAP
    print("Calculating SHAP values...")
    # TreeExplainer is fast for Trees
    explainer = shap.TreeExplainer(model)
    
    # Subsample for speed if needed, but X_train size is usually manageable
    X_sample = X_train[features].sample(min(1000, len(X_train)), random_state=SEED)
    shap_values = explainer.shap_values(X_sample)
    
    # Handle LightGBM binary classification SHAP output (sometimes list of arrays)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
        
    # Summary Plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title("SHAP Summary Plot (Top Features)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_summary.png"))
    plt.close()
    
    # Partial Dependence Plots for Top 3 Features
    # Calculate importance to find top features
    importance_df = pd.DataFrame({
        'feature': features,
        'importance': np.abs(shap_values).mean(axis=0)
    }).sort_values('importance', ascending=False)
    
    # Select top features that are numeric for PDP to avoid issues with categorical grids
    numeric_features = X_train.select_dtypes(include=['number']).columns.tolist()
    top_features = [f for f in importance_df['feature'].tolist() if f in numeric_features][:3]
    
    if top_features:
        print(f"Generating PDP for top numeric features: {top_features}")
        try:
            fig, ax = plt.subplots(figsize=(12, 4))
            PartialDependenceDisplay.from_estimator(
                model, 
                X_sample, 
                top_features, 
                ax=ax,
                kind='average'
            )
            plt.suptitle("Partial Dependence Plots (Top Numeric Drivers)")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "pdp_plots.png"))
            plt.close()
        except Exception as e:
            print(f"Skipping PDP due to error: {e}")
    else:
        print("No top numeric features found for PDP.")

    # Stability Analysis (Simple version: Check top 5 features across folds)
    # We'd need models from all folds, here we just show XAI for one. 
    # For governance, we'd log this.
    
    return importance_df

def main():
    # Ensure output directory exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # 1. Load Data
    df_all, target, train_ids, test_ids = load_data()
    
    # 2. Step 0: Audit
    df_audit = audit_missingness(df_all)
    
    # 3. Step 1: Imputation
    df_imputed = robust_imputation(df_audit)
    
    # 4. Step 2: Feature Selection
    df_selected = feature_selection(df_imputed, target, train_ids)
    
    # Prepare for Modeling
    n_train = len(target)
    train_df = df_selected.iloc[:n_train]
    test_df = df_selected.iloc[n_train:]
    
    # 5. Step 3: Modeling (Explainable GBM)
    oof_preds, test_preds, models, features = train_explainable_gbm(train_df, target, test_df)
    
    # 6. Step 4: XAI
    importance_df = run_xai(models, train_df, features, OUTPUT_DIR)
    
    # Save Feature Importance for Governance
    importance_df.to_csv(os.path.join(OUTPUT_DIR, "feature_importance_governance.csv"), index=False)
    
    # 7. Step 5: Submission & Metrics
    print("\n--- Step 5: Evaluation & Governance ---")
    
    # Find Best Threshold for F1
    prec, rec, thresholds = precision_recall_curve(target, oof_preds)
    f1_scores = 2 * (prec * rec) / (prec + rec + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    print(f"Best F1 Score: {best_f1:.4f} at threshold {best_thresh:.4f}")
    
    # Top N Strategy (Standardized)
    test_sorted_idx = np.argsort(test_preds)[::-1]
    targets_n = [2250, 2500, 2778]
    
    for n in targets_n:
        if n <= len(test_ids):
            thresh_idx = n - 1
            thresh_val = test_preds[test_sorted_idx[thresh_idx]]
            binary_preds = (test_preds >= thresh_val).astype(int)
            
            save_file = os.path.join(DATA_DIR, f'pipeline4_submission_top{n}.csv')
            sub = pd.DataFrame({'claim_number': test_ids, 'subrogation': binary_preds})
            sub.to_csv(save_file, index=False)
            print(f"Saved Top {n} to {save_file}")

    # Save Probabilities for Hybrid approach
    prob_file = os.path.join(DATA_DIR, 'pipeline4_submission_probs.csv')
    sub_prob = pd.DataFrame({'claim_number': test_ids, 'subrogation_prob': test_preds})
    sub_prob.to_csv(prob_file, index=False)
    print(f"Saved Probabilities to {prob_file}")

if __name__ == "__main__":
    main()

