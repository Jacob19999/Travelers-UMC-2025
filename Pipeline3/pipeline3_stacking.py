import pandas as pd
import numpy as np
import os
import sys
import warnings
import gc
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import xgboost as xgb
import lightgbm as lgb
from scipy.special import erfinv

# Optional: Check for tensorflow for MLP
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, callbacks
    HAS_TF = True
except ImportError:
    HAS_TF = False

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
SUB_PATH = os.path.join(DATA_DIR, 'pipeline3_submission.csv')

def rank_gauss(x):
    """RankGauss normalization."""
    N = x.shape[0]
    temp = x.argsort()
    rank_x = temp.argsort() / (N + 1)
    rank_x -= rank_x.mean()
    rank_x *= 2
    efi_x = erfinv(rank_x)
    efi_x -= efi_x.mean()
    return efi_x

def load_data():
    print("Loading data...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    
    train_df['is_train'] = 1
    test_df['is_train'] = 0
    
    # Store target and IDs
    target = train_df['subrogation']
    train_ids = train_df['claim_number']
    test_ids = test_df['claim_number']
    
    # Concatenate
    df = pd.concat([train_df.drop(['subrogation'], axis=1), test_df], axis=0).reset_index(drop=True)
    
    return df, target, train_ids, test_ids

def create_view_a(df_orig):
    """
    View A: Basic engineered features (similar to Pipeline 1/2).
    """
    print("Creating View A (Basic FE)...")
    df = df_orig.copy()
    
    # Date Features
    if 'claim_date' in df.columns:
        df['claim_date'] = pd.to_datetime(df['claim_date'])
        df['claim_year'] = df['claim_date'].dt.year
        df['claim_month'] = df['claim_date'].dt.month
        df['claim_day'] = df['claim_date'].dt.day
        df['claim_dayofweek'] = df['claim_date'].dt.dayofweek
        df.drop('claim_date', axis=1, inplace=True)
    
    # Age of Vehicle
    if 'vehicle_made_year' in df.columns and 'claim_year' in df.columns:
        df['age_of_vehicle'] = df['claim_year'] - df['vehicle_made_year']

    # Binary Conversions
    binary_map = {'Y': 1, 'N': 0, 'yes': 1, 'no': 0}
    for col in ['witness_present_ind', 'in_network_bodyshop', 'policy_report_filed_ind']:
        if col in df.columns:
            df[col] = df[col].map(binary_map).fillna(0)

    # Interaction Features
    if 'claim_est_payout' in df.columns and 'vehicle_price' in df.columns:
        df['payout_ratio'] = df['claim_est_payout'] / (df['vehicle_price'] + 1)
    if 'annual_income' in df.columns and 'vehicle_price' in df.columns:
        df['income_to_car_price'] = df['annual_income'] / (df['vehicle_price'] + 1)
    if 'year_of_born' in df.columns and 'claim_year' in df.columns:
        df['driver_age'] = df['claim_year'] - df['year_of_born']

    # Fill Missing Values
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = df[col].fillna('Unknown').astype(str)
        
    # Label Encoding for View A
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        
    return df

def create_view_b(df_orig):
    """
    View B: One-hot expanded for low-cardinality categoricals.
    """
    print("Creating View B (One-Hot)...")
    df = df_orig.copy()
    
    # Identify low cardinality categorical columns
    cat_cols = df.select_dtypes(include=['object']).columns
    low_card_cols = [c for c in cat_cols if df[c].nunique() < 50] # Threshold for "low"
    
    # We might want to keep numeric columns too, or just return the one-hot part?
    # Usually for "View B" we want the full dataset but with one-hot instead of LabelEnc.
    
    # First, do basic filling similar to View A so we don't have NaNs
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
        
    # One-hot encode
    df = pd.get_dummies(df, columns=low_card_cols, dummy_na=True)
    
    # Drop remaining high-cardinality object columns or Label Encode them?
    # Let's Label Encode remaining object columns (like zip_code if it's high card)
    remaining_cat = df.select_dtypes(include=['object']).columns
    for col in remaining_cat:
        le = LabelEncoder()
        df[col] = df[col].fillna('Unknown')
        df[col] = le.fit_transform(df[col])
        
    return df

def create_view_c(df_view_b, n_components=50):
    """
    View C: SVD / PCA projections.
    """
    print(f"Creating View C (SVD n={n_components})...")
    # Use View B (One-Hot) as base
    svd = TruncatedSVD(n_components=n_components, random_state=SEED)
    # Normalize before SVD? Usually good idea
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_view_b)
    
    X_svd = svd.fit_transform(X_scaled)
    
    df_svd = pd.DataFrame(X_svd, columns=[f'svd_{i}' for i in range(n_components)])
    
    # We can return just the SVD components, or append to View A.
    # Prompt says: "Append these components to view A or use separately."
    # Let's return just the components for now, can append later if needed by model.
    return df_svd

def create_view_d(df_view_a, k=20):
    """
    View D: Cluster-distance features.
    """
    print(f"Creating View D (KMeans k={k})...")
    # Use View A (Numeric + LE)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_view_a)
    
    kmeans = KMeans(n_clusters=k, random_state=SEED, n_init=10)
    kmeans.fit(X_scaled)
    
    # Distances to centroids
    dists = kmeans.transform(X_scaled)
    
    # Create DataFrame
    df_d = pd.DataFrame(dists, columns=[f'dist_clust_{i}' for i in range(k)])
    df_d['cluster_id'] = kmeans.labels_
    
    return df_d

# --- Level 1 Models ---

def get_oof_predictions(model_name, model, X_train, y_train, X_test, is_neural_net=False):
    print(f"Training {model_name}...")
    kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    
    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        if is_neural_net and HAS_TF:
             # Simple MLP handling
             # Scale data
             scaler = StandardScaler()
             X_tr_s = scaler.fit_transform(X_tr)
             X_val_s = scaler.transform(X_val)
             X_test_s = scaler.transform(X_test) # re-transforming X_test every fold is inefficient but simple
             
             model_instance = keras.models.clone_model(model)
             model_instance.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
             es = callbacks.EarlyStopping(monitor='val_auc', mode='max', patience=5, restore_best_weights=True)
             
             model_instance.fit(X_tr_s, y_tr, validation_data=(X_val_s, y_val), 
                                epochs=30, batch_size=512, callbacks=[es], verbose=0)
             
             val_pred = model_instance.predict(X_val_s).flatten()
             test_pred = model_instance.predict(X_test_s).flatten()
             
        else:
            # Tree models / Sklearn
            try:
                model.fit(X_tr, y_tr)
            except:
                # Some models might need eval_set (LGB/XGB)
                if 'XGB' in model_name or 'LGB' in model_name:
                    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
                else:
                    model.fit(X_tr, y_tr)
            
            if hasattr(model, 'predict_proba'):
                val_pred = model.predict_proba(X_val)[:, 1]
                test_pred = model.predict_proba(X_test)[:, 1]
            else:
                val_pred = model.predict(X_val) # Usually not for binary class unless regression
                test_pred = model.predict(X_test)

        oof_preds[val_idx] = val_pred
        test_preds += test_pred / N_FOLDS
        
    auc = roc_auc_score(y_train, oof_preds)
    print(f"  {model_name} OOF AUC: {auc:.4f}")
    return oof_preds, test_preds

def build_mlp(input_dim):
    if not HAS_TF: return None
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

def main():
    # 1. Load Data
    df_all, target, train_ids, test_ids = load_data()
    
    # 2. Create Views
    print("\n--- Creating Feature Views ---")
    view_a = create_view_a(df_all) # Basic
    view_b = create_view_b(df_all) # One-Hot
    view_c = create_view_c(view_b, n_components=30) # SVD from View B
    view_d = create_view_d(view_a, k=15) # Clusters from View A
    
    # Prepare splits
    n_train = len(target)
    
    X_train_A = view_a.iloc[:n_train]
    X_test_A = view_a.iloc[n_train:]
    
    X_train_B = view_b.iloc[:n_train]
    X_test_B = view_b.iloc[n_train:]
    
    # View C and D are just features, append to A or use standalone?
    # Prompt: "Append these components to view A or use separately."
    # Let's try appending C and D to A for some models, or treating them as separate views.
    # For simplicity/diversity, let's define:
    # View AC = A + C
    # View AD = A + D
    
    view_ac = pd.concat([view_a, view_c], axis=1)
    X_train_AC = view_ac.iloc[:n_train]
    X_test_AC = view_ac.iloc[n_train:]

    view_ad = pd.concat([view_a, view_d], axis=1)
    X_train_AD = view_ad.iloc[:n_train]
    X_test_AD = view_ad.iloc[n_train:]
    
    # 3. Level 1 Models
    print("\n--- Training Level 1 Models ---")
    
    meta_train = pd.DataFrame()
    meta_test = pd.DataFrame()
    
    # Model definitions
    models_config = [
        # Model Name, Model Object, View (X_train, X_test)
        ('XGB_A', xgb.XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=SEED), X_train_A, X_test_A),
        ('LGB_A', lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31, subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=SEED, verbose=-1), X_train_A, X_test_A),
        ('RF_A', RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=5, n_jobs=-1, random_state=SEED), X_train_A, X_test_A),
        ('ET_A', ExtraTreesClassifier(n_estimators=200, max_depth=10, min_samples_leaf=5, n_jobs=-1, random_state=SEED), X_train_A, X_test_A),
        # XGB on View AC (with SVD)
        ('XGB_AC', xgb.XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6, n_jobs=-1, random_state=SEED), X_train_AC, X_test_AC),
        # RF on View AD (with Clusters)
        ('RF_AD', RandomForestClassifier(n_estimators=200, max_depth=10, n_jobs=-1, random_state=SEED), X_train_AD, X_test_AD),
        # Logit on View B (One-Hot) - Good for linear patterns
        ('Logit_B', LogisticRegression(penalty='l2', C=1.0, solver='liblinear', random_state=SEED), X_train_B, X_test_B),
    ]
    
    for name, model, X_tr, X_te in models_config:
        oof, pred = get_oof_predictions(name, model, X_tr, target, X_te)
        meta_train[name] = oof
        meta_test[name] = pred
        
    # Optional NN
    if HAS_TF:
        print("Training NN_A...")
        mlp_model = build_mlp(X_train_A.shape[1])
        oof, pred = get_oof_predictions('NN_A', mlp_model, X_train_A, target, X_test_A, is_neural_net=True)
        meta_train['NN_A'] = oof
        meta_test['NN_A'] = pred
        
    # 4. Level 2 Meta-Model
    print("\n--- Training Level 2 Meta-Models ---")
    
    X_meta = meta_train
    X_meta_test = meta_test
    y_meta = target
    
    # Meta-Model 1: Logistic Regression
    print("Meta-Model: Logistic Regression")
    meta_logit = LogisticRegression(penalty='l2', C=1.0, random_state=SEED)
    meta_logit_oof, meta_logit_pred = get_oof_predictions("Meta_Logit", meta_logit, X_meta, y_meta, X_meta_test)
    
    # Meta-Model 2: XGBoost (Small)
    print("Meta-Model: XGBoost")
    meta_xgb = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=SEED)
    meta_xgb_oof, meta_xgb_pred = get_oof_predictions("Meta_XGB", meta_xgb, X_meta, y_meta, X_meta_test)
    
    # Meta-Model 3: LightGBM (Small)
    print("Meta-Model: LightGBM")
    meta_lgb = lgb.LGBMClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=SEED, verbose=-1)
    meta_lgb_oof, meta_lgb_pred = get_oof_predictions("Meta_LGB", meta_lgb, X_meta, y_meta, X_meta_test)
    
    # 5. Final Blending
    # Simple average of Meta-Models or weighted average
    print("\n--- Final Ensemble Blending ---")
    
    # Let's blend Logit and LGB (as suggested)
    final_oof = 0.5 * meta_lgb_oof + 0.5 * meta_logit_oof
    final_pred = 0.5 * meta_lgb_pred + 0.5 * meta_logit_pred
    
    print(f"Final Ensemble OOF AUC: {roc_auc_score(y_meta, final_oof):.4f}")
    
    # Find Best Threshold for F1
    prec, rec, thresholds = precision_recall_curve(y_meta, final_oof)
    f1_scores = 2 * (prec * rec) / (prec + rec + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    print(f"Best Threshold (CV): {best_thresh:.4f}, Best F1 (CV): {best_f1:.4f}")
    
    # 6. Generate Submissions
    print("\nGenerating Submissions...")
    
    # Top N Strategy
    test_sorted_idx = np.argsort(final_pred)[::-1]
    targets = [2250, 2500, 2778]
    
    for n in targets:
        if n <= len(test_ids):
            thresh_idx = n - 1
            thresh_val = final_pred[test_sorted_idx[thresh_idx]]
            binary_preds = (final_pred >= thresh_val).astype(int)
            
            save_file = os.path.join(DATA_DIR, f'pipeline3_submission_top{n}.csv')
            sub = pd.DataFrame({'claim_number': test_ids, 'subrogation': binary_preds})
            sub.to_csv(save_file, index=False)
            print(f"Saved Top {n} to {save_file}")

    # Best Threshold Submission
    binary_preds_best = (final_pred >= best_thresh).astype(int)
    save_file_best = os.path.join(DATA_DIR, 'pipeline3_submission_best_f1.csv')
    sub = pd.DataFrame({'claim_number': test_ids, 'subrogation': binary_preds_best})
    sub.to_csv(save_file_best, index=False)
    print(f"Saved Best F1 submission to {save_file_best}")

if __name__ == "__main__":
    main()

