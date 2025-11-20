import sys
import subprocess

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def check_and_install_packages():
    packages = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'tensorflow': 'tensorflow',
        'scikit-learn': 'sklearn',
        'scipy': 'scipy',
        'lightgbm': 'lightgbm',
        'catboost': 'catboost',
        'optuna': 'optuna'
    }
    
    for package, import_name in packages.items():
        try:
            __import__(import_name)
        except ImportError:
            print(f"Installing {package}...")
            install(package)

check_and_install_packages()

import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.special import erfinv
import lightgbm as lgb
from catboost import CatBoostClassifier
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve
import warnings
import random
import itertools
from scipy.optimize import minimize

warnings.filterwarnings('ignore')

warnings.filterwarnings('ignore')

# Set random seeds
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'Data')
TRAIN_PATH = os.path.join(DATA_DIR, 'Training_TriGuard.csv')
TEST_PATH = os.path.join(DATA_DIR, 'Testing_TriGuard.csv')
SUBMISSION_PATH = os.path.join(DATA_DIR, 'pipeline2_submission.csv')

def rank_gauss(x):
    """
    RankGauss normalization:
    1. Sort values -> Rank
    2. Scale to (0, 1)
    3. Apply inverse error function
    """
    N = x.shape[0]
    temp = x.argsort()
    rank_x = temp.argsort() / (N + 1)
    rank_x -= rank_x.mean()
    rank_x *= 2
    efi_x = erfinv(rank_x)
    efi_x -= efi_x.mean()
    return efi_x

def swap_noise(X, p=0.15):
    """
    Apply swap noise to a batch X.
    For each feature, swap with a value from another random row with probability p.
    """
    X_noisy = X.copy()
    rows, cols = X.shape
    for j in range(cols):
        mask = np.random.rand(rows) < p
        n_swap = np.sum(mask)
        if n_swap > 0:
            X_noisy[mask, j] = np.random.choice(X[:, j], size=n_swap)
    return X_noisy

class DAEDataGenerator(keras.utils.Sequence):
    def __init__(self, X_num_bin, X_cat, cat_cols, batch_size=1024, shuffle=True, p_swap=0.15):
        self.X_num_bin = X_num_bin # Numeric + Binary
        self.X_cat = X_cat # List of categorical arrays
        self.cat_cols = cat_cols
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.p_swap = p_swap
        self.indices = np.arange(len(X_num_bin))
        
    def __len__(self):
        return int(np.ceil(len(self.X_num_bin) / self.batch_size))
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
            
    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        
        # Gather batch data
        batch_num_bin = self.X_num_bin[batch_indices]
        batch_cats = [cat[batch_indices] for cat in self.X_cat]
        
        # Apply noise
        noisy_num_bin = swap_noise(batch_num_bin, self.p_swap)
        
        # Prepare Inputs (Dictionary)
        inputs = {'input_num_bin': noisy_num_bin}
        
        for i, cat_col in enumerate(batch_cats):
            # Reshape to (N,) if needed
            c_flat = cat_col.flatten()
            # We can swap noise categorical indices too
            c_noisy = swap_noise(c_flat.reshape(-1, 1), self.p_swap).flatten()
            inputs[f'input_{self.cat_cols[i]}'] = c_noisy
            
        # Targets (Dictionary)
        targets = {'output_num_bin': batch_num_bin}
        for i, cat in enumerate(batch_cats):
            targets[f'output_{self.cat_cols[i]}'] = cat
            
        return inputs, targets

def load_and_preprocess():
    print("Loading data...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    
    train_df['is_train'] = 1
    test_df['is_train'] = 0
    
    # Drop target and IDs for feature engineering
    target = train_df['subrogation']
    train_ids = train_df['claim_number']
    test_ids = test_df['claim_number']
    
    df = pd.concat([train_df.drop(['subrogation'], axis=1), test_df], axis=0).reset_index(drop=True)
    
    # --- Feature Engineering ---
    print("Feature Engineering...")
    
    # Date Features
    df['claim_date'] = pd.to_datetime(df['claim_date'])
    df['claim_year'] = df['claim_date'].dt.year
    df['claim_month'] = df['claim_date'].dt.month
    df['claim_day'] = df['claim_date'].dt.day
    df['claim_dayofweek'] = df['claim_date'].dt.dayofweek
    
    # Age of Vehicle
    if 'vehicle_made_year' in df.columns:
        df['age_of_vehicle'] = df['claim_year'] - df['vehicle_made_year']
    
    # Binary Conversions
    binary_map = {'Y': 1, 'N': 0, 'yes': 1, 'no': 0}
    if 'witness_present_ind' in df.columns:
        df['witness_present_ind'] = df['witness_present_ind'].map(binary_map).fillna(0)
    if 'in_network_bodyshop' in df.columns:
        df['in_network_bodyshop'] = df['in_network_bodyshop'].map(binary_map).fillna(0)
        
    # --- Interaction Features ---
    print("Creating Interaction Features...")
    # Interactions between key numeric features
    key_interactions = ['liab_prct', 'claim_est_payout', 'age_of_vehicle', 'vehicle_price', 'annual_income', 'past_num_of_claims']
    key_interactions = [c for c in key_interactions if c in df.columns]
    
    for col1, col2 in itertools.combinations(key_interactions, 2):
        df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
        df[f'{col1}_div_{col2}'] = df[col1] / (df[col2] + 1e-5)

    # --- Enhanced Feature Engineering ---
    if 'claim_est_payout' in df.columns and 'vehicle_price' in df.columns:
        df['payout_ratio'] = df['claim_est_payout'] / (df['vehicle_price'] + 1)
        
    if 'annual_income' in df.columns and 'vehicle_price' in df.columns:
        df['income_to_car_price'] = df['annual_income'] / (df['vehicle_price'] + 1)
        
    if 'year_of_born' in df.columns and 'claim_year' in df.columns:
        df['driver_age'] = df['claim_year'] - df['year_of_born']
        if 'age_of_DL' in df.columns:
             df['dl_ratio'] = df['age_of_DL'] / (df['driver_age'] - 16 + 1e-5)

    # --- Column Types ---
    # Based on inspection
    numeric_cols = [
        'year_of_born', 'safety_rating', 'annual_income', 'past_num_of_claims', 
        'liab_prct', 'claim_est_payout', 'vehicle_made_year', 'vehicle_price', 
        'vehicle_weight', 'age_of_DL', 'vehicle_mileage', 'age_of_vehicle',
        'payout_ratio', 'income_to_car_price', 'driver_age', 'dl_ratio'
    ]
    
    # Add interaction columns to numeric_cols
    interaction_cols = [c for c in df.columns if '_x_' in c or '_div_' in c]
    numeric_cols.extend(interaction_cols)
    
    binary_cols = [
        'email_or_tel_available', 'high_education_ind', 'address_change_ind', 
        'policy_report_filed_ind', 'witness_present_ind', 'in_network_bodyshop'
    ]
    
    cat_cols = [
        'gender', 'living_status', 'zip_code', 'claim_day_of_week', 
        'accident_site', 'channel', 'vehicle_category', 'vehicle_color', 'accident_type'
    ]
    
    # Verify columns exist
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    binary_cols = [c for c in binary_cols if c in df.columns]
    cat_cols = [c for c in cat_cols if c in df.columns]
    
    # Fill Missing Values
    # Numeric: Median
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
        
    # Binary: Mode or 0
    for col in binary_cols:
        df[col] = df[col].fillna(0)
        
    # Categorical: "Unknown"
    for col in cat_cols:
        df[col] = df[col].fillna('Unknown').astype(str)
        
    # --- RankGauss for Numeric ---
    print("Applying RankGauss...")
    for col in numeric_cols:
        df[col] = rank_gauss(df[col].values)
        
    # --- Label Encoding for Categorical ---
    print("Label Encoding...")
    cat_counts = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        cat_counts[col] = len(le.classes_)
        
    # Split back
    train_processed = df[df['is_train'] == 1].drop(['is_train', 'claim_number', 'claim_date'], axis=1)
    test_processed = df[df['is_train'] == 0].drop(['is_train', 'claim_number', 'claim_date'], axis=1)
    
    return train_processed, test_processed, target, train_ids, test_ids, numeric_cols, binary_cols, cat_cols, cat_counts

def build_dae(input_dim_num_bin, cat_counts, cat_cols, bottleneck_dim=128):
    # Inputs
    input_num_bin = layers.Input(shape=(input_dim_num_bin,), name='input_num_bin')
    inputs = [input_num_bin]
    
    # Embeddings
    embeddings = []
    input_cats = []
    
    for col in cat_cols:
        n_classes = cat_counts[col]
        embed_dim = min(50, (n_classes + 1) // 2)
        input_cat = layers.Input(shape=(1,), name=f'input_{col}')
        input_cats.append(input_cat)
        emb = layers.Embedding(input_dim=n_classes, output_dim=embed_dim, name=f'embed_{col}')(input_cat)
        embeddings.append(layers.Flatten()(emb))
        
    inputs.extend(input_cats)
    
    # Concatenate
    if embeddings:
        merged = layers.Concatenate()([input_num_bin] + embeddings)
    else:
        merged = input_num_bin
        
    # Encoder
    x = layers.Dense(1000, activation='relu')(merged)
    x = layers.Dense(500, activation='relu')(x)
    bottleneck = layers.Dense(bottleneck_dim, activation='relu', name='bottleneck')(x)
    
    # Decoder
    d = layers.Dense(500, activation='relu')(bottleneck)
    d = layers.Dense(1000, activation='relu')(d)
    
    # Outputs (Reconstruction)
    # 1. Numeric + Binary Reconstruction
    output_num_bin = layers.Dense(input_dim_num_bin, activation='linear', name='output_num_bin')(d)
    outputs = [output_num_bin]
    
    # 2. Categorical Reconstruction (Classification)
    for col in cat_cols:
        n_classes = cat_counts[col]
        out_cat = layers.Dense(n_classes, activation='softmax', name=f'output_{col}')(d)
        outputs.append(out_cat)
        
    model = models.Model(inputs=inputs, outputs=outputs)
    
    # Loss dictionary
    losses = {'output_num_bin': 'mse'}
    loss_weights = {'output_num_bin': 1.0}
    for col in cat_cols:
        losses[f'output_{col}'] = 'sparse_categorical_crossentropy'
        loss_weights[f'output_{col}'] = 0.5 # Weight categorical losses
        
    model.compile(optimizer='adam', loss=losses, loss_weights=loss_weights)
    return model

def get_dae_features(model, X_num_bin, X_cat, cat_cols):
    # Create a sub-model that outputs the bottleneck
    bottleneck_model = models.Model(inputs=model.inputs, outputs=model.get_layer('bottleneck').output)
    
    # Build input dictionary
    inputs = {'input_num_bin': X_num_bin}
    for i, col in enumerate(cat_cols):
        inputs[f'input_{col}'] = X_cat[:, i]
        
    features = bottleneck_model.predict(inputs, batch_size=1024, verbose=1)
    return features

def build_nn_classifier(input_dim):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
    return model

def tune_lightgbm(X, y):
    print("\nTuning LightGBM with Optuna...")
    def objective(trial):
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'scale_pos_weight': 3.32,  # Fixed based on probe
            'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
        }
        
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        aucs = []
        
        for train_idx, val_idx in skf.split(X, y):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model = lgb.LGBMClassifier(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False)]
            )
            preds = model.predict_proba(X_val)[:, 1]
            aucs.append(roc_auc_score(y_val, preds))
            
        return np.mean(aucs)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30) # 30 trials for speed
    print(f"Best params: {study.best_params}")
    
    # Ensure fixed params are included in best_params
    best_params = study.best_params.copy()
    best_params['scale_pos_weight'] = 3.32
    best_params['objective'] = 'binary'
    best_params['metric'] = 'auc'
    best_params['verbosity'] = -1
    best_params['boosting_type'] = 'gbdt'
    
    return best_params

def optimize_ensemble_weights(oof_list, y_true):
    """Find optimal weights for ensemble to maximize F1"""
    print("\nOptimizing Ensemble Weights for F1...")
    
    n_models = len(oof_list)
    
    def f1_loss(weights):
        # Normalize weights
        w = np.array(weights)
        if np.sum(w) == 0: return 0
        w = w / np.sum(w)
        
        ensemble_pred = np.zeros_like(oof_list[0])
        for i in range(n_models):
            ensemble_pred += w[i] * oof_list[i]
        
        # Find best F1 for this ensemble combination
        precision, recall, thresholds = precision_recall_curve(y_true, ensemble_pred)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        # Handle NaNs
        f1_scores = np.nan_to_num(f1_scores)
        best_f1 = np.max(f1_scores)
        
        return -best_f1
    
    # Initial weights
    init_weights = [1.0/n_models] * n_models
    bounds = tuple([(0, 1)] * n_models)
    
    # Use method that handles bounds well
    result = minimize(f1_loss, init_weights, bounds=bounds, method='SLSQP')
    best_weights = result.x / np.sum(result.x)
    
    print(f"Best Weights: {best_weights}")
    print(f"Best Ensemble F1 (CV): {-result.fun:.4f}")
    return best_weights

def main():
    # 1. Preprocessing
    X_train_full, X_test_full, y_train, train_ids, test_ids, num_cols, bin_cols, cat_cols, cat_counts = load_and_preprocess()
    
    # Prepare arrays for DAE
    num_bin_cols = num_cols + bin_cols
    X_train_num_bin = X_train_full[num_bin_cols].values.astype('float32')
    X_test_num_bin = X_test_full[num_bin_cols].values.astype('float32')
    
    X_train_cat = X_train_full[cat_cols].values.astype('int32')
    X_test_cat = X_test_full[cat_cols].values.astype('int32')
    
    # Concatenate for DAE training (Unsupervised)
    X_all_num_bin = np.vstack([X_train_num_bin, X_test_num_bin])
    X_all_cat = np.vstack([X_train_cat, X_test_cat])
    
    # 2. Train DAE (3 Variants)
    print("\nTraining DAE (3 Variants)...")
    
    # Prepare arrays for DAE
    X_all_cat_list = [X_all_cat[:, i] for i in range(len(cat_cols))]
    
    # Validation split for DAE (just use last 10% of data)
    val_len = int(len(X_all_num_bin) * 0.1)
    X_train_dae_nb = X_all_num_bin[:-val_len]
    X_val_dae_nb = X_all_num_bin[-val_len:]
    X_train_dae_cat = [c[:-val_len] for c in X_all_cat_list]
    X_val_dae_cat = [c[-val_len:] for c in X_all_cat_list]
    
    # Validation inputs and targets (Dictionaries)
    val_inputs = {'input_num_bin': X_val_dae_nb}
    val_targets = {'output_num_bin': X_val_dae_nb}
    
    for i, col in enumerate(cat_cols):
        val_inputs[f'input_{col}'] = X_val_dae_cat[i]
        val_targets[f'output_{col}'] = X_val_dae_cat[i]

    dae_variants_train = []
    dae_variants_test = []
    
    for v in range(3):
        print(f"\n--- DAE Variant {v+1}/3 ---")
        # Perturb seed for diversity
        current_seed = SEED + v * 111
        tf.random.set_seed(current_seed)
        np.random.seed(current_seed)
        random.seed(current_seed)
        
        dae_model = build_dae(len(num_bin_cols), cat_counts, cat_cols, bottleneck_dim=256)
        
        # Re-create generator with new seed context
        train_gen = DAEDataGenerator(X_train_dae_nb, X_train_dae_cat, cat_cols, batch_size=1024, p_swap=0.15)
        
        early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        dae_model.fit(
            train_gen,
            validation_data=(val_inputs, val_targets),
            epochs=30, 
            callbacks=[early_stopping],
            verbose=1
        )
        
        print(f"Extracting features for Variant {v+1}...")
        feats_tr = get_dae_features(dae_model, X_train_num_bin, X_train_cat, cat_cols)
        feats_te = get_dae_features(dae_model, X_test_num_bin, X_test_cat, cat_cols)
        
        dae_variants_train.append(feats_tr)
        dae_variants_test.append(feats_te)
        
    # Concatenate features
    train_dae_feats = np.hstack(dae_variants_train)
    test_dae_feats = np.hstack(dae_variants_test)
    
    print(f"Combined DAE Features Shape: {train_dae_feats.shape}")
    
    # 4. Supervised Training
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    
    # Helper for OOF and Preds
    def get_oof_preds(model_name, model_fn, X_train, X_test, y_train, **kwargs):
        print(f"\nTraining {model_name}...")
        oof = np.zeros(len(y_train))
        test_preds = np.zeros(len(X_test))
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            model = model_fn(**kwargs)
            
            if 'fit_params' in kwargs:
                fit_p = kwargs['fit_params'].copy()
                if 'eval_set' in fit_p: # Placeholder logic
                    fit_p['eval_set'] = [(X_val, y_val)]
                model.fit(X_tr, y_tr, **fit_p)
                pred_val = model.predict_proba(X_val)[:, 1]
                pred_test = model.predict_proba(X_test)[:, 1]
            elif 'nn' in model_name.lower():
                # NN special handling
                neg, pos = np.bincount(y_tr)
                total = neg + pos
                class_weight = {0: (1 / neg) * (total / 2.0), 1: (1 / pos) * (total / 2.0)}
                es = callbacks.EarlyStopping(monitor='val_auc', mode='max', patience=10, restore_best_weights=True)
                model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=50, batch_size=512, 
                          class_weight=class_weight, callbacks=[es], verbose=0)
                pred_val = model.predict(X_val).flatten()
                pred_test = model.predict(X_test).flatten()
            else:
                # LGB/Cat generic
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
                pred_val = model.predict_proba(X_val)[:, 1]
                pred_test = model.predict_proba(X_test)[:, 1]
                
            oof[val_idx] = pred_val
            test_preds += pred_test / 5
            
        print(f"{model_name} OOF AUC: {roc_auc_score(y_train, oof):.4f}")
        return oof, test_preds

    # --- Model Definitions ---
    
    # Model 1: NN on DAE
    def create_nn_dae():
        return build_nn_classifier(train_dae_feats.shape[1])
        
    oof_nn_dae, pred_nn_dae = get_oof_preds("NN (DAE)", create_nn_dae, train_dae_feats, test_dae_feats, y_train)
    
    # Combine features for tree models
    X_train_combined = np.hstack([X_train_full.values, train_dae_feats])
    X_test_combined = np.hstack([X_test_full.values, test_dae_feats])
    
    # Feature names for LGB
    orig_feat_names = list(X_train_full.columns)
    dae_feat_names = [f'dae_{i}' for i in range(train_dae_feats.shape[1])]
    feature_names = orig_feat_names + dae_feat_names
    feature_names_clean = [f.replace(' ', '_').replace(':', '_') for f in feature_names]
    
    # Model 2: LGBM Tuned (DAE+Raw)
    best_lgb_params = tune_lightgbm(X_train_combined, y_train)
    best_lgb_params['random_state'] = SEED
    best_lgb_params['verbose'] = -1
    
    def create_lgb():
        return lgb.LGBMClassifier(**best_lgb_params)
        
    # LGB fit params handled inside wrapper logic roughly, but let's be explicit for manual loop if needed. 
    # For simplicity using manual loop logic above adapted.
    # Actually, let's just run the loop manually for control over callbacks
    
    print("\nTraining Model 2 (LGBM Tuned)...")
    oof_lgb, pred_lgb = np.zeros(len(y_train)), np.zeros(len(test_ids))
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_combined, y_train)):
        X_tr, X_val = X_train_combined[train_idx], X_train_combined[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        model = lgb.LGBMClassifier(**best_lgb_params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
        oof_lgb[val_idx] = model.predict_proba(X_val)[:, 1]
        pred_lgb += model.predict_proba(X_test_combined)[:, 1] / 5
    print(f"LGBM OOF AUC: {roc_auc_score(y_train, oof_lgb):.4f}")

    # Model 3: CatBoost Default (DAE+Raw)
    cat_params = {
        'iterations': 1000, 'learning_rate': 0.03, 'depth': 6, 'l2_leaf_reg': 3,
        'loss_function': 'Logloss', 'eval_metric': 'AUC', 'random_seed': SEED,
        'verbose': False, 'early_stopping_rounds': 50, 'scale_pos_weight': 3.32
    }
    print("\nTraining Model 3 (CatBoost Default)...")
    oof_cat, pred_cat = np.zeros(len(y_train)), np.zeros(len(test_ids))
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_combined, y_train)):
        X_tr, X_val = X_train_combined[train_idx], X_train_combined[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        model = CatBoostClassifier(**cat_params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        oof_cat[val_idx] = model.predict_proba(X_val)[:, 1]
        pred_cat += model.predict_proba(X_test_combined)[:, 1] / 5
    print(f"CatBoost OOF AUC: {roc_auc_score(y_train, oof_cat):.4f}")
    
    # Model 4: CatBoost Depth 4 (DAE+Raw)
    cat_params_d4 = cat_params.copy()
    cat_params_d4['depth'] = 4
    cat_params_d4['iterations'] = 1500 # More iters for shallower trees
    
    print("\nTraining Model 4 (CatBoost Depth 4)...")
    oof_cat_d4, pred_cat_d4 = np.zeros(len(y_train)), np.zeros(len(test_ids))
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_combined, y_train)):
        X_tr, X_val = X_train_combined[train_idx], X_train_combined[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        model = CatBoostClassifier(**cat_params_d4)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        oof_cat_d4[val_idx] = model.predict_proba(X_val)[:, 1]
        pred_cat_d4 += model.predict_proba(X_test_combined)[:, 1] / 5
    print(f"CatBoost D4 OOF AUC: {roc_auc_score(y_train, oof_cat_d4):.4f}")
    
    # Model 5: LGBM Monotonic (DAE+Raw)
    # Constraint: liab_prct should be negative (higher liab -> less subrogation)
    monotone_constraints = [0] * len(feature_names)
    if 'liab_prct' in orig_feat_names:
        idx = orig_feat_names.index('liab_prct')
        monotone_constraints[idx] = -1
        print("Applied monotonic constraint to liab_prct")
        
    lgb_params_mono = best_lgb_params.copy()
    lgb_params_mono['monotone_constraints'] = monotone_constraints
    
    print("\nTraining Model 5 (LGBM Monotonic)...")
    oof_lgb_mono, pred_lgb_mono = np.zeros(len(y_train)), np.zeros(len(test_ids))
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_combined, y_train)):
        X_tr, X_val = X_train_combined[train_idx], X_train_combined[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        model = lgb.LGBMClassifier(**lgb_params_mono)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
        oof_lgb_mono[val_idx] = model.predict_proba(X_val)[:, 1]
        pred_lgb_mono += model.predict_proba(X_test_combined)[:, 1] / 5
    print(f"LGBM Mono OOF AUC: {roc_auc_score(y_train, oof_lgb_mono):.4f}")
    
    # Model 6: NN on Raw (Numeric) + DAE
    # We avoid using integer encoded cats in NN
    X_train_nn = np.hstack([X_train_num_bin, train_dae_feats])
    X_test_nn = np.hstack([X_test_num_bin, test_dae_feats])
    
    def create_nn_raw_dae():
        return build_nn_classifier(X_train_nn.shape[1])
    
    oof_nn_raw, pred_nn_raw = get_oof_preds("NN (Raw+DAE)", create_nn_raw_dae, X_train_nn, X_test_nn, y_train)

    # 6. Ensembling
    print("\nEnsembling 6 Models...")
    
    oof_list = [oof_nn_dae, oof_lgb, oof_cat, oof_cat_d4, oof_lgb_mono, oof_nn_raw]
    pred_list = [pred_nn_dae, pred_lgb, pred_cat, pred_cat_d4, pred_lgb_mono, pred_nn_raw]
    
    # Optimize weights
    best_weights = optimize_ensemble_weights(oof_list, y_train)
    
    ensemble_oof = np.zeros_like(oof_list[0])
    ensemble_test = np.zeros_like(pred_list[0])
    
    for i, w in enumerate(best_weights):
        ensemble_oof += w * oof_list[i]
        ensemble_test += w * pred_list[i]
    
    print(f"Ensemble OOF AUC: {roc_auc_score(y_train, ensemble_oof):.4f}")
    
    # Find optimal threshold for F1
    prec, rec, thresholds = precision_recall_curve(y_train, ensemble_oof)
    f1_scores = 2 * (prec * rec) / (prec + rec + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    print(f"Best Threshold (CV): {best_thresh:.4f}, Best F1 (CV): {best_f1:.4f}")
    
    # Check specific threshold range
    print("\nChecking specific threshold range 0.45-0.48:")
    for t in np.arange(0.45, 0.49, 0.01):
        y_pred_t = (ensemble_oof >= t).astype(int)
        score = f1_score(y_train, y_pred_t)
        print(f"Threshold {t:.2f}: F1 = {score:.4f}")
    
    # Generate Submissions
    print("\nGenerating Submissions...")
    
    # 1. Best CV Threshold
    final_preds_best = (ensemble_test >= best_thresh).astype(int)
    save_path_best = os.path.join(DATA_DIR, f'pipeline2_submission_best_f1.csv')
    pd.DataFrame({'claim_number': test_ids, 'subrogation': final_preds_best}).to_csv(save_path_best, index=False)
    print(f"Saved Best CV {best_thresh:.4f} -> {save_path_best}")
    
    # 2. Lower Threshold (0.47 approx from range)
    final_preds_low = (ensemble_test >= 0.47).astype(int)
    save_path_low = os.path.join(DATA_DIR, f'pipeline2_submission_0.47.csv')
    pd.DataFrame({'claim_number': test_ids, 'subrogation': final_preds_low}).to_csv(save_path_low, index=False)
    print(f"Saved Low Thresh 0.47 -> {save_path_low}")
    
    # 3. Top N Strategy
    test_sorted_idx = np.argsort(ensemble_test)[::-1]
    targets = [2500, 2750, 3000, 3250]
    for n in targets:
        if n-1 < len(test_sorted_idx):
            thresh_val = ensemble_test[test_sorted_idx[n-1]]
            final_preds = (ensemble_test >= thresh_val).astype(int)
            save_path = os.path.join(DATA_DIR, f'pipeline2_submission_top{n}.csv')
            pd.DataFrame({'claim_number': test_ids, 'subrogation': final_preds}).to_csv(save_path, index=False)
            print(f"Saved Top {n} -> {save_path}")
if __name__ == '__main__':
    main()
