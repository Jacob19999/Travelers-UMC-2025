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
    key_interactions = ['liab_prct', 'claim_est_payout', 'age_of_vehicle', 'vehicle_price', 'annual_income']
    key_interactions = [c for c in key_interactions if c in df.columns]
    
    for col1, col2 in itertools.combinations(key_interactions, 2):
        df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
        df[f'{col1}_div_{col2}'] = df[col1] / (df[col2] + 1e-5)

    # --- Column Types ---
    # Based on inspection
    numeric_cols = [
        'year_of_born', 'safety_rating', 'annual_income', 'past_num_of_claims', 
        'liab_prct', 'claim_est_payout', 'vehicle_made_year', 'vehicle_price', 
        'vehicle_weight', 'age_of_DL', 'vehicle_mileage', 'age_of_vehicle'
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

def build_dae(input_dim_num_bin, cat_counts, cat_cols):
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
    bottleneck = layers.Dense(128, activation='relu', name='bottleneck')(x)
    
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
            'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
            'is_unbalance': True
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
    return study.best_params

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
    
    # 2. Train DAE
    print("\nTraining DAE...")
    dae_model = build_dae(len(num_bin_cols), cat_counts, cat_cols)
    
    # Prepare Data Generator for DAE
    # We need to pass list of categorical arrays
    X_all_cat_list = [X_all_cat[:, i] for i in range(len(cat_cols))]
    
    dae_gen = DAEDataGenerator(X_all_num_bin, X_all_cat_list, cat_cols, batch_size=1024, p_swap=0.15)
    
    # Validation split for DAE (just use last 10% of data)
    val_len = int(len(X_all_num_bin) * 0.1)
    X_train_dae_nb = X_all_num_bin[:-val_len]
    X_val_dae_nb = X_all_num_bin[-val_len:]
    X_train_dae_cat = [c[:-val_len] for c in X_all_cat_list]
    X_val_dae_cat = [c[-val_len:] for c in X_all_cat_list]
    
    train_gen = DAEDataGenerator(X_train_dae_nb, X_train_dae_cat, cat_cols, batch_size=1024, p_swap=0.15)
    
    # Validation inputs and targets (Dictionaries)
    val_inputs = {'input_num_bin': X_val_dae_nb}
    val_targets = {'output_num_bin': X_val_dae_nb}
    
    for i, col in enumerate(cat_cols):
        val_inputs[f'input_{col}'] = X_val_dae_cat[i]
        val_targets[f'output_{col}'] = X_val_dae_cat[i]
        
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    dae_model.fit(
        train_gen,
        validation_data=(val_inputs, val_targets),
        epochs=30, 
        callbacks=[early_stopping],
        verbose=1
    )
    
    # 3. Extract DAE Features
    print("\nExtracting DAE Features...")
    train_dae_feats = get_dae_features(dae_model, X_train_num_bin, X_train_cat, cat_cols)
    test_dae_feats = get_dae_features(dae_model, X_test_num_bin, X_test_cat, cat_cols)
    
    print(f"DAE Features Shape: {train_dae_feats.shape}")
    
    # 4. Supervised Training
    
    # Model 2A: NN on DAE Features
    print("\nTraining Model 2A (NN on DAE)...")
    nn_preds_test = np.zeros(len(test_ids))
    nn_oof = np.zeros(len(y_train))
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_dae_feats, y_train)):
        print(f"Fold {fold+1}")
        X_tr, X_val = train_dae_feats[train_idx], train_dae_feats[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model_nn = build_nn_classifier(train_dae_feats.shape[1])
        
        # Class weights
        neg, pos = np.bincount(y_tr)
        total = neg + pos
        weight_for_0 = (1 / neg) * (total / 2.0)
        weight_for_1 = (1 / pos) * (total / 2.0)
        class_weight = {0: weight_for_0, 1: weight_for_1}
        
        es = callbacks.EarlyStopping(monitor='val_auc', mode='max', patience=10, restore_best_weights=True)
        
        model_nn.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=512,
            class_weight=class_weight,
            callbacks=[es],
            verbose=0
        )
        
        nn_oof[val_idx] = model_nn.predict(X_val).flatten()
        nn_preds_test += model_nn.predict(test_dae_feats).flatten() / 5
        
    print(f"NN OOF AUC: {roc_auc_score(y_train, nn_oof):.4f}")
    
    # Model 2B: LightGBM on DAE + Raw Features
    print("\nTraining Model 2B (LightGBM on DAE + Raw)...")
    
    # Combine features
    X_train_combined = np.hstack([X_train_full.values, train_dae_feats])
    X_test_combined = np.hstack([X_test_full.values, test_dae_feats])
    
    # Tune LightGBM
    best_lgb_params = tune_lightgbm(X_train_combined, y_train)
    best_lgb_params['random_state'] = SEED
    best_lgb_params['verbose'] = -1
    
    # Fix feature names for LightGBM
    orig_feat_names = list(X_train_full.columns)
    dae_feat_names = [f'dae_{i}' for i in range(train_dae_feats.shape[1])]
    feature_names = orig_feat_names + dae_feat_names
    # Sanitize names
    feature_names = [f.replace(' ', '_').replace(':', '_') for f in feature_names]
    
    lgb_preds_test = np.zeros(len(test_ids))
    lgb_oof = np.zeros(len(y_train))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_combined, y_train)):
        print(f"Fold {fold+1}")
        X_tr, X_val = X_train_combined[train_idx], X_train_combined[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        lgb_model = lgb.LGBMClassifier(**best_lgb_params)
        
        lgb_model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
        )
        
        lgb_oof[val_idx] = lgb_model.predict_proba(X_val)[:, 1]
        lgb_preds_test += lgb_model.predict_proba(X_test_combined)[:, 1] / 5
        
    print(f"LGB OOF AUC: {roc_auc_score(y_train, lgb_oof):.4f}")
    
    # Model 2C: CatBoost on DAE + Raw Features
    print("\nTraining Model 2C (CatBoost on DAE + Raw)...")
    
    cat_preds_test = np.zeros(len(test_ids))
    cat_oof = np.zeros(len(y_train))
    
    # CatBoost works well with default params usually, but let's set some
    cat_params = {
        'iterations': 1000,
        'learning_rate': 0.03,
        'depth': 6,
        'l2_leaf_reg': 3,
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'random_seed': SEED,
        'verbose': False,
        'early_stopping_rounds': 50,
        'auto_class_weights': 'Balanced'
    }
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_combined, y_train)):
        print(f"Fold {fold+1}")
        X_tr, X_val = X_train_combined[train_idx], X_train_combined[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        cat_model = CatBoostClassifier(**cat_params)
        
        cat_model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        cat_oof[val_idx] = cat_model.predict_proba(X_val)[:, 1]
        cat_preds_test += cat_model.predict_proba(X_test_combined)[:, 1] / 5
        
    print(f"CatBoost OOF AUC: {roc_auc_score(y_train, cat_oof):.4f}")
    
    # 5. Ensembling
    print("\nEnsembling...")
    # Weighted average (NN: 0.2, LGB: 0.4, Cat: 0.4)
    ensemble_oof = 0.2 * nn_oof + 0.4 * lgb_oof + 0.4 * cat_oof
    ensemble_test = 0.2 * nn_preds_test + 0.4 * lgb_preds_test + 0.4 * cat_preds_test
    
    print(f"Ensemble OOF AUC: {roc_auc_score(y_train, ensemble_oof):.4f}")
    
    # Find optimal threshold for F1
    prec, rec, thresholds = precision_recall_curve(y_train, ensemble_oof)
    f1_scores = 2 * (prec * rec) / (prec + rec + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    print(f"Best Threshold: {best_thresh:.4f}, Best F1: {best_f1:.4f}")
    
    # Apply to test
    # Or use the user's logic about expected positives (~2778)
    # The user previously mentioned 2778 positives.
    # Let's sort predictions and take top 2778
    target_positives = 2778
    sorted_preds = np.sort(ensemble_test)[::-1]
    if target_positives < len(sorted_preds):
        thresh_target = sorted_preds[target_positives-1]
        print(f"Threshold for {target_positives} positives: {thresh_target:.4f}")
    else:
        thresh_target = best_thresh
        
    final_preds = (ensemble_test >= thresh_target).astype(int)
    print(f"Predicted positives: {final_preds.sum()}")
    
    # Submission
    sub = pd.DataFrame({'claim_number': test_ids, 'subrogation': final_preds})
    sub.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")

if __name__ == '__main__':
    main()
