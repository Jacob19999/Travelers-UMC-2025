"""
Pipeline 2 - Preprocessing & RankGauss Normalization (patched version)

This script now:
- Removes target + ID columns before transformations
- Keeps binary numeric features untouched
- Applies RankGauss normalization ONLY to real-valued numeric features
- Encodes categoricals as integer indices (for embeddings / DAE)
- Returns y (target) separately and the categorical encoder

Conforms to:
Pipeline 2 Step 0 (à la Porto Seguro):
- Raw numeric → RankGauss
- Binary untouched
- Categorical → integer indices (for embeddings)
"""

import importlib.util
import sys
import subprocess
import os

REQUIRED_PACKAGES = {
    'pandas': 'pandas',
    'numpy': 'numpy',
    'sklearn': 'scikit-learn',
    'xgboost': 'xgboost',
    'lightgbm': 'lightgbm',
    'scipy': 'scipy',
    'tensorflow': 'tensorflow'
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, 'Data')

TRAIN_FILE = os.path.join(DATA_DIR, 'Training_TriGuard.csv')
TEST_FILE = os.path.join(DATA_DIR, 'Testing_TriGuard.csv')

TARGET_COL = "subrogation"
ID_COLS = ["claim_number"]

EXCLUDE_FROM_CATEGORICAL = [
    'claim_number',
    'claim_date'
]


# -----------------------------------------------------------
# Package checking
# -----------------------------------------------------------

def check_package_installed(package_name, import_name=None):
    if import_name is None:
        import_name = package_name
    spec = importlib.util.find_spec(import_name)
    return spec is not None


def check_required_packages(required_packages=REQUIRED_PACKAGES, verbose=True, auto_install=False):
    if verbose:
        print("=" * 80)
        print("Checking required packages for Pipeline 2...")
        print("=" * 80)

    missing_packages = []
    installed_packages = []

    for import_name, pip_name in required_packages.items():
        is_installed = check_package_installed(import_name)
        if is_installed:
            installed_packages.append((import_name, pip_name))
            if verbose:
                print(f"  [OK] {pip_name} is installed")
        else:
            missing_packages.append((import_name, pip_name))
            if verbose:
                print(f"  [MISSING] {pip_name} is NOT installed")

    if missing_packages:
        print("\nMissing packages:")
        print("  pip install " + " ".join([p for _, p in missing_packages]))
        if auto_install:
            for _, pip_name in missing_packages:
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", pip_name])
                    print(f"[OK] installed {pip_name}")
                except Exception:
                    print(f"[ERROR] failed to install {pip_name}")
                    return False
            return True
        return False

    if verbose:
        print("[OK] All required packages installed.")
    return True


# -----------------------------------------------------------
# Data loading
# -----------------------------------------------------------

def load_data(train_file=TRAIN_FILE, test_file=TEST_FILE, verbose=True):
    import pandas as pd

    if verbose:
        print("\n[Step 0.1] Loading data...")

    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training file not found: {train_file}")

    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test file not found: {test_file}")

    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    if verbose:
        print(f"Train shape: {train_df.shape}")
        print(f"Test shape: {test_df.shape}")

    return train_df, test_df


def parse_datetime(train_df, test_df, date_column='claim_date', date_format='%m/%d/%Y', verbose=True):
    import pandas as pd

    if verbose:
        print(f"[Step 0.1b] Parsing {date_column} to datetime...")

    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df[date_column] = pd.to_datetime(train_df[date_column], format=date_format)
    test_df[date_column] = pd.to_datetime(test_df[date_column], format=date_format)

    return train_df, test_df


# -----------------------------------------------------------
# Categorical → Integer Indices (Ordinal Encoding)
# -----------------------------------------------------------

def step0_1_preprocessing(train_file=TRAIN_FILE, test_file=TEST_FILE,
                          handle_unknown='use_encoded_value', verbose=True):

    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import OrdinalEncoder

    if verbose:
        print("=" * 80)
        print("Pipeline 2 - Step 0.1: Data Loading + Categorical Ordinal Encoding")
        print("=" * 80)

    # Load
    train_df, test_df = load_data(train_file, test_file, verbose=verbose)
    train_df, test_df = parse_datetime(train_df, test_df, verbose=verbose)

    # -----------------------------------------
    # Remove target + ID columns BEFORE encoding
    # -----------------------------------------
    y = train_df[TARGET_COL].copy()

    drop_cols = [TARGET_COL] + ID_COLS
    train_df = train_df.drop(columns=drop_cols, errors="ignore")
    test_df = test_df.drop(columns=ID_COLS, errors="ignore")

    # Identify categoricals
    cat_cols = train_df.select_dtypes(include=['object', 'category']).columns.tolist()
    categorical_columns = [c for c in cat_cols if c not in EXCLUDE_FROM_CATEGORICAL]

    # Identify numeric (true numeric features; encoded categoricals will be added later)
    numeric_columns = train_df.select_dtypes(include=['number']).columns.tolist()

    if verbose:
        print(f"Categorical columns (to encode as integers): {len(categorical_columns)}")
        print(f"Numeric columns (raw): {len(numeric_columns)}")

    # Ordinal encode categoricals → integer indices
    train_encoded = train_df.copy()
    test_encoded = test_df.copy()

    if len(categorical_columns) > 0:
        encoder = OrdinalEncoder(
            handle_unknown=handle_unknown,
            unknown_value=-1
        )

        # Work on copies with string dtype + explicit missing token
        train_cat = train_df[categorical_columns].astype("object").fillna("__MISSING__")
        test_cat = test_df[categorical_columns].astype("object").fillna("__MISSING__")

        encoder.fit(train_cat)

        train_cat_enc = encoder.transform(train_cat)
        test_cat_enc = encoder.transform(test_cat)

        # Assign back as integer-coded columns
        for i, col in enumerate(categorical_columns):
            train_encoded[col] = train_cat_enc[:, i].astype("int64")
            test_encoded[col] = test_cat_enc[:, i].astype("int64")

        if verbose:
            print("Categorical columns encoded as integer indices using OrdinalEncoder.")
    else:
        encoder = None
        if verbose:
            print("No categorical columns detected; skipping OrdinalEncoder.")

    return train_encoded, test_encoded, categorical_columns, numeric_columns, encoder, y


# -----------------------------------------------------------
# RankGauss Normalization
# -----------------------------------------------------------

def rankgauss_normalize(train_df, test_df, numeric_columns=None, verbose=True):
    import numpy as np
    from scipy.special import erfinv
    import pandas as pd

    train_df_norm = train_df.copy()
    test_df_norm = test_df.copy()

    # Auto-detect numeric columns if not provided
    if numeric_columns is None:
        numeric_columns = train_df.select_dtypes(include=[np.number]).columns.tolist()

    # We assume numeric_columns refers ONLY to "true" numeric features,
    # not to the integer-coded categoricals (they should be excluded upstream).

    # Exclude binary numeric columns (0/1 only)
    binary_cols = []
    real_numeric_cols = []

    for col in numeric_columns:
        vals = set(train_df[col].dropna().unique())
        if vals.issubset({0, 1}):
            binary_cols.append(col)
        else:
            real_numeric_cols.append(col)

    if verbose:
        print(f"\nApplying RankGauss to {len(real_numeric_cols)} real numeric features")
        print(f"Binary features kept as-is: {binary_cols}")

    for col in real_numeric_cols:
        train_vals = train_df[col].dropna().astype(float)
        n = len(train_vals)
        if n == 0:
            continue

        sorted_idx = np.argsort(train_vals.values)
        ranks = np.empty_like(sorted_idx)
        ranks[sorted_idx] = np.arange(1, n + 1)

        u = ranks / (n + 1)
        x = np.sqrt(2) * erfinv(np.clip(2 * u - 1, -1 + 1e-12, 1 - 1e-12))
        mu = x.mean()
        x -= mu

        mapping = dict(zip(train_vals.values, x))
        train_df_norm[col] = train_df[col].map(mapping)

        test_series = test_df[col].copy().astype(float)
        mask = ~test_series.isna()

        if mask.any():
            tvals = test_series[mask].values
            train_sorted = np.sort(train_vals.values)
            ranks_test = np.searchsorted(train_sorted, tvals, side='right')
            u_test = ranks_test / (n + 1)
            x_test = np.sqrt(2) * erfinv(np.clip(2 * u_test - 1, -1 + 1e-12, 1 - 1e-12))
            x_test -= mu
            test_series.loc[mask] = x_test

        test_df_norm[col] = test_series

        if verbose:
            print(f"  - {col}: RankGauss normalized")

    return train_df_norm, test_df_norm, real_numeric_cols


# -----------------------------------------------------------
# Full Step 0
# -----------------------------------------------------------

def step0_preprocessing(train_file=TRAIN_FILE, test_file=TEST_FILE, verbose=True):
    if verbose:
        print("=" * 80)
        print("Pipeline 2 - Step 0: Full Preprocessing + RankGauss")
        print("=" * 80)

    (
        train_encoded,
        test_encoded,
        categorical_columns,
        numeric_columns,
        cat_encoder,
        y
    ) = step0_1_preprocessing(train_file, test_file, verbose=verbose)

    # IMPORTANT:
    # Pass ONLY the original numeric_columns into RankGauss,
    # so categorical integer indices are NOT normalized.
    train_norm, test_norm, normalized_numeric = rankgauss_normalize(
        train_encoded,
        test_encoded,
        numeric_columns=numeric_columns,
        verbose=verbose
    )

    return train_norm, test_norm, y, categorical_columns, normalized_numeric, cat_encoder


# -----------------------------------------------------------
# Step 1 - Denoising Autoencoder
# -----------------------------------------------------------

def construct_input_matrix(train_df, test_df, cat_cols, num_cols, verbose=True):
    """Concatenate RankGauss numeric + binary + categorical integer indices."""
    import numpy as np
    
    # Identify binary columns (0/1 only)
    binary_cols = []
    real_numeric_cols = []
    for col in num_cols:
        vals = set(train_df[col].dropna().unique())
        if vals.issubset({0, 1}):
            binary_cols.append(col)
        else:
            real_numeric_cols.append(col)
    
    # Build feature list: real numeric (RankGauss) + binary + categorical indices
    feature_cols = real_numeric_cols + binary_cols + cat_cols
    
    X_train = train_df[feature_cols].values.astype(np.float32)
    X_test = test_df[feature_cols].values.astype(np.float32)
    
    # Fill NaN with 0 (should be rare after RankGauss, but handle for safety)
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)
    
    if verbose:
        print(f"Input matrix: {len(real_numeric_cols)} RankGauss + {len(binary_cols)} binary + {len(cat_cols)} categorical = {X_train.shape[1]} features")
    
    return X_train, X_test, feature_cols


def apply_swap_noise(X, p_swap=0.15, random_state=None):
    """Apply swap noise: replace values with random values from same column."""
    import numpy as np
    
    if random_state is not None:
        np.random.seed(random_state)
    
    X_noisy = X.copy()
    n_samples, n_features = X.shape
    
    # Vectorized swap noise: for each (sample, feature), swap with probability p_swap
    swap_mask = np.random.rand(n_samples, n_features) < p_swap
    if swap_mask.any():
        # Random row indices for swapping (per position)
        swap_indices = np.random.randint(0, n_samples, size=(n_samples, n_features))
        # For each swapped position (i,j), replace with X[swap_indices[i,j], j]
        row_idx, col_idx = np.where(swap_mask)
        X_noisy[row_idx, col_idx] = X[swap_indices[row_idx, col_idx], col_idx]
    
    return X_noisy


def build_dae(input_dim, bottleneck_dim=128, verbose=True):
    """Build Denoising Autoencoder architecture."""
    from tensorflow import keras
    from tensorflow.keras import layers
    
    # Encoder
    encoder = keras.Sequential([
        layers.Dense(1000, activation='relu', input_shape=(input_dim,)),
        layers.Dense(500, activation='relu'),
        layers.Dense(bottleneck_dim, activation='relu', name='bottleneck')
    ], name='encoder')
    
    # Decoder
    decoder = keras.Sequential([
        layers.Dense(500, activation='relu', input_shape=(bottleneck_dim,)),
        layers.Dense(1000, activation='relu'),
        layers.Dense(input_dim, activation='linear', name='output')
    ], name='decoder')
    
    # Full DAE
    inputs = layers.Input(shape=(input_dim,))
    encoded = encoder(inputs)
    decoded = decoder(encoded)
    dae = keras.Model(inputs, decoded, name='dae')
    
    if verbose:
        dae.summary()
    
    return dae, encoder, decoder


def train_dae(X_train, X_val=None, p_swap=0.15, bottleneck_dim=128, 
              batch_size=2048, epochs=30, lr=1e-3, verbose=True, 
              validation_split=0.1, early_stopping_patience=5):
    """Train Denoising Autoencoder with swap noise."""
    import numpy as np
    from tensorflow import keras
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    
    input_dim = X_train.shape[1]
    
    # Split validation if not provided
    if X_val is None:
        n_val = int(len(X_train) * validation_split)
        indices = np.random.permutation(len(X_train))
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
        X_val = X_train[val_indices]
        X_train_split = X_train[train_indices]
    else:
        X_train_split = X_train
    
    # Build model
    dae, encoder, decoder = build_dae(input_dim, bottleneck_dim, verbose=False)
    dae.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss='mse')
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=early_stopping_patience, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5)
    ]
    
    # Training data generator with swap noise
    class SwapNoiseGenerator(keras.utils.Sequence):
        def __init__(self, X, batch_size, p_swap):
            self.X = X
            self.batch_size = batch_size
            self.p_swap = p_swap
        
        def __len__(self):
            return int(np.ceil(len(self.X) / self.batch_size))
        
        def __getitem__(self, idx):
            batch = self.X[idx * self.batch_size:(idx + 1) * self.batch_size]
            batch_noisy = apply_swap_noise(batch, p_swap=self.p_swap)
            return batch_noisy, batch
    
    train_gen = SwapNoiseGenerator(X_train_split, batch_size, p_swap)
    val_gen = SwapNoiseGenerator(X_val, batch_size, p_swap)
    
    # Train
    if verbose:
        print(f"Training DAE: input_dim={input_dim}, bottleneck={bottleneck_dim}, p_swap={p_swap}")
    
    history = dae.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1 if verbose else 0
    )
    
    return dae, encoder, decoder, history


def extract_bottleneck_features(encoder, X_train, X_test, verbose=True):
    """Extract bottleneck activations from trained encoder."""
    import numpy as np
    
    train_features = encoder.predict(X_train, verbose=0)
    test_features = encoder.predict(X_test, verbose=0)
    
    if verbose:
        print(f"Bottleneck features: train={train_features.shape}, test={test_features.shape}")
    
    return train_features, test_features


def step1_dae(train_df, test_df, cat_cols, num_cols, p_swap=0.15, 
              bottleneck_dim=128, batch_size=2048, epochs=30, 
              validation_split=0.1, verbose=True):
    """Step 1: Build and train Denoising Autoencoder."""
    if verbose:
        print("=" * 80)
        print("Pipeline 2 - Step 1: Denoising Autoencoder")
        print("=" * 80)
    
    # Construct input matrix
    X_train, X_test, feature_cols = construct_input_matrix(
        train_df, test_df, cat_cols, num_cols, verbose=verbose
    )
    
    # Train DAE
    dae, encoder, decoder, history = train_dae(
        X_train, X_val=None, p_swap=p_swap, bottleneck_dim=bottleneck_dim,
        batch_size=batch_size, epochs=epochs, validation_split=validation_split,
        verbose=verbose
    )
    
    # Extract bottleneck features
    train_bottleneck, test_bottleneck = extract_bottleneck_features(
        encoder, X_train, X_test, verbose=verbose
    )
    
    return dae, encoder, decoder, train_bottleneck, test_bottleneck, feature_cols, history


def main():
    ok = check_required_packages(verbose=True, auto_install=True)
    if not ok:
        print("[ERROR] Missing packages")
        sys.exit(1)

    train_df_norm, test_df_norm, y, cat_cols, num_cols, cat_encoder = step0_preprocessing()

    print("\nStep 0 completed.")
    print("Shapes:")
    print(f"  X_train: {train_df_norm.shape}")
    print(f"  X_test:  {test_df_norm.shape}")
    print(f"  y:       {y.shape}")
    
    # Step 1: Train DAE
    dae, encoder, decoder, train_bottleneck, test_bottleneck, feature_cols, history = step1_dae(
        train_df_norm, test_df_norm, cat_cols, num_cols, 
        p_swap=0.15, bottleneck_dim=128, epochs=30, verbose=True
    )
    
    print("\nStep 1 completed.")
    print(f"DAE bottleneck features: train={train_bottleneck.shape}, test={test_bottleneck.shape}")

    return train_df_norm, test_df_norm, y, cat_cols, num_cols, cat_encoder, \
           dae, encoder, decoder, train_bottleneck, test_bottleneck


if __name__ == "__main__":
    main()
