"""
Analyze top features to understand what drives subrogation success
This will help us create better domain-specific features
"""

import pandas as pd
import numpy as np
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, 'Data')

TRAIN_FILE = os.path.join(DATA_DIR, 'Training_TriGuard.csv')
TARGET_COL = 'subrogation'

print("Loading data...")
train_df = pd.read_csv(TRAIN_FILE)
y = train_df[TARGET_COL]
train_df = train_df.drop(columns=[TARGET_COL, 'claim_number'])

print(f"\nDataset shape: {train_df.shape}")
print(f"Target distribution: {y.value_counts().to_dict()}")
print(f"Target rate: {y.mean():.4f}")

print("\n" + "="*80)
print("ANALYZING FEATURES FOR SUBROGATION SUCCESS")
print("="*80)

# Analyze numeric features
numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
print(f"\n1. NUMERIC FEATURES ANALYSIS ({len(numeric_cols)} features)")
print("-"*80)

numeric_analysis = []
for col in numeric_cols:
    if train_df[col].nunique() > 1:
        subro_mean = train_df[y == 1][col].mean()
        no_subro_mean = train_df[y == 0][col].mean()
        diff = subro_mean - no_subro_mean
        diff_pct = (diff / (no_subro_mean + 1e-10)) * 100
        
        # Correlation with target
        corr = train_df[col].corr(y)
        
        numeric_analysis.append({
            'feature': col,
            'subro_mean': subro_mean,
            'no_subro_mean': no_subro_mean,
            'difference': diff,
            'diff_pct': diff_pct,
            'correlation': corr,
            'subro_median': train_df[y == 1][col].median(),
            'no_subro_median': train_df[y == 0][col].median()
        })

numeric_df = pd.DataFrame(numeric_analysis)
numeric_df = numeric_df.sort_values('correlation', key=abs, ascending=False)

print("\nTop 20 numeric features by absolute correlation with target:")
print(numeric_df.head(20)[['feature', 'correlation', 'subro_mean', 'no_subro_mean', 'difference', 'diff_pct']].to_string(index=False))

# Analyze categorical features
categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()
print(f"\n\n2. CATEGORICAL FEATURES ANALYSIS ({len(categorical_cols)} features)")
print("-"*80)

categorical_analysis = []
for col in categorical_cols:
    if train_df[col].nunique() > 1:
        value_counts = train_df[col].value_counts()
        subro_rates = train_df.groupby(col).apply(lambda x: y[x.index].mean())
        
        # Calculate weighted average subro rate difference
        overall_rate = y.mean()
        rate_diff = (subro_rates - overall_rate).abs().mean()
        
        # Most subro-prone category
        max_subro_cat = subro_rates.idxmax()
        max_subro_rate = subro_rates.max()
        min_subro_cat = subro_rates.idxmin()
        min_subro_rate = subro_rates.min()
        
        categorical_analysis.append({
            'feature': col,
            'n_unique': train_df[col].nunique(),
            'overall_rate': overall_rate,
            'max_subro_cat': max_subro_cat,
            'max_subro_rate': max_subro_rate,
            'min_subro_cat': min_subro_cat,
            'min_subro_rate': min_subro_rate,
            'rate_range': max_subro_rate - min_subro_rate,
            'avg_rate_diff': rate_diff
        })

categorical_df = pd.DataFrame(categorical_analysis)
categorical_df = categorical_df.sort_values('rate_range', ascending=False)

print("\nTop 15 categorical features by subrogation rate range:")
print(categorical_df.head(15)[['feature', 'n_unique', 'rate_range', 'max_subro_cat', 'max_subro_rate', 'min_subro_cat', 'min_subro_rate']].to_string(index=False))

# Analyze feature combinations
print(f"\n\n3. FEATURE COMBINATIONS ANALYSIS")
print("-"*80)

# Top numeric features
top_numeric = numeric_df.head(5)['feature'].tolist()
print(f"\nAnalyzing combinations of top 5 numeric features: {top_numeric}")

# Top categorical features  
top_categorical = categorical_df.head(3)['feature'].tolist()
print(f"Analyzing combinations with top 3 categorical features: {top_categorical}")

# Analyze key combinations
print("\nKey insights for feature engineering:")
print("\nA. High subrogation indicators (from numeric analysis):")
for idx, row in numeric_df.head(5).iterrows():
    if row['correlation'] > 0:
        print(f"   - Higher {row['feature']} → Higher subro chance (corr: {row['correlation']:.3f})")
    else:
        print(f"   - Lower {row['feature']} → Higher subro chance (corr: {row['correlation']:.3f})")

print("\nB. High subrogation categories (from categorical analysis):")
for idx, row in categorical_df.head(5).iterrows():
    print(f"   - {row['feature']}: '{row['max_subro_cat']}' has {row['max_subro_rate']:.3f} subro rate (vs {row['overall_rate']:.3f} overall)")

# Save analysis
output_file = os.path.join(SCRIPT_DIR, 'feature_analysis_results.csv')
numeric_df.to_csv(output_file.replace('.csv', '_numeric.csv'), index=False)
categorical_df.to_csv(output_file.replace('.csv', '_categorical.csv'), index=False)
print(f"\n\nAnalysis saved to:")
print(f"  - {output_file.replace('.csv', '_numeric.csv')}")
print(f"  - {output_file.replace('.csv', '_categorical.csv')}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)

