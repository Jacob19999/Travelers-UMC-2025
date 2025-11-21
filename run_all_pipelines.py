"""
Master script to run all pipelines, extract metrics, generate comparison plots,
and save submissions to Output folder.
"""

import os
import sys
import subprocess
import re
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'Data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'Output')
PIPELINE_DIRS = {
    'Pipeline0': os.path.join(BASE_DIR, 'Pipeline0'),
    'Pipeline2': os.path.join(BASE_DIR, 'Pipeline2'),
    'Pipeline3': os.path.join(BASE_DIR, 'Pipeline3'),
    'Pipeline4': os.path.join(BASE_DIR, 'Pipeline4'),
}

# Create Output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'submissions'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'plots'), exist_ok=True)

# Results storage
results = {
    'pipeline': [],
    'auc': [],
    'f1': [],
    'threshold': [],
    'status': []
}

def extract_metrics_from_output(output_text):
    """Extract AUC and F1 from pipeline output text."""
    auc = None
    f1 = None
    threshold = None
    
    # Pattern for AUC
    auc_patterns = [
        r'AUC[:\s]+([0-9]+\.[0-9]+)',
        r'OOF AUC[:\s]+([0-9]+\.[0-9]+)',
        r'roc_auc[:\s]+([0-9]+\.[0-9]+)',
        r'Ensemble OOF AUC[:\s]+([0-9]+\.[0-9]+)',
    ]
    
    # Pattern for F1
    f1_patterns = [
        r'F1[:\s]+([0-9]+\.[0-9]+)',
        r'Best F1[:\s]+([0-9]+\.[0-9]+)',
        r'Best Ensemble F1[:\s]+([0-9]+\.[0-9]+)',
        r'f1_score[:\s]+([0-9]+\.[0-9]+)',
    ]
    
    # Pattern for threshold
    thresh_patterns = [
        r'Threshold[:\s]+([0-9]+\.[0-9]+)',
        r'Best Threshold[:\s]+([0-9]+\.[0-9]+)',
    ]
    
    for pattern in auc_patterns:
        match = re.search(pattern, output_text, re.IGNORECASE)
        if match:
            auc = float(match.group(1))
            break
    
    for pattern in f1_patterns:
        match = re.search(pattern, output_text, re.IGNORECASE)
        if match:
            f1 = float(match.group(1))
            break
    
    for pattern in thresh_patterns:
        match = re.search(pattern, output_text, re.IGNORECASE)
        if match:
            threshold = float(match.group(1))
            break
    
    return auc, f1, threshold

def run_pipeline(pipeline_name, script_path):
    """Run a pipeline script and capture metrics."""
    print(f"\n{'='*80}")
    print(f"Running {pipeline_name}")
    print(f"{'='*80}")
    
    try:
        # Change to pipeline directory
        pipeline_dir = os.path.dirname(script_path)
        script_file = os.path.basename(script_path)
        
        # Set environment for UTF-8 encoding (fixes Windows Unicode issues)
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        
        # Run the script
        result = subprocess.run(
            [sys.executable, script_file],
            cwd=pipeline_dir,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',  # Replace problematic characters instead of failing
            env=env,
            timeout=3600  # 1 hour timeout
        )
        
        output = result.stdout + result.stderr
        
        # Extract metrics
        auc, f1, threshold = extract_metrics_from_output(output)
        
        # Store results
        results['pipeline'].append(pipeline_name)
        results['auc'].append(auc if auc else np.nan)
        results['f1'].append(f1 if f1 else np.nan)
        results['threshold'].append(threshold if threshold else np.nan)
        results['status'].append('SUCCESS' if result.returncode == 0 else 'FAILED')
        
        # Print summary
        print(f"Status: {'SUCCESS' if result.returncode == 0 else 'FAILED'}")
        if auc:
            print(f"AUC: {auc:.4f}")
        if f1:
            print(f"F1: {f1:.4f}")
        if threshold:
            print(f"Threshold: {threshold:.4f}")
        
        if result.returncode != 0:
            print(f"Error output:\n{result.stderr[:500]}")
        
        return result.returncode == 0, output
        
    except subprocess.TimeoutExpired:
        print(f"ERROR: {pipeline_name} timed out after 1 hour")
        results['pipeline'].append(pipeline_name)
        results['auc'].append(np.nan)
        results['f1'].append(np.nan)
        results['threshold'].append(np.nan)
        results['status'].append('TIMEOUT')
        return False, ""
    except Exception as e:
        print(f"ERROR running {pipeline_name}: {str(e)}")
        results['pipeline'].append(pipeline_name)
        results['auc'].append(np.nan)
        results['f1'].append(np.nan)
        results['threshold'].append(np.nan)
        results['status'].append('ERROR')
        return False, ""

def copy_submissions(pipeline_name):
    """Copy submission files from Data directory to Output/submissions."""
    submission_files = []
    
    # Find all submission files for this pipeline
    for file in os.listdir(DATA_DIR):
        if file.startswith(f'pipeline{pipeline_name[-1]}_submission') and file.endswith('.csv'):
            src = os.path.join(DATA_DIR, file)
            dst = os.path.join(OUTPUT_DIR, 'submissions', file)
            try:
                shutil.copy2(src, dst)
                submission_files.append(file)
            except Exception as e:
                print(f"Warning: Could not copy {file}: {e}")
    
    return submission_files

def generate_comparison_plots():
    """Generate AUC and F1 comparison plots."""
    df = pd.DataFrame(results)
    
    # Filter out failed pipelines
    df_success = df[df['status'] == 'SUCCESS'].copy()
    
    if len(df_success) == 0:
        print("\nWARNING: No successful pipeline runs to plot!")
        return
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (14, 6)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: AUC Comparison
    ax1 = axes[0]
    bars1 = ax1.bar(df_success['pipeline'], df_success['auc'], 
                    color='steelblue', alpha=0.8, edgecolor='black')
    ax1.set_ylabel('AUC-ROC Score', fontsize=12, fontweight='bold')
    ax1.set_title('Pipeline AUC-ROC Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylim([0.75, max(df_success['auc'].max() * 1.1, 0.85)])
    ax1.grid(axis='y', alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        if not np.isnan(height):
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Plot 2: F1 Comparison
    ax2 = axes[1]
    bars2 = ax2.bar(df_success['pipeline'], df_success['f1'],
                    color='coral', alpha=0.8, edgecolor='black')
    ax2.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax2.set_title('Pipeline F1 Score Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylim([0.50, max(df_success['f1'].max() * 1.1, 0.65)])
    ax2.grid(axis='y', alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        if not np.isnan(height):
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(OUTPUT_DIR, 'plots', 'pipeline_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved comparison plot to: {plot_path}")
    plt.close()
    
    # Create detailed metrics table
    fig2, ax3 = plt.subplots(figsize=(10, 6))
    ax3.axis('tight')
    ax3.axis('off')
    
    # Prepare table data
    table_data = []
    for idx, row in df.iterrows():
        table_data.append([
            row['pipeline'],
            f"{row['auc']:.4f}" if not np.isnan(row['auc']) else "N/A",
            f"{row['f1']:.4f}" if not np.isnan(row['f1']) else "N/A",
            f"{row['threshold']:.4f}" if not np.isnan(row['threshold']) else "N/A",
            row['status']
        ])
    
    table = ax3.table(cellText=table_data,
                     colLabels=['Pipeline', 'AUC', 'F1', 'Threshold', 'Status'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.25, 0.15, 0.15, 0.15, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # Style header
    for i in range(5):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style rows based on status
    for i, row in enumerate(table_data, start=1):
        if row[4] == 'SUCCESS':
            for j in range(5):
                table[(i, j)].set_facecolor('#E8F5E9')
        else:
            for j in range(5):
                table[(i, j)].set_facecolor('#FFEBEE')
    
    plt.title('Pipeline Performance Summary', fontsize=14, fontweight='bold', pad=20)
    
    table_path = os.path.join(OUTPUT_DIR, 'plots', 'pipeline_metrics_table.png')
    plt.savefig(table_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved metrics table to: {table_path}")
    plt.close()

def save_results_csv():
    """Save results to CSV."""
    df = pd.DataFrame(results)
    csv_path = os.path.join(OUTPUT_DIR, 'pipeline_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved results CSV to: {csv_path}")

def main():
    """Main execution function."""
    print("="*80)
    print("MASTER PIPELINE RUNNER")
    print("="*80)
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nNote: Pipeline1 is notebook-based and must be run manually.")
    print("="*80)
    
    # Pipeline scripts to run
    # Note: Pipeline1 is notebook-based and not included in automated runs
    pipelines = {
        'Pipeline0': os.path.join(PIPELINE_DIRS['Pipeline0'], 'target_encoding_advanced_features.py'),
        'Pipeline2': os.path.join(PIPELINE_DIRS['Pipeline2'], 'pipeline2_dae.py'),
        'Pipeline3': os.path.join(PIPELINE_DIRS['Pipeline3'], 'pipeline3_stacking.py'),
        'Pipeline4': os.path.join(PIPELINE_DIRS['Pipeline4'], 'pipeline4_actuarial.py'),
    }
    
    # Run each pipeline
    for pipeline_name, script_path in pipelines.items():
        if not os.path.exists(script_path):
            print(f"\nWARNING: {script_path} not found, skipping {pipeline_name}")
            results['pipeline'].append(pipeline_name)
            results['auc'].append(np.nan)
            results['f1'].append(np.nan)
            results['threshold'].append(np.nan)
            results['status'].append('NOT_FOUND')
            continue
        
        success, output = run_pipeline(pipeline_name, script_path)
        
        # Copy submissions
        if success:
            submissions = copy_submissions(pipeline_name)
            if submissions:
                print(f"✓ Copied {len(submissions)} submission file(s) to Output/submissions/")
    
    # Generate plots and save results
    print(f"\n{'='*80}")
    print("GENERATING COMPARISON PLOTS")
    print(f"{'='*80}")
    generate_comparison_plots()
    save_results_csv()
    
    # Final summary
    print(f"\n{'='*80}")
    print("EXECUTION SUMMARY")
    print(f"{'='*80}")
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    print(f"\n{'='*80}")
    print(f"All outputs saved to: {OUTPUT_DIR}")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()

