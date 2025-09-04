"""
Selection Bias Analysis

Compares characteristics of kept vs removed samples to detect systematic differences
that could introduce bias in statistical relationships.
"""

import pandas as pd
import numpy as np
import re
from scipy.stats import ttest_ind, mannwhitneyu
from pathlib import Path
import matplotlib.pyplot as plt

def parse_gestational_weeks(week_str):
    """Convert gestational week format to decimal weeks."""
    if pd.isna(week_str):
        return np.nan
    
    week_str = str(week_str).strip()
    pattern = r'(\d+)w(?:\+(\d+))?'
    match = re.search(pattern, week_str)
    
    if match:
        weeks = int(match.group(1))
        days = int(match.group(2)) if match.group(2) else 0
        return weeks + days / 7.0
    else:
        try:
            return float(week_str)
        except:
            return np.nan

def test_selection_bias(data_file_path="src/data/data.xlsx", save_plots=True, verbose=True):
    """
    Test if removed samples differ systematically from kept samples.
    
    Args:
        data_file_path: Path to the Excel data file
        save_plots: Whether to save comparison plots
        verbose: Whether to print detailed results
    
    Returns:
        tuple: (original_data, final_data)
    """
    
    if verbose:
        print("🔬 Testing Selection Bias from Filtering")
        print("=" * 50)
    
    # Load raw data
    data_file = Path(data_file_path)
    df_raw = pd.read_excel(data_file, sheet_name='男胎检测数据')
    
    # Parse key variables
    df_raw['weeks'] = df_raw['检测孕周'].apply(parse_gestational_weeks)
    df_raw['BMI'] = pd.to_numeric(df_raw['孕妇BMI'], errors='coerce')
    df_raw['Y_concentration'] = pd.to_numeric(df_raw['Y染色体浓度'], errors='coerce')
    df_raw['GC_content'] = pd.to_numeric(df_raw['GC含量'], errors='coerce')
    
    # Remove samples with missing target variable (can't analyze without Y)
    df_analysis = df_raw.dropna(subset=['Y_concentration', 'weeks', 'BMI', 'GC_content']).copy()
    
    if verbose:
        print(f"📊 Analysis dataset: {len(df_analysis)} samples with complete data")
    
    # Apply filters step by step and track what gets removed
    
    # 1. Gestational age filter
    age_kept = (df_analysis['weeks'] >= 10) & (df_analysis['weeks'] <= 25)
    age_removed = ~age_kept
    
    if verbose:
        print(f"\n1️⃣ Gestational Age Filter (10-25 weeks):")
        print(f"   Kept: {age_kept.sum()} samples")
        print(f"   Removed: {age_removed.sum()} samples")
        
        if age_removed.sum() > 0:
            print("   Characteristics of REMOVED samples:")
            removed_age = df_analysis[age_removed]
            print(f"     Mean weeks: {removed_age['weeks'].mean():.2f} ± {removed_age['weeks'].std():.2f}")
            print(f"     Mean Y conc: {removed_age['Y_concentration'].mean():.4f} ± {removed_age['Y_concentration'].std():.4f}")
            print(f"     Mean BMI: {removed_age['BMI'].mean():.2f} ± {removed_age['BMI'].std():.2f}")
    
    # Apply age filter for next step
    df_after_age = df_analysis[age_kept].copy()
    
    # 2. GC content filter (THE BIG ONE)
    gc_kept = (df_after_age['GC_content'] >= 0.40) & (df_after_age['GC_content'] <= 0.60)
    gc_removed = ~gc_kept
    
    if verbose:
        print(f"\n2️⃣ GC Content Filter (40-60%):")
        print(f"   Kept: {gc_kept.sum()} samples")
        print(f"   Removed: {gc_removed.sum()} samples ({gc_removed.sum()/len(df_after_age)*100:.1f}%)")
    
    bias_detected = False
    
    if gc_removed.sum() > 0:
        kept_gc = df_after_age[gc_kept]
        removed_gc = df_after_age[gc_removed]
        
        if verbose:
            print("\n   📊 Comparison of KEPT vs REMOVED by GC filter:")
            print("   " + "="*50)
        
        variables = ['weeks', 'BMI', 'Y_concentration', 'GC_content']
        comparison_results = []
        
        for var in variables:
            kept_vals = kept_gc[var].dropna()
            removed_vals = removed_gc[var].dropna()
            
            if len(kept_vals) > 0 and len(removed_vals) > 0:
                # T-test for means
                t_stat, t_pval = ttest_ind(kept_vals, removed_vals)
                
                # Mann-Whitney U test for distributions
                u_stat, u_pval = mannwhitneyu(kept_vals, removed_vals, alternative='two-sided')
                
                comparison_results.append({
                    'variable': var,
                    'kept_mean': kept_vals.mean(),
                    'kept_std': kept_vals.std(),
                    'kept_n': len(kept_vals),
                    'removed_mean': removed_vals.mean(),
                    'removed_std': removed_vals.std(),
                    'removed_n': len(removed_vals),
                    'difference': kept_vals.mean() - removed_vals.mean(),
                    't_pval': t_pval,
                    'u_pval': u_pval
                })
                
                if t_pval < 0.05 and var in ['weeks', 'BMI', 'Y_concentration']:
                    bias_detected = True
                
                if verbose:
                    print(f"\n   {var.upper()}:")
                    print(f"     Kept: {kept_vals.mean():.4f} ± {kept_vals.std():.4f} (n={len(kept_vals)})")
                    print(f"     Removed: {removed_vals.mean():.4f} ± {removed_vals.std():.4f} (n={len(removed_vals)})")
                    print(f"     Difference: {kept_vals.mean() - removed_vals.mean():+.4f}")
                    print(f"     T-test p-value: {t_pval:.4f} {'***' if t_pval < 0.001 else '**' if t_pval < 0.01 else '*' if t_pval < 0.05 else 'ns'}")
                    print(f"     Mann-Whitney p-value: {u_pval:.4f}")
        
        # Create visualization if requested
        if save_plots:
            if verbose:
                print(f"\n   📈 Creating comparison plots...")
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('Distribution Comparison: Kept vs Removed by GC Filter', fontsize=14)
            
            for i, var in enumerate(['weeks', 'BMI', 'Y_concentration', 'GC_content']):
                ax = axes[i//2, i%2]
                
                kept_vals = kept_gc[var].dropna()
                removed_vals = removed_gc[var].dropna()
                
                ax.hist(kept_vals, alpha=0.6, label=f'Kept (n={len(kept_vals)})', bins=20, color='blue')
                ax.hist(removed_vals, alpha=0.6, label=f'Removed (n={len(removed_vals)})', bins=20, color='red')
                ax.set_xlabel(var.replace('_', ' ').title())
                ax.set_ylabel('Frequency')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = Path("output/figures/selection_bias_comparison.png")
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            if verbose:
                print(f"   Saved plot to: {plot_path}")
    
    # 3. Chromosomal abnormality filter
    df_after_gc = df_after_age[gc_kept].copy()
    aneu_kept = df_after_gc['染色体的非整倍体'].isna()
    aneu_removed = ~aneu_kept
    
    if verbose:
        print(f"\n3️⃣ Chromosomal Abnormality Filter:")
        print(f"   Kept: {aneu_kept.sum()} samples")
        print(f"   Removed: {aneu_removed.sum()} samples")
    
    # Summary of total filtering impact
    final_n = aneu_kept.sum()
    initial_n = len(df_analysis)
    
    if verbose:
        print(f"\n🎯 Overall Filtering Impact:")
        print(f"   Initial samples: {initial_n}")
        print(f"   Final samples: {final_n}")
        print(f"   Total removed: {initial_n - final_n} ({(initial_n - final_n)/initial_n*100:.1f}%)")
        print(f"   Retention rate: {final_n/initial_n*100:.1f}%")
    
        print(f"\n⚠️  GC Filter Selection Bias: {'DETECTED' if bias_detected else 'NOT DETECTED'}")
        if bias_detected:
            print("   The GC filter systematically removes samples with different characteristics!")
            print("   This could explain weakened correlations.")
        else:
            print("   The GC filter removes samples randomly with respect to key variables.")
    
    return df_analysis, df_after_age[gc_kept], bias_detected

if __name__ == "__main__":
    original_data, final_data, bias_detected = test_selection_bias()
