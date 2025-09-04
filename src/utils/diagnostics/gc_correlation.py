"""
GC Content Correlation Analysis

Tests if GC content correlates with weeks, BMI, Y_concentration.
If yes, filtering on GC introduces bias.
"""

import pandas as pd
import numpy as np
import re
from scipy.stats import pearsonr, spearmanr
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

def test_gc_correlations(data_file_path="src/data/data.xlsx", save_plots=True, verbose=True):
    """
    Test if GC content correlates with key variables.
    
    Args:
        data_file_path: Path to the Excel data file
        save_plots: Whether to save correlation plots
        verbose: Whether to print detailed results
    
    Returns:
        tuple: (correlation_df, analysis_data, bias_detected)
    """
    
    if verbose:
        print("🔬 Testing GC Content Correlations")
        print("=" * 40)
        print("If GC correlates with Y/weeks/BMI, then filtering on GC creates bias!")
    
    # Load raw data
    data_file = Path(data_file_path)
    df_raw = pd.read_excel(data_file, sheet_name='男胎检测数据')
    
    # Parse key variables
    df_raw['weeks'] = df_raw['检测孕周'].apply(parse_gestational_weeks)
    df_raw['BMI'] = pd.to_numeric(df_raw['孕妇BMI'], errors='coerce')
    df_raw['Y_concentration'] = pd.to_numeric(df_raw['Y染色体浓度'], errors='coerce')
    df_raw['GC_content'] = pd.to_numeric(df_raw['GC含量'], errors='coerce')
    
    # Keep only samples with complete data for analysis
    df_analysis = df_raw.dropna(subset=['Y_concentration', 'weeks', 'BMI', 'GC_content']).copy()
    
    # Apply minimal filters (just age) to get reasonable dataset
    df_analysis = df_analysis[(df_analysis['weeks'] >= 10) & (df_analysis['weeks'] <= 25)].copy()
    
    if verbose:
        print(f"📊 Analysis dataset: {len(df_analysis)} samples")
        print(f"   GC content range: {df_analysis['GC_content'].min():.4f} to {df_analysis['GC_content'].max():.4f}")
        normal_gc = ((df_analysis['GC_content'] >= 0.40) & (df_analysis['GC_content'] <= 0.60)).sum()
        outside_gc = ((df_analysis['GC_content'] < 0.40) | (df_analysis['GC_content'] > 0.60)).sum()
        print(f"   Normal range (40-60%): {normal_gc} samples")
        print(f"   Outside range: {outside_gc} samples")
    
    # Test correlations between GC and key variables
    variables = ['weeks', 'BMI', 'Y_concentration']
    
    if verbose:
        print(f"\n📊 GC Content Correlations:")
        print("-" * 40)
    
    correlations = []
    
    for var in variables:
        data_var = df_analysis[var].dropna()
        data_gc = df_analysis.loc[data_var.index, 'GC_content'].dropna()
        
        if len(data_var) > 10:  # Need sufficient sample size
            # Match indices
            common_idx = data_var.index.intersection(data_gc.index)
            var_vals = data_var.loc[common_idx]
            gc_vals = data_gc.loc[common_idx]
            
            # Correlations
            r_pearson, p_pearson = pearsonr(gc_vals, var_vals)
            r_spearman, p_spearman = spearmanr(gc_vals, var_vals)
            
            correlations.append({
                'Variable': var,
                'N': len(common_idx),
                'Pearson_r': r_pearson,
                'Pearson_p': p_pearson,
                'Spearman_r': r_spearman,
                'Spearman_p': p_spearman,
                'Significant': p_pearson < 0.05
            })
            
            if verbose:
                print(f"{var.upper()}:")
                print(f"  Pearson r = {r_pearson:.4f} (p = {p_pearson:.4f}) {'***' if p_pearson < 0.001 else '**' if p_pearson < 0.01 else '*' if p_pearson < 0.05 else 'ns'}")
                print(f"  Spearman r = {r_spearman:.4f} (p = {p_spearman:.4f})")
                print(f"  Sample size: {len(common_idx)}")
    
    # Create correlation summary
    corr_df = pd.DataFrame(correlations)
    
    # Visualize relationships if requested
    if save_plots:
        if verbose:
            print(f"\n📈 Creating correlation plots...")
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('GC Content vs Key Variables', fontsize=14)
        
        for i, var in enumerate(variables):
            ax = axes[i]
            
            var_data = df_analysis[var].dropna()
            gc_data = df_analysis.loc[var_data.index, 'GC_content'].dropna()
            common_idx = var_data.index.intersection(gc_data.index)
            
            if len(common_idx) > 0:
                x_vals = gc_data.loc[common_idx]
                y_vals = var_data.loc[common_idx]
                
                ax.scatter(x_vals, y_vals, alpha=0.6, s=20)
                
                # Add trend line
                z = np.polyfit(x_vals, y_vals, 1)
                p = np.poly1d(z)
                ax.plot(x_vals, p(x_vals), "r--", alpha=0.8, linewidth=1)
                
                # Mark GC filter boundaries
                ax.axvline(x=0.40, color='red', linestyle='--', alpha=0.5, label='Filter boundaries')
                ax.axvline(x=0.60, color='red', linestyle='--', alpha=0.5)
                
                ax.set_xlabel('GC Content')
                ax.set_ylabel(var.replace('_', ' ').title())
                ax.grid(True, alpha=0.3)
                
                # Add correlation info
                if i < len(correlations):
                    r_val = correlations[i]['Pearson_r']
                    p_val = correlations[i]['Pearson_p']
                    ax.set_title(f'r = {r_val:.3f} (p = {p_val:.4f})')
                
                if i == 0:
                    ax.legend()
        
        plt.tight_layout()
        plot_path = Path("output/figures/gc_correlations.png")
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        if verbose:
            print(f"   Saved plot to: {plot_path}")
    
    # Summary assessment
    significant_corrs = [c for c in correlations if c['Significant']]
    bias_detected = len(significant_corrs) > 0
    
    if verbose:
        print(f"\n🎯 Bias Assessment:")
        print("-" * 30)
        
        if bias_detected:
            print(f"⚠️  BIAS DETECTED!")
            print(f"   GC content significantly correlates with {len(significant_corrs)} key variable(s):")
            for corr in significant_corrs:
                direction = "positive" if corr['Pearson_r'] > 0 else "negative"
                strength = "strong" if abs(corr['Pearson_r']) > 0.5 else "moderate" if abs(corr['Pearson_r']) > 0.3 else "weak"
                print(f"     • {corr['Variable']}: {strength} {direction} correlation (r = {corr['Pearson_r']:.3f})")
            
            print(f"\n   🔍 Implications:")
            print(f"   • Filtering on GC content systematically removes samples")
            print(f"   • This introduces selection bias that can weaken observed correlations")
            print(f"   • Consider inverse probability weighting or relaxed GC thresholds")
            
        else:
            print(f"✅ NO BIAS DETECTED")
            print(f"   GC content doesn't significantly correlate with key variables")
            print(f"   Filtering on GC is unlikely to introduce substantial bias")
        
        # Practical recommendation
        outside_gc = ((df_analysis['GC_content'] < 0.40) | (df_analysis['GC_content'] > 0.60)).sum()
        total_samples = len(df_analysis)
        
        print(f"\n💡 Recommendation:")
        print(f"   • GC filter removes {outside_gc}/{total_samples} samples ({outside_gc/total_samples*100:.1f}%)")
        
        if bias_detected:
            print(f"   • Consider relaxing GC thresholds (e.g., 35-65%) to reduce bias")
            print(f"   • Or use inverse probability weighting in final models")
            print(f"   • Report both filtered and unfiltered results")
        else:
            print(f"   • Current GC filter (40-60%) appears reasonable")
            print(f"   • Weak correlations likely reflect true biology, not filter bias")
    
    return corr_df, df_analysis, bias_detected

if __name__ == "__main__":
    correlations, data, bias_detected = test_gc_correlations()
