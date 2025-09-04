"""
Filter Impact Analysis

Tests how correlations change at each filtering stage to identify which filters 
cause signal loss or bias.
"""

import pandas as pd
import numpy as np
import re
from scipy.stats import pearsonr, norm
from pathlib import Path

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

def fisher_z_test(r1, n1, r2, n2):
    """Test if two correlation coefficients differ significantly using Fisher z-transform."""
    # Fisher z-transform
    z1 = 0.5 * np.log((1 + r1) / (1 - r1))
    z2 = 0.5 * np.log((1 + r2) / (1 - r2))
    
    # Standard error of difference
    se_diff = np.sqrt(1/(n1-3) + 1/(n2-3))
    
    # Test statistic
    z_stat = (z1 - z2) / se_diff
    
    # Two-tailed p-value
    p_value = 2 * (1 - norm.cdf(abs(z_stat)))
    
    return z_stat, p_value

def test_filter_impact(data_file_path="src/data/data.xlsx", verbose=True):
    """
    Test correlations at each filtering stage.
    
    Args:
        data_file_path: Path to the Excel data file
        verbose: Whether to print detailed results
    
    Returns:
        tuple: (results_df, raw_data, clean_data)
    """
    
    if verbose:
        print("🔬 Testing Filter Impact on Correlations")
        print("=" * 60)
    
    # Load raw data
    data_file = Path(data_file_path)
    df_raw = pd.read_excel(data_file, sheet_name='男胎检测数据')
    
    # Parse key variables
    df_raw['weeks'] = df_raw['检测孕周'].apply(parse_gestational_weeks)
    df_raw['BMI'] = pd.to_numeric(df_raw['孕妇BMI'], errors='coerce')
    df_raw['Y_concentration'] = pd.to_numeric(df_raw['Y染色体浓度'], errors='coerce')
    df_raw['GC_content'] = pd.to_numeric(df_raw['GC含量'], errors='coerce')
    
    # Remove samples with missing target variable (can't analyze without Y)
    df_raw = df_raw.dropna(subset=['Y_concentration'])
    
    results = []
    
    # Stage 0: Raw data (before filters)
    stage0 = df_raw.dropna(subset=['weeks', 'BMI', 'Y_concentration']).copy()
    if len(stage0) > 0:
        r_weeks_0, p_weeks_0 = pearsonr(stage0['weeks'], stage0['Y_concentration'])
        r_bmi_0, p_bmi_0 = pearsonr(stage0['BMI'], stage0['Y_concentration'])
        results.append({
            'Stage': 'Raw Data',
            'N': len(stage0),
            'r_weeks': r_weeks_0,
            'p_weeks': p_weeks_0,
            'r_bmi': r_bmi_0,
            'p_bmi': p_bmi_0,
            'Filter': 'None'
        })
    
    # Stage 1: Gestational age filter (10-25 weeks)
    stage1 = stage0[(stage0['weeks'] >= 10) & (stage0['weeks'] <= 25)].copy()
    if len(stage1) > 0:
        r_weeks_1, p_weeks_1 = pearsonr(stage1['weeks'], stage1['Y_concentration'])
        r_bmi_1, p_bmi_1 = pearsonr(stage1['BMI'], stage1['Y_concentration'])
        results.append({
            'Stage': 'After Age Filter',
            'N': len(stage1),
            'r_weeks': r_weeks_1,
            'p_weeks': p_weeks_1,
            'r_bmi': r_bmi_1,
            'p_bmi': p_bmi_1,
            'Filter': 'Weeks 10-25'
        })
    
    # Stage 2: GC content filter (40-60%) - THE SUSPICIOUS ONE
    stage2 = stage1[(stage1['GC_content'] >= 0.40) & (stage1['GC_content'] <= 0.60)].copy()
    if len(stage2) > 0:
        r_weeks_2, p_weeks_2 = pearsonr(stage2['weeks'], stage2['Y_concentration'])
        r_bmi_2, p_bmi_2 = pearsonr(stage2['BMI'], stage2['Y_concentration'])
        results.append({
            'Stage': 'After GC Filter',
            'N': len(stage2),
            'r_weeks': r_weeks_2,
            'p_weeks': p_weeks_2,
            'r_bmi': r_bmi_2,
            'p_bmi': p_bmi_2,
            'Filter': 'GC 40-60%'
        })
    
    # Stage 3: Chromosomal abnormality filter
    stage3 = stage2[stage2['染色体的非整倍体'].isna()].copy()
    if len(stage3) > 0:
        r_weeks_3, p_weeks_3 = pearsonr(stage3['weeks'], stage3['Y_concentration'])
        r_bmi_3, p_bmi_3 = pearsonr(stage3['BMI'], stage3['Y_concentration'])
        results.append({
            'Stage': 'Final Clean',
            'N': len(stage3),
            'r_weeks': r_weeks_3,
            'p_weeks': p_weeks_3,
            'r_bmi': r_bmi_3,
            'p_bmi': p_bmi_3,
            'Filter': 'No aneuploidy'
        })
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    if verbose:
        print("📊 Correlation Changes by Filter Stage:")
        print(results_df.round(4))
    
        # Test for significant changes (Fisher z-test)
        print("\n🔍 Statistical Tests for Correlation Changes:")
        
        if len(results) >= 2:
            # Test biggest drop: Raw vs Final
            r1_weeks, n1 = results[0]['r_weeks'], results[0]['N']
            r2_weeks, n2 = results[-1]['r_weeks'], results[-1]['N']
            z_stat, p_val = fisher_z_test(r1_weeks, n1, r2_weeks, n2)
            
            print(f"Weeks correlation change (Raw → Final):")
            print(f"  Raw: r = {r1_weeks:.4f} (N = {n1})")
            print(f"  Final: r = {r2_weeks:.4f} (N = {n2})")
            print(f"  Fisher z-test: z = {z_stat:.3f}, p = {p_val:.4f}")
            print(f"  {'Significant change!' if p_val < 0.05 else 'No significant change'}")
            
            # Test GC filter impact specifically
            if len(results) >= 3:
                r_before_gc = results[1]['r_weeks']  # After age filter
                n_before_gc = results[1]['N']
                r_after_gc = results[2]['r_weeks']   # After GC filter  
                n_after_gc = results[2]['N']
                z_stat_gc, p_val_gc = fisher_z_test(r_before_gc, n_before_gc, r_after_gc, n_after_gc)
                
                print(f"\nGC filter impact on weeks correlation:")
                print(f"  Before GC: r = {r_before_gc:.4f} (N = {n_before_gc})")
                print(f"  After GC: r = {r_after_gc:.4f} (N = {n_after_gc})")
                print(f"  Samples lost: {n_before_gc - n_after_gc} ({(n_before_gc - n_after_gc)/n_before_gc*100:.1f}%)")
                print(f"  Fisher z-test: z = {z_stat_gc:.3f}, p = {p_val_gc:.4f}")
                print(f"  {'GC filter significantly weakened correlation!' if p_val_gc < 0.05 else 'GC filter did not significantly change correlation'}")
    
        # Summary assessment
        print(f"\n🎯 Summary:")
        if len(results_df) > 0:
            initial_r = results_df.iloc[0]['r_weeks']
            final_r = results_df.iloc[-1]['r_weeks']
            change = final_r - initial_r
            pct_change = (change / abs(initial_r)) * 100 if initial_r != 0 else 0
            
            print(f"   Initial weeks correlation: {initial_r:.4f}")
            print(f"   Final weeks correlation: {final_r:.4f}")
            print(f"   Change: {change:+.4f} ({pct_change:+.1f}%)")
            print(f"   Sample retention: {results_df.iloc[-1]['N']}/{results_df.iloc[0]['N']} ({results_df.iloc[-1]['N']/results_df.iloc[0]['N']*100:.1f}%)")
            
            if abs(pct_change) > 20:
                print("   ⚠️  SUBSTANTIAL correlation change detected!")
            elif abs(pct_change) > 10:
                print("   ⚠️  Moderate correlation change detected")
            else:
                print("   ✅ Minimal correlation change")
    
    return results_df, stage0, stage3

if __name__ == "__main__":
    results_df, raw_data, clean_data = test_filter_impact()
