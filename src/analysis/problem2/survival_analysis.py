"""
Survival analysis functions for Problem 2.

This module contains high-level functions for conducting the complete
survival analysis workflow including group-specific analysis and validation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

from ...models.aft_models import AFTSurvivalAnalyzer, OptimalWeeksCalculator, SurvivalResults
from .bmi_grouping import BMIGrouper


def compute_group_survival_analysis(df_intervals: pd.DataFrame, 
                                  analyzer: AFTSurvivalAnalyzer,
                                  grouper: BMIGrouper,
                                  confidence_levels: List[float] = [0.90, 0.95],
                                  time_grid: Optional[np.ndarray] = None) -> Tuple[Dict, Dict]:
    """
    Compute comprehensive group-specific survival analysis.
    
    Args:
        df_intervals: DataFrame with interval-censored data and BMI groups
        analyzer: Fitted AFT survival analyzer
        grouper: BMI grouper with fitted grouping methods
        confidence_levels: List of confidence levels for optimal weeks
        time_grid: Time grid for survival functions (default: 10-25 weeks)
        
    Returns:
        Tuple of (group_survival_results, optimal_weeks_results)
    """
    if time_grid is None:
        time_grid = np.linspace(10, 25, 100)
    
    # Get all grouping methods available
    group_columns = [col for col in df_intervals.columns if col.startswith('bmi_group_')]
    
    group_survival_results = {}
    optimal_weeks_results = {}
    
    last_observed_week = 24.0
    
    for group_col in group_columns:
        method_name = group_col.replace('bmi_group_', '').title()
        
        print(f"\n🔄 Processing {method_name} grouping method...")
        
        groups = df_intervals[group_col].unique()
        method_survival_results = {}
        method_optimal_weeks = {}
        
        for group_name in sorted(groups):
            group_data = df_intervals[df_intervals[group_col] == group_name]
            n_mothers = len(group_data)
            
            if n_mothers < 5:
                print(f"  ⚠️ Skipping {group_name}: too few mothers ({n_mothers})")
                continue
            
            # Calculate representative BMI for this group
            group_bmi_mean = group_data['bmi'].mean()
            
            try:
                # Generate survival function for this group
                survival_results = analyzer.predict_survival_function([group_bmi_mean], time_grid)
                survival_result = list(survival_results.values())[0]
                
                # Update with group-specific information
                survival_result.group_name = group_name
                survival_result.n_mothers = n_mothers
                survival_result.bmi = group_bmi_mean
                
                method_survival_results[group_name] = survival_result
                
                # Calculate optimal weeks for this group
                group_optimal_weeks = OptimalWeeksCalculator.calculate_group_optimal_weeks(
                    {group_name: survival_result}, confidence_levels
                )
                method_optimal_weeks[group_name] = group_optimal_weeks[group_name]
                
                print(f"  ✅ {group_name}: BMI {group_bmi_mean:.1f}, N={n_mothers}, P(week 24)={survival_result.prob_at_last_week:.1%}")
                
            except Exception as e:
                print(f"  ❌ {group_name}: Failed to compute survival function - {e}")
        
        group_survival_results[method_name] = method_survival_results
        optimal_weeks_results[method_name] = method_optimal_weeks
    
    return group_survival_results, optimal_weeks_results


def create_enhanced_summary_table(group_survival_results: Dict, 
                                optimal_weeks_results: Dict,
                                confidence_levels: List[float] = [0.90, 0.95]) -> pd.DataFrame:
    """
    Create enhanced summary table with optimal weeks and group information.
    
    Args:
        group_survival_results: Dictionary of survival results by method and group
        optimal_weeks_results: Dictionary of optimal weeks by method and group
        confidence_levels: List of confidence levels
        
    Returns:
        DataFrame with comprehensive summary
    """
    summary_data = []
    never_details = []
    
    for method_name, method_results in group_survival_results.items():
        if not method_results:
            continue
        
        method_optimal_weeks = optimal_weeks_results.get(method_name, {})
        
        for group_name, survival_result in method_results.items():
            group_optimal_weeks = method_optimal_weeks.get(group_name, {})
            
            # Process optimal weeks for different confidence levels
            optimal_week_strs = {}
            
            for conf_level in confidence_levels:
                optimal_result = group_optimal_weeks.get(conf_level, np.inf)
                
                if isinstance(optimal_result, dict):
                    # "Never" case with detailed info
                    week_str = f"Never (P={optimal_result['prob_at_last_week']:.1%}@24w)"
                    never_details.append({
                        'Method': method_name,
                        'Group': group_name,
                        'Confidence': f"{conf_level:.0%}",
                        'Target_Prob': optimal_result['target_prob'],
                        'Actual_Prob_at_24w': optimal_result['prob_at_last_week'],
                        'Status': optimal_result['status'],
                        'BMI_Mean': survival_result.bmi,
                        'N_Mothers': survival_result.n_mothers
                    })
                else:
                    # Regular week value
                    week_str = f"{optimal_result:.1f}" if optimal_result != np.inf else "Never"
                
                optimal_week_strs[conf_level] = week_str
            
            # Calculate BMI range (approximation since we only have mean)
            bmi_range = f"{survival_result.bmi-1:.1f}-{survival_result.bmi+1:.1f}"  # Approximate
            
            summary_data.append({
                'Grouping Method': method_name,
                'BMI Group': group_name,
                'N Mothers': survival_result.n_mothers,
                'BMI Range': bmi_range,
                'Mean BMI': f"{survival_result.bmi:.1f}",
                'Prob at 24w': f"{survival_result.prob_at_last_week:.1%}",
                'Optimal Week 90%': optimal_week_strs.get(0.90, "N/A"),
                'Optimal Week 95%': optimal_week_strs.get(0.95, "N/A")
            })
    
    return pd.DataFrame(summary_data), pd.DataFrame(never_details)


def create_survival_curves_plot(group_survival_results: Dict, 
                              output_path: Optional[Path] = None,
                              figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
    """
    Create comprehensive survival curves visualization.
    
    Args:
        group_survival_results: Dictionary of survival results by method
        output_path: Optional path to save the plot
        figsize: Figure size tuple
        
    Returns:
        Matplotlib figure object
    """
    n_methods = len(group_survival_results)
    if n_methods == 0:
        raise ValueError("No survival results to plot")
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    colors = plt.cm.Set1(np.linspace(0, 1, 8))  # Use colormap for consistent colors
    
    for i, (method_name, method_results) in enumerate(group_survival_results.items()):
        if i >= 4:  # Maximum 4 subplots
            break
        
        ax = axes[i]
        
        for j, (group_name, survival_result) in enumerate(method_results.items()):
            color = colors[j % len(colors)]
            
            # Plot survival curve
            ax.plot(survival_result.times, survival_result.survival_probs, 
                   color=color, linewidth=2.5, 
                   label=f'{group_name} (BMI {survival_result.bmi:.1f})')
        
        # Add confidence level lines
        ax.axhline(y=0.10, color='red', linestyle='--', alpha=0.7, label='90% reached threshold')
        ax.axhline(y=0.05, color='darkred', linestyle='--', alpha=0.7, label='95% reached threshold')
        
        ax.set_xlabel('Gestational Weeks')
        ax.set_ylabel('Survival Probability\n(Prob Y<4%)')
        ax.set_title(f'{method_name} BMI Groups\nSurvival Curves')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_xlim(10, 25)
        ax.set_ylim(0, 1)
    
    # Hide unused subplots
    for i in range(n_methods, 4):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Survival curves saved to: {output_path}")
    
    return fig


def generate_clinical_recommendations(summary_df: pd.DataFrame, 
                                    never_df: pd.DataFrame) -> Dict[str, str]:
    """
    Generate clinical recommendations based on analysis results.
    
    Args:
        summary_df: Summary DataFrame with optimal weeks
        never_df: DataFrame with "never" cases details
        
    Returns:
        Dictionary with clinical recommendations
    """
    recommendations = {}
    
    # Count achievable vs never cases
    total_groups = len(summary_df)
    achievable_90 = sum(1 for val in summary_df['Optimal Week 90%'] if not val.startswith('Never'))
    achievable_95 = sum(1 for val in summary_df['Optimal Week 95%'] if not val.startswith('Never'))
    
    recommendations['achievability'] = {
        'total_groups': total_groups,
        'achievable_90_pct': achievable_90,
        'achievable_95_pct': achievable_95,
        'achievable_90_rate': achievable_90 / total_groups if total_groups > 0 else 0,
        'achievable_95_rate': achievable_95 / total_groups if total_groups > 0 else 0
    }
    
    # Clinical guidance
    if achievable_90 >= 0.8 * total_groups:
        recommendations['conf_90'] = "✅ 90% confidence level recommended for routine clinical use"
    else:
        recommendations['conf_90'] = "⚠️ Consider lower confidence thresholds for clinical practice"
    
    if achievable_95 < 0.5 * total_groups:
        recommendations['conf_95'] = "❌ 95% confidence level NOT recommended for routine use"
    else:
        recommendations['conf_95'] = "✅ 95% confidence level feasible for clinical use"
    
    # BMI-specific guidance
    try:
        # Extract numeric BMI values and weeks for correlation analysis
        bmi_values = []
        week_90_values = []
        
        for _, row in summary_df.iterrows():
            try:
                bmi = float(row['Mean BMI'])
                week_90_str = row['Optimal Week 90%']
                
                if not week_90_str.startswith('Never'):
                    week_90 = float(week_90_str)
                    bmi_values.append(bmi)
                    week_90_values.append(week_90)
            except:
                continue
        
        if len(bmi_values) > 3:
            correlation = np.corrcoef(bmi_values, week_90_values)[0,1]
            
            low_bmi_weeks = [w for b, w in zip(bmi_values, week_90_values) if b < 30]
            high_bmi_weeks = [w for b, w in zip(bmi_values, week_90_values) if b >= 35]
            
            low_bmi_avg = np.mean(low_bmi_weeks) if len(low_bmi_weeks) > 0 else np.nan
            high_bmi_avg = np.mean(high_bmi_weeks) if len(high_bmi_weeks) > 0 else np.nan
            
            recommendations['bmi_correlation'] = correlation
            recommendations['low_bmi_avg_week'] = low_bmi_avg
            recommendations['high_bmi_avg_week'] = high_bmi_avg
            
            if not np.isnan(low_bmi_avg) and not np.isnan(high_bmi_avg):
                recommendations['bmi_effect'] = high_bmi_avg - low_bmi_avg
        
    except Exception as e:
        recommendations['bmi_analysis_error'] = str(e)
    
    return recommendations


def save_results(summary_df: pd.DataFrame, 
                never_df: pd.DataFrame,
                recommendations: Dict,
                output_path: Path):
    """
    Save all analysis results to files.
    
    Args:
        summary_df: Summary DataFrame
        never_df: Never cases DataFrame  
        recommendations: Clinical recommendations dictionary
        output_path: Base output directory path
    """
    # Save summary table
    summary_file = output_path / 'p2_enhanced_optimal_weeks_summary.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f"💾 Enhanced summary saved to: {summary_file}")
    
    # Save never cases details
    if len(never_df) > 0:
        never_file = output_path / 'p2_never_cases_detailed_analysis.csv'
        never_df.to_csv(never_file, index=False)
        print(f"💾 Never cases analysis saved to: {never_file}")
    
    # Save recommendations as text
    recommendations_file = output_path / 'p2_clinical_recommendations.txt'
    with open(recommendations_file, 'w') as f:
        f.write("Clinical Recommendations - Problem 2\n")
        f.write("=" * 50 + "\n\n")
        
        for key, value in recommendations.items():
            f.write(f"{key}: {value}\n")
    
    print(f"💾 Clinical recommendations saved to: {recommendations_file}")


def run_complete_analysis(df_intervals: pd.DataFrame,
                         analyzer: AFTSurvivalAnalyzer,
                         output_path: Path,
                         confidence_levels: List[float] = [0.90, 0.95]) -> Dict:
    """
    Run the complete survival analysis workflow.
    
    Args:
        df_intervals: DataFrame with interval-censored data
        analyzer: Fitted AFT survival analyzer
        output_path: Output directory path
        confidence_levels: List of confidence levels
        
    Returns:
        Dictionary with all analysis results
    """
    print("🚀 Starting complete survival analysis workflow...")
    
    # Step 1: Set up BMI grouping
    grouper = BMIGrouper()
    df_with_groups = grouper.get_all_groupings(df_intervals)
    
    # Step 2: Compute group-specific survival analysis
    group_survival_results, optimal_weeks_results = compute_group_survival_analysis(
        df_with_groups, analyzer, grouper, confidence_levels
    )
    
    # Step 3: Create summary tables
    summary_df, never_df = create_enhanced_summary_table(
        group_survival_results, optimal_weeks_results, confidence_levels
    )
    
    # Step 4: Generate clinical recommendations
    recommendations = generate_clinical_recommendations(summary_df, never_df)
    
    # Step 5: Create visualizations
    fig = create_survival_curves_plot(
        group_survival_results, 
        output_path / 'p2_survival_curves_by_bmi_groups.png'
    )
    
    # Step 6: Save all results
    save_results(summary_df, never_df, recommendations, output_path)
    
    print("✅ Complete survival analysis workflow finished!")
    
    return {
        'summary_df': summary_df,
        'never_df': never_df,
        'recommendations': recommendations,
        'group_survival_results': group_survival_results,
        'optimal_weeks_results': optimal_weeks_results,
        'figure': fig
    }
