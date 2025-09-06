"""
Enhanced BMI grouping analysis for Problem 3.

This module extends Problem 2's BMI grouping with group contrasts and enhanced reporting.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings

# Import base BMI grouping from Problem 2 
from ..problem2.bmi_grouping import BMIGrouper


def create_enhanced_bmi_groups(df_X: pd.DataFrame, 
                              method: str = 'clinical',
                              verbose: bool = True) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    """
    Create enhanced BMI groups with detailed statistics for Problem 3.
    
    This function extends Problem 2's BMI grouping with enhanced reporting
    and group statistics needed for the extended analysis.
    
    Args:
        df_X: Feature matrix with BMI data and covariates
        method: Grouping method ('clinical', 'quantile', 'kmeans')
        verbose: Whether to print detailed information
        
    Returns:
        Tuple of (bmi_groups_dict, group_statistics_dict)
    """
    if verbose:
        print(f"🔄 Creating enhanced BMI groups using {method} method...")
    
    # Ensure BMI column exists
    bmi_col = 'bmi' if 'bmi' in df_X.columns else None
    if bmi_col is None and 'bmi_std' in df_X.columns:
        # If only standardized BMI exists, create original BMI for grouping
        # Note: This is approximate - ideally we'd store the original standardization params
        bmi_std_mean = df_X['bmi_std'].mean()
        bmi_std_std = df_X['bmi_std'].std()
        
        # Approximate reverse standardization (assuming typical BMI mean~25, std~5)
        df_X = df_X.copy()
        df_X['bmi'] = df_X['bmi_std'] * 5 + 25  # Approximate
        bmi_col = 'bmi'
        
        if verbose:
            print(f"   📊 Created approximate BMI from standardized values")
    
    if bmi_col is None:
        raise ValueError("No BMI column found in df_X")
    
    # Initialize BMI grouper from Problem 2
    grouper = BMIGrouper()
    
    # Create copy of df_X for processing
    df_with_groups = df_X.copy()
    
    # Apply BMI grouping using the specified method
    group_assignments = grouper.apply_grouping(df_with_groups, method=method)
    
    # Add group column to dataframe
    group_col = f'bmi_group_{method}'
    df_with_groups[group_col] = group_assignments
    
    if group_col not in df_with_groups.columns or df_with_groups[group_col].isna().all():
        raise ValueError(f"BMI grouping failed - {group_col} not created or all NaN")
    
    # Create groups dictionary
    bmi_groups = {}
    group_stats = {}
    
    unique_groups = df_with_groups[group_col].unique()
    unique_groups = [g for g in unique_groups if pd.notna(g)]
    
    if verbose:
        print(f"   📊 Found {len(unique_groups)} BMI groups: {unique_groups}")
    
    for group_name in unique_groups:
        # Get group data
        group_data = df_with_groups[df_with_groups[group_col] == group_name].copy()
        bmi_groups[group_name] = group_data
        
        # Calculate group statistics
        bmi_values = group_data[bmi_col].dropna()
        age_values = group_data['age'].dropna() if 'age' in group_data.columns else []
        
        group_stats[group_name] = {
            'n_patients': len(group_data),
            'n_observations': len(group_data),  # Same as patients for interval data
            'bmi_range': (bmi_values.min(), bmi_values.max()) if len(bmi_values) > 0 else (np.nan, np.nan),
            'mean_bmi': bmi_values.mean() if len(bmi_values) > 0 else np.nan,
            'mean_age': age_values.mean() if len(age_values) > 0 else np.nan,
            'group_identifier': group_name
        }
        
        if verbose:
            print(f"   📋 {group_name}: {len(group_data)} patients, BMI {bmi_values.min():.1f}-{bmi_values.max():.1f}")
    
    # Add group column back to original df_X for downstream use
    df_X_enhanced = df_X.copy()
    df_X_enhanced[group_col] = df_with_groups[group_col]
    
    if verbose:
        print(f"   ✅ Enhanced BMI grouping completed with {len(bmi_groups)} groups")
    
    return bmi_groups, group_stats


def compute_group_survival_extended(bmi_groups: Dict[str, pd.DataFrame],
                                  df_X: pd.DataFrame,
                                  aft_model: Any,
                                  selected_covariates: List[str],
                                  time_grid: Optional[np.ndarray] = None,
                                  verbose: bool = True) -> Dict[str, np.ndarray]:
    """
    Compute group survival functions using plug-in averaging over empirical distribution.
    
    S_g(t) = (1/|I_g|) * sum_{i in I_g} S(t|X_i)
    
    Args:
        bmi_groups: Dictionary of BMI groups from create_enhanced_bmi_groups
        df_X: Feature matrix with covariates
        aft_model: Fitted AFT model
        selected_covariates: List of covariates used in model
        time_grid: Time points for survival evaluation
        verbose: Whether to print progress information
        
    Returns:
        Dictionary of group survival functions
    """
    if time_grid is None:
        time_grid = np.linspace(10, 25, 100)
    
    if verbose:
        print(f"🔄 Computing group survival functions for {len(bmi_groups)} groups...")
    
    group_survival_funcs = {}
    
    for group_name, group_data in bmi_groups.items():
        n_group = len(group_data)
        
        if n_group == 0:
            if verbose:
                print(f"   ⚠️  Skipping {group_name}: no data")
            continue
        
        # Compute average survival function for this group
        survival_sum = np.zeros_like(time_grid)
        n_valid = 0
        
        for idx, row in group_data.iterrows():
            try:
                # Extract covariate values for this individual
                X_individual = pd.DataFrame({
                    col: [row[col]] for col in selected_covariates
                    if col in row.index and pd.notna(row[col])
                })
                
                if len(X_individual.columns) == len(selected_covariates):
                    # Predict survival function
                    survival_individual = aft_model.predict_survival_function(
                        X_individual, times=time_grid
                    )
                    survival_sum += survival_individual.values[:, 0]
                    n_valid += 1
                    
            except Exception as e:
                warnings.warn(f"Failed to predict for individual {idx}: {e}")
                continue
        
        if n_valid > 0:
            group_survival_funcs[group_name] = survival_sum / n_valid
            if verbose:
                print(f"   ✅ {group_name}: computed survival for {n_valid}/{n_group} patients")
        else:
            if verbose:
                print(f"   ⚠️  {group_name}: no valid predictions")
    
    return group_survival_funcs


def calculate_group_optimal_weeks(group_survival_funcs: Dict[str, np.ndarray],
                                time_grid: np.ndarray,
                                confidence_levels: List[float] = [0.90, 0.95],
                                verbose: bool = True) -> Dict[str, Dict[str, float]]:
    """
    Calculate optimal testing weeks t_g*(tau) for each group and confidence level.
    
    t_g*(tau) = inf{t: 1-S_g(t) >= tau}
    
    Args:
        group_survival_funcs: Group-specific survival functions
        time_grid: Time points corresponding to survival functions
        confidence_levels: List of confidence levels (tau)
        verbose: Whether to print progress information
        
    Returns:
        Dictionary: group -> tau -> optimal_week
    """
    optimal_weeks = {}
    
    for group_name, survival_func in group_survival_funcs.items():
        optimal_weeks[group_name] = {}
        
        # Calculate attainment probabilities
        attainment_prob = 1 - survival_func
        
        for tau in confidence_levels:
            # Find first time when attainment probability >= tau
            optimal_week = np.inf
            
            for i, t in enumerate(time_grid):
                if attainment_prob[i] >= tau:
                    optimal_week = t
                    break
            
            tau_key = f'tau_{tau}'
            optimal_weeks[group_name][tau_key] = optimal_week
    
    if verbose:
        print("📊 Group Optimal Weeks Calculated:")
        for group_name, group_weeks in optimal_weeks.items():
            weeks_str = ", ".join([f"{k}={v:.1f}" if not np.isinf(v) else f"{k}=>25" for k, v in group_weeks.items()])
            print(f"   • {group_name}: {weeks_str}")
    
    return optimal_weeks


def compute_group_contrasts(optimal_weeks: Dict[str, Dict[str, float]],
                           clinical_significance_threshold: float = 1.0,
                           verbose: bool = True) -> Dict[str, Dict[str, Any]]:
    """
    Compute between-group contrasts: Δt_{g,h}*(tau) = t_g*(tau) - t_h*(tau)
    
    Args:
        optimal_weeks: Optimal weeks per group/tau
        clinical_significance_threshold: Threshold for clinical significance (weeks)
        verbose: Whether to print progress information
        
    Returns:
        Group contrasts with clinical significance assessment
    """
    group_contrasts = {}
    group_names = list(optimal_weeks.keys())
    
    if verbose:
        print(f"🔄 Computing {len(group_names)} choose 2 = {len(group_names)*(len(group_names)-1)//2} group contrasts...")
    
    # Compute all pairwise contrasts
    for i, group_g in enumerate(group_names):
        for j, group_h in enumerate(group_names):
            if i < j:  # Avoid duplicates and self-contrasts
                contrast_name = f"{group_g}_vs_{group_h}"
                group_contrasts[contrast_name] = {}
                
                # Get all tau levels
                tau_levels = set(optimal_weeks[group_g].keys()) & set(optimal_weeks[group_h].keys())
                
                for tau_key in tau_levels:
                    week_g = optimal_weeks[group_g][tau_key]
                    week_h = optimal_weeks[group_h][tau_key]
                    
                    # Calculate contrast (g - h)
                    if not np.isinf(week_g) and not np.isinf(week_h):
                        contrast_value = week_g - week_h
                        abs_contrast = abs(contrast_value)
                        
                        # Assess clinical significance
                        clinical_significance = abs_contrast >= clinical_significance_threshold
                        
                        group_contrasts[contrast_name][tau_key] = {
                            'difference': contrast_value,
                            'absolute_difference': abs_contrast,
                            'clinical_significance': clinical_significance,
                            'direction': 'earlier' if contrast_value < 0 else 'later'
                        }
                    else:
                        # Handle infinite values
                        group_contrasts[contrast_name][tau_key] = {
                            'difference': np.inf,
                            'absolute_difference': np.inf,
                            'clinical_significance': False,
                            'direction': 'unknown'
                        }
    
    if verbose:
        print(f"   ✅ Computed contrasts for {len(group_contrasts)} group pairs")
    
    return group_contrasts


def assess_clinical_significance(group_contrasts: Dict[str, Dict[str, Dict[str, float]]],
                               threshold: float = 1.0) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Assess clinical significance of group contrasts.
    
    Args:
        group_contrasts: Between-group contrasts
        threshold: Clinical significance threshold in weeks
        
    Returns:
        Enhanced contrasts with clinical significance assessment
    """
    clinical_assessment = {}
    
    for method_name, method_contrasts in group_contrasts.items():
        clinical_assessment[method_name] = {}
        
        for contrast_name, contrast_values in method_contrasts.items():
            clinical_assessment[method_name][contrast_name] = {}
            
            for tau_key, contrast_value in contrast_values.items():
                abs_contrast = abs(contrast_value)
                
                # Assess clinical significance
                if abs_contrast >= threshold:
                    significance = "clinically_significant"
                elif abs_contrast >= threshold / 2:
                    significance = "borderline"
                else:
                    significance = "not_significant"
                
                clinical_assessment[method_name][contrast_name][tau_key] = {
                    'contrast_value': contrast_value,
                    'absolute_contrast': abs_contrast,
                    'clinical_significance': significance,
                    'direction': 'earlier' if contrast_value < 0 else 'later',
                    'meets_threshold': abs_contrast >= threshold
                }
    
    return clinical_assessment


def create_group_contrast_table(clinical_assessment: Dict[str, Dict[str, Dict[str, Any]]],
                              method_name: str = 'clinical') -> pd.DataFrame:
    """
    Create formatted table of group contrasts with clinical interpretation.
    
    Args:
        clinical_assessment: Clinical significance assessment results
        method_name: BMI grouping method to display
        
    Returns:
        Formatted DataFrame table
    """
    if method_name not in clinical_assessment:
        available_methods = list(clinical_assessment.keys())
        if available_methods:
            method_name = available_methods[0]
            warnings.warn(f"Method '{method_name}' not found. Using '{method_name}' instead.")
        else:
            return pd.DataFrame()
    
    table_data = []
    
    for contrast_name, contrast_data in clinical_assessment[method_name].items():
        groups = contrast_name.replace('_vs_', ' vs ')
        
        for tau_key, assessment in contrast_data.items():
            tau_value = float(tau_key.replace('tau_', ''))
            
            table_data.append({
                'Group_Contrast': groups,
                'Confidence_Level': f"{tau_value:.0%}",
                'Week_Difference': f"{assessment['contrast_value']:+.1f}",
                'Direction': assessment['direction'].title(),
                'Clinical_Significance': assessment['clinical_significance'].replace('_', ' ').title(),
                'Meets_Threshold': '✅' if assessment['meets_threshold'] else '❌'
            })
    
    df_contrasts = pd.DataFrame(table_data)
    
    print(f"📊 Group Contrast Table ({method_name} grouping):")
    print(df_contrasts.to_string(index=False))
    
    return df_contrasts


def perform_enhanced_group_analysis(df_intervals: pd.DataFrame,
                                  aft_model: Any,
                                  selected_covariates: List[str],
                                  confidence_levels: List[float] = [0.90, 0.95],
                                  clinical_threshold: float = 1.0) -> Dict[str, Any]:
    """
    Perform comprehensive group analysis with contrasts and clinical assessment.
    
    Args:
        df_intervals: DataFrame with BMI groups and interval data
        aft_model: Fitted AFT model
        selected_covariates: List of covariates used in model
        confidence_levels: Confidence levels for optimal weeks
        clinical_threshold: Clinical significance threshold (weeks)
        
    Returns:
        Complete analysis results
    """
    print("🔍 Starting Enhanced Group Analysis...")
    
    # Step 1: Compute group survival functions
    time_grid = np.linspace(10, 25, 100)
    group_survival_funcs = compute_group_survival_extended(
        df_intervals, aft_model, selected_covariates, time_grid
    )
    
    # Step 2: Calculate optimal weeks per group
    optimal_weeks = calculate_group_optimal_weeks(
        group_survival_funcs, time_grid, confidence_levels
    )
    
    # Step 3: Compute between-group contrasts
    group_contrasts = compute_group_contrasts(optimal_weeks)
    
    # Step 4: Assess clinical significance
    clinical_assessment = assess_clinical_significance(
        group_contrasts, clinical_threshold
    )
    
    # Step 5: Create summary tables
    contrast_tables = {}
    for method_name in clinical_assessment.keys():
        contrast_tables[method_name] = create_group_contrast_table(
            clinical_assessment, method_name
        )
    
    results = {
        'group_survival_functions': group_survival_funcs,
        'optimal_weeks': optimal_weeks,
        'group_contrasts': group_contrasts,
        'clinical_assessment': clinical_assessment,
        'contrast_tables': contrast_tables,
        'time_grid': time_grid,
        'analysis_params': {
            'confidence_levels': confidence_levels,
            'clinical_threshold': clinical_threshold,
            'n_time_points': len(time_grid)
        }
    }
    
    print("✅ Enhanced Group Analysis Complete")
    
    return results


def create_group_survival_plots(group_analysis: Dict[str, Any],
                               method_name: str = 'clinical',
                               figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Create comprehensive plots of group survival functions and optimal weeks.
    
    Args:
        group_analysis: Results from perform_enhanced_group_analysis
        method_name: BMI grouping method to plot
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    if method_name not in group_analysis['group_survival_functions']:
        available_methods = list(group_analysis['group_survival_functions'].keys())
        if available_methods:
            method_name = available_methods[0]
        else:
            raise ValueError("No group survival functions available")
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f'Group Survival Analysis - {method_name.title()} BMI Grouping', fontsize=16)
    
    time_grid = group_analysis['time_grid']
    group_survival_funcs = group_analysis['group_survival_functions'][method_name]
    optimal_weeks = group_analysis['optimal_weeks'][method_name]
    
    # Plot 1: Survival curves
    ax1 = axes[0, 0]
    colors = plt.cm.Set1(np.linspace(0, 1, len(group_survival_funcs)))
    
    for i, (group_name, survival_func) in enumerate(group_survival_funcs.items()):
        ax1.plot(time_grid, survival_func, label=f'Group {group_name}', 
                color=colors[i], linewidth=2)
    
    ax1.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='90% Attainment')
    ax1.axhline(y=0.05, color='darkred', linestyle='--', alpha=0.7, label='95% Attainment')
    ax1.set_xlabel('Gestational Week')
    ax1.set_ylabel('Survival Probability')
    ax1.set_title('Group Survival Functions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Optimal weeks comparison
    ax2 = axes[0, 1]
    
    groups = list(optimal_weeks.keys())
    tau_90_weeks = [optimal_weeks[g].get('tau_0.90', np.nan) for g in groups]
    tau_95_weeks = [optimal_weeks[g].get('tau_0.95', np.nan) for g in groups]
    
    x = np.arange(len(groups))
    width = 0.35
    
    ax2.bar(x - width/2, tau_90_weeks, width, label='90% Confidence', alpha=0.8)
    ax2.bar(x + width/2, tau_95_weeks, width, label='95% Confidence', alpha=0.8)
    
    ax2.set_xlabel('BMI Group')
    ax2.set_ylabel('Optimal Week')
    ax2.set_title('Optimal Testing Weeks by Group')
    ax2.set_xticks(x)
    ax2.set_xticklabels(groups, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Group contrasts heatmap
    ax3 = axes[1, 0]
    
    if method_name in group_analysis['clinical_assessment']:
        contrasts = group_analysis['clinical_assessment'][method_name]
        
        # Create contrast matrix for heatmap
        contrast_names = list(contrasts.keys())
        tau_levels = ['tau_0.90', 'tau_0.95']
        
        contrast_matrix = []
        for tau_key in tau_levels:
            row = [contrasts[name][tau_key]['contrast_value'] for name in contrast_names]
            contrast_matrix.append(row)
        
        im = ax3.imshow(contrast_matrix, cmap='RdBu_r', aspect='auto')
        ax3.set_xticks(range(len(contrast_names)))
        ax3.set_xticklabels([name.replace('_vs_', ' vs\n') for name in contrast_names], rotation=45)
        ax3.set_yticks(range(len(tau_levels)))
        ax3.set_yticklabels(['90%', '95%'])
        ax3.set_title('Group Contrasts (Week Differences)')
        
        # Add text annotations
        for i in range(len(tau_levels)):
            for j in range(len(contrast_names)):
                text = ax3.text(j, i, f'{contrast_matrix[i][j]:.1f}',
                               ha="center", va="center", color="black", fontweight='bold')
    
    # Plot 4: Clinical significance summary
    ax4 = axes[1, 1]
    
    if method_name in group_analysis['clinical_assessment']:
        significance_counts = {'clinically_significant': 0, 'borderline': 0, 'not_significant': 0}
        
        for contrast_data in group_analysis['clinical_assessment'][method_name].values():
            for assessment in contrast_data.values():
                sig_level = assessment['clinical_significance']
                significance_counts[sig_level] += 1
        
        labels = ['Clinically\nSignificant', 'Borderline', 'Not\nSignificant']
        sizes = [significance_counts['clinically_significant'], 
                significance_counts['borderline'],
                significance_counts['not_significant']]
        colors_pie = ['#ff6b6b', '#feca57', '#48cae4']
        
        ax4.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%', startangle=90)
        ax4.set_title('Clinical Significance\nof Group Contrasts')
    
    plt.tight_layout()
    return fig
