"""
BMI grouping methods for Problem 2 analysis.

This module provides various methods for grouping maternal BMI values
for group-specific survival analysis.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from sklearn.tree import DecisionTreeRegressor


class BMIGrouper:
    """
    Unified interface for BMI grouping methods.
    
    Supports multiple grouping strategies:
    - CART-based grouping using decision trees
    - Clinical BMI categories 
    - Data-driven quantiles (tertiles, quartiles)
    """
    
    def __init__(self):
        self.grouping_methods = {}
        self.cutpoints = {}
    
    def fit_cart_grouping(self, df: pd.DataFrame, 
                         target_col: str = 'predicted_median',
                         max_depth: int = 3, 
                         min_samples_leaf: int = 30,
                         random_state: int = 42) -> List[float]:
        """
        Fit CART-based BMI grouping using predicted median times.
        
        Args:
            df: DataFrame with BMI and target variable
            target_col: Column name for the target variable
            max_depth: Maximum tree depth
            min_samples_leaf: Minimum samples per leaf
            random_state: Random seed
            
        Returns:
            List of BMI cutpoints
        """
        # Filter valid predictions
        valid_mask = ~pd.isna(df[target_col])
        df_valid = df[valid_mask].copy()
        
        if len(df_valid) < min_samples_leaf * 2:
            print(f"⚠️ Insufficient data for CART grouping: {len(df_valid)} valid samples")
            return []
        
        # Fit decision tree
        tree = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state
        )
        
        X_tree = df_valid[['bmi']].values
        y_tree = df_valid[target_col].values
        
        tree.fit(X_tree, y_tree)
        
        # Extract cutpoints
        cutpoints = self._extract_tree_cutpoints(tree)
        self.cutpoints['cart'] = cutpoints
        
        return cutpoints
    
    def _extract_tree_cutpoints(self, tree) -> List[float]:
        """Extract BMI cutpoints from fitted decision tree."""
        cutpoints = []
        
        def traverse_tree(node, depth=0):
            if tree.tree_.children_left[node] != tree.tree_.children_right[node]:
                # Not a leaf node
                feature = tree.tree_.feature[node]
                threshold = tree.tree_.threshold[node]
                
                if feature == 0:  # BMI feature (first and only feature)
                    cutpoints.append(threshold)
                
                # Traverse children
                traverse_tree(tree.tree_.children_left[node], depth + 1)
                traverse_tree(tree.tree_.children_right[node], depth + 1)
        
        traverse_tree(0)
        return sorted(list(set(cutpoints)))
    
    def fit_quantile_grouping(self, df: pd.DataFrame, 
                             n_groups: int = 3) -> List[float]:
        """
        Fit quantile-based BMI grouping.
        
        Args:
            df: DataFrame with BMI data
            n_groups: Number of groups (3 for tertiles, 4 for quartiles)
            
        Returns:
            List of BMI cutpoints
        """
        quantiles = [i/n_groups for i in range(1, n_groups)]
        cutpoints = df['bmi'].quantile(quantiles).tolist()
        
        method_name = {3: 'tertile', 4: 'quartile'}.get(n_groups, f'{n_groups}_quantile')
        self.cutpoints[method_name] = cutpoints
        
        return cutpoints
    
    def apply_grouping(self, df: pd.DataFrame, method: str) -> pd.Series:
        """
        Apply a specific grouping method to the data.
        
        Args:
            df: DataFrame with BMI data
            method: Grouping method ('cart', 'clinical', 'tertile', 'quartile')
            
        Returns:
            Series with group assignments
        """
        if method == 'cart':
            if 'cart' not in self.cutpoints:
                raise ValueError("CART grouping not fitted. Call fit_cart_grouping first.")
            return df['bmi'].apply(lambda x: self._assign_cart_group(x, self.cutpoints['cart']))
        
        elif method == 'clinical':
            return df['bmi'].apply(assign_clinical_group)
        
        elif method == 'tertile':
            if 'tertile' not in self.cutpoints:
                self.fit_quantile_grouping(df, n_groups=3)
            return df['bmi'].apply(lambda x: self._assign_quantile_group(x, self.cutpoints['tertile'], 'T'))
        
        elif method == 'quartile':
            if 'quartile' not in self.cutpoints:
                self.fit_quantile_grouping(df, n_groups=4)
            return df['bmi'].apply(lambda x: self._assign_quantile_group(x, self.cutpoints['quartile'], 'Q'))
        
        else:
            raise ValueError(f"Unknown grouping method: {method}")
    
    def _assign_cart_group(self, bmi: float, cutpoints: List[float]) -> str:
        """Assign CART group based on BMI and cutpoints."""
        group_idx = 0
        for cutpoint in cutpoints:
            if bmi > cutpoint:
                group_idx += 1
        return f"CART_G{group_idx + 1}"
    
    def _assign_quantile_group(self, bmi: float, cutpoints: List[float], prefix: str) -> str:
        """Assign quantile group based on BMI and cutpoints."""
        group_idx = 0
        for cutpoint in cutpoints:
            if bmi > cutpoint:
                group_idx += 1
        
        if prefix == 'T':  # Tertiles
            labels = ['Low BMI (T1)', 'Medium BMI (T2)', 'High BMI (T3)']
        elif prefix == 'Q':  # Quartiles
            labels = ['Q1 (Low)', 'Q2 (Med-Low)', 'Q3 (Med-High)', 'Q4 (High)']
        else:
            labels = [f'{prefix}{i+1}' for i in range(len(cutpoints) + 1)]
        
        return labels[group_idx] if group_idx < len(labels) else labels[-1]
    
    def get_all_groupings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all grouping methods and return DataFrame with group columns.
        
        Args:
            df: DataFrame with BMI data
            
        Returns:
            DataFrame with added grouping columns
        """
        df_result = df.copy()
        
        # Apply all methods
        methods = ['clinical', 'tertile', 'quartile']
        
        # Add CART if data allows
        if 'predicted_median' in df.columns:
            try:
                self.fit_cart_grouping(df)
                methods.insert(0, 'cart')
            except Exception as e:
                print(f"⚠️ CART grouping failed: {e}")
        
        for method in methods:
            try:
                group_col = f'bmi_group_{method}'
                df_result[group_col] = self.apply_grouping(df, method)
            except Exception as e:
                print(f"⚠️ {method} grouping failed: {e}")
        
        return df_result
    
    def get_grouping_summary(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Get summary statistics for all grouping methods.
        
        Args:
            df: DataFrame with BMI data and grouping columns
            
        Returns:
            Dictionary of summary DataFrames by method
        """
        summaries = {}
        
        group_cols = [col for col in df.columns if col.startswith('bmi_group_')]
        
        for col in group_cols:
            method = col.replace('bmi_group_', '')
            
            summary = df.groupby(col).agg({
                'bmi': ['count', 'min', 'max', 'mean', 'std']
            }).round(2)
            
            # Add predicted median if available
            if 'predicted_median' in df.columns:
                pred_summary = df.groupby(col)['predicted_median'].agg(['mean', 'std']).round(2)
                summary = pd.concat([summary, pred_summary], axis=1)
            
            summaries[method] = summary
        
        return summaries


def assign_clinical_group(bmi: float) -> str:
    """
    Assign clinical BMI categories.
    
    Args:
        bmi: BMI value
        
    Returns:
        Clinical BMI category string
    """
    if bmi < 25:
        return "Normal (<25)"
    elif bmi < 30:
        return "Overweight (25-30)"
    elif bmi < 35:
        return "Obese I (30-35)"
    else:
        return "Obese II+ (≥35)"


def assign_tertile_group(bmi: float, cutpoints: List[float]) -> str:
    """
    Assign tertile-based group.
    
    Args:
        bmi: BMI value
        cutpoints: List of tertile cutpoints
        
    Returns:
        Tertile group string
    """
    if len(cutpoints) != 2:
        raise ValueError("Tertile grouping requires exactly 2 cutpoints")
    
    if bmi <= cutpoints[0]:
        return "Low BMI (T1)"
    elif bmi <= cutpoints[1]:
        return "Medium BMI (T2)"
    else:
        return "High BMI (T3)"


def assign_quartile_group(bmi: float, cutpoints: List[float]) -> str:
    """
    Assign quartile-based group.
    
    Args:
        bmi: BMI value
        cutpoints: List of quartile cutpoints
        
    Returns:
        Quartile group string
    """
    if len(cutpoints) != 3:
        raise ValueError("Quartile grouping requires exactly 3 cutpoints")
    
    if bmi <= cutpoints[0]:
        return "Q1 (Low)"
    elif bmi <= cutpoints[1]:
        return "Q2 (Med-Low)"
    elif bmi <= cutpoints[2]:
        return "Q3 (Med-High)"
    else:
        return "Q4 (High)"


def calculate_predicted_median_times(df_intervals: pd.DataFrame, aft_model, verbose: bool = True) -> pd.DataFrame:
    """
    Calculate individual predicted median survival times for CART grouping.
    
    Args:
        df_intervals: DataFrame with interval-censored data and BMI
        aft_model: Fitted AFT model
        verbose: Whether to print progress information
        
    Returns:
        DataFrame with added predicted_median column
    """
    if verbose:
        print("🔄 Calculating individual predicted median survival times...")
    
    df_result = df_intervals.copy()
    
    # Get BMI standardization parameters
    bmi_mean = df_intervals['bmi'].mean()
    bmi_std = df_intervals['bmi'].std()
    
    median_times = []
    
    for idx, row in df_intervals.iterrows():
        try:
            # Standardize BMI
            bmi_z = (row['bmi'] - bmi_mean) / bmi_std
            X_query = pd.DataFrame({'bmi_z': [bmi_z]})
            
            # Predict median survival time (50th percentile)
            median_time = aft_model.predict_percentile(X_query, p=0.5).iloc[0]
            median_times.append(median_time)
            
        except Exception as e:
            if verbose:
                print(f"⚠️ Failed to predict median for index {idx}: {e}")
            median_times.append(np.nan)
    
    df_result['predicted_median'] = median_times
    
    if verbose:
        valid_predictions = ~pd.isna(df_result['predicted_median'])
        print(f"✅ Predicted median times calculated: {valid_predictions.sum()}/{len(df_result)} valid")
        
        if valid_predictions.sum() > 0:
            median_stats = df_result[valid_predictions]['predicted_median'].describe()
            print(f"\n📊 Predicted Median Time Statistics:")
            print(f"  Mean: {median_stats['mean']:.2f} weeks")
            print(f"  Std: {median_stats['std']:.2f} weeks")
            print(f"  Range: {median_stats['min']:.2f} - {median_stats['max']:.2f} weeks")
    
    return df_result


def evaluate_grouping_strategy(df_grouped: pd.DataFrame, grouping_col: str, 
                              penalty_lambda: float = 1.0, verbose: bool = True) -> Dict[str, Any]:
    """
    Evaluate a BMI grouping strategy using risk-based scoring.
    
    Args:
        df_grouped: DataFrame with BMI groups and predicted median times
        grouping_col: Column name for BMI groups
        penalty_lambda: Penalty weight for number of groups
        verbose: Whether to print evaluation details
        
    Returns:
        Dictionary with evaluation metrics
    """
    if verbose:
        print(f"📊 Evaluating grouping strategy: {grouping_col}")
    
    # Group statistics
    group_stats = df_grouped.groupby(grouping_col).agg({
        'predicted_median': ['count', 'mean', 'std'],
        'bmi': ['mean', 'std']
    }).round(3)
    
    # Calculate risk score components
    groups = df_grouped[grouping_col].unique()
    K = len(groups)  # Number of groups
    
    # Simple risk score: weighted average of group medians + complexity penalty
    group_sizes = df_grouped[grouping_col].value_counts()
    total_size = len(df_grouped)
    
    weighted_median = 0
    for group in groups:
        group_data = df_grouped[df_grouped[grouping_col] == group]
        if len(group_data) > 0:
            group_median = group_data['predicted_median'].mean()
            group_weight = len(group_data) / total_size
            weighted_median += group_weight * group_median
    
    # Risk score: lower is better (earlier detection is better)
    risk_score = weighted_median + penalty_lambda * K
    
    # Additional metrics
    between_group_variance = df_grouped.groupby(grouping_col)['predicted_median'].mean().var()
    within_group_variance = df_grouped.groupby(grouping_col)['predicted_median'].var().mean()
    
    evaluation = {
        'grouping_method': grouping_col,
        'n_groups': K,
        'risk_score': risk_score,
        'weighted_median': weighted_median,
        'complexity_penalty': penalty_lambda * K,
        'between_group_variance': between_group_variance,
        'within_group_variance': within_group_variance,
        'group_sizes': group_sizes.to_dict(),
        'group_stats': group_stats
    }
    
    if verbose:
        print(f"  Number of groups: {K}")
        print(f"  Risk score: {risk_score:.3f}")
        print(f"  Weighted median: {weighted_median:.3f} weeks")
        print(f"  Complexity penalty: {penalty_lambda * K:.3f}")
        print(f"  Between-group variance: {between_group_variance:.3f}")
        print(f"  Within-group variance: {within_group_variance:.3f}")
        
        print(f"\n  Group sizes:")
        for group, size in group_sizes.items():
            pct = 100 * size / total_size
            print(f"    {group}: {size} ({pct:.1f}%)")
    
    return evaluation


def perform_group_specific_analysis(df_intervals: pd.DataFrame, aft_model, 
                                   confidence_levels: List[float] = [0.90, 0.95],
                                   time_grid: Optional[np.ndarray] = None,
                                   grouping_methods: List[str] = ['clinical', 'tertile'],
                                   verbose: bool = True) -> Dict[str, Any]:
    """
    Perform comprehensive group-specific optimal weeks analysis.
    
    Args:
        df_intervals: DataFrame with interval-censored data
        aft_model: Fitted AFT model
        confidence_levels: List of confidence levels for optimal weeks
        time_grid: Time points for analysis (default: 10-25 weeks)
        grouping_methods: List of grouping methods to analyze
        verbose: Whether to print detailed progress
        
    Returns:
        Dictionary with group-specific results
    """
    if verbose:
        print("🎯 Performing group-specific optimal weeks analysis...")
    
    if time_grid is None:
        time_grid = np.linspace(10, 25, 100)
    
    # Step 1: Calculate predicted median times
    df_with_medians = calculate_predicted_median_times(df_intervals, aft_model, verbose=verbose)
    
    # Step 2: Initialize BMI grouper and apply all groupings
    grouper = BMIGrouper()
    df_grouped = grouper.get_all_groupings(df_with_medians)
    
    # Step 3: Evaluate grouping strategies
    evaluations = {}
    available_groupings = [col for col in df_grouped.columns if col.startswith('bmi_group_')]
    
    if verbose:
        print(f"\n📊 Evaluating {len(available_groupings)} grouping strategies...")
    
    for group_col in available_groupings:
        method = group_col.replace('bmi_group_', '')
        if method in grouping_methods or 'all' in grouping_methods:
            try:
                evaluation = evaluate_grouping_strategy(df_grouped, group_col, verbose=verbose)
                evaluations[method] = evaluation
            except Exception as e:
                if verbose:
                    print(f"⚠️ Evaluation failed for {method}: {e}")
    
    # Step 4: Select best grouping strategy
    if evaluations:
        best_method = min(evaluations.keys(), key=lambda x: evaluations[x]['risk_score'])
        best_grouping_col = f'bmi_group_{best_method}'
        
        if verbose:
            print(f"\n🏆 Best grouping strategy: {best_method}")
            print(f"  Risk score: {evaluations[best_method]['risk_score']:.3f}")
    else:
        best_method = 'clinical'
        best_grouping_col = 'bmi_group_clinical'
        if verbose:
            print("⚠️ No successful evaluations, defaulting to clinical grouping")
    
    # Step 5: Calculate group-specific optimal weeks
    groups = df_grouped[best_grouping_col].unique()
    group_results = {}
    
    if verbose:
        print(f"\n🔄 Calculating optimal weeks for {len(groups)} BMI groups...")
    
    # Get BMI standardization parameters
    bmi_mean = df_intervals['bmi'].mean()
    bmi_std = df_intervals['bmi'].std()
    
    for group in groups:
        if verbose:
            print(f"\n  📊 Analyzing group: {group}")
        
        group_data = df_grouped[df_grouped[best_grouping_col] == group]
        
        # Use representative BMI (median) for group
        representative_bmi = group_data['bmi'].median()
        bmi_z = (representative_bmi - bmi_mean) / bmi_std
        
        # Generate survival curve for this group
        X_query = pd.DataFrame({'bmi_z': [bmi_z]})
        
        try:
            surv_func = aft_model.predict_survival_function(X_query, times=time_grid)
            survival_values = surv_func.iloc[:, 0].values
            
            # Calculate optimal weeks for each confidence level
            optimal_weeks = {}
            for conf in confidence_levels:
                threshold = 1 - conf
                crossing_indices = np.where(survival_values <= threshold)[0]
                
                if len(crossing_indices) > 0:
                    optimal_week = time_grid[crossing_indices[0]]
                else:
                    optimal_week = np.inf
                
                optimal_weeks[conf] = optimal_week
                
                if verbose:
                    week_str = f"{optimal_week:.1f}" if optimal_week != np.inf else "Never"
                    print(f"    {int(conf*100)}% confidence: Week {week_str}")
            
            group_results[group] = {
                'n_mothers': len(group_data),
                'representative_bmi': representative_bmi,
                'bmi_range': f"{group_data['bmi'].min():.1f}-{group_data['bmi'].max():.1f}",
                'optimal_weeks': optimal_weeks,
                'survival_curve': {'times': time_grid, 'survival': survival_values}
            }
            
        except Exception as e:
            if verbose:
                print(f"    ❌ Failed to analyze group {group}: {e}")
            group_results[group] = {
                'n_mothers': len(group_data),
                'representative_bmi': representative_bmi,
                'error': str(e)
            }
    
    # Compile final results
    results = {
        'best_grouping_method': best_method,
        'best_grouping_column': best_grouping_col,
        'grouping_evaluations': evaluations,
        'group_results': group_results,
        'confidence_levels': confidence_levels,
        'df_grouped': df_grouped
    }
    
    if verbose:
        print(f"\n✅ Group-specific analysis completed for {len(group_results)} groups")
    
    return results


def create_group_optimal_weeks_summary(group_analysis_results: Dict[str, Any], 
                                      verbose: bool = True) -> pd.DataFrame:
    """
    Create a summary table of group-specific optimal weeks.
    
    Args:
        group_analysis_results: Results from perform_group_specific_analysis()
        verbose: Whether to print the summary table
        
    Returns:
        DataFrame with group-specific optimal weeks summary
    """
    group_results = group_analysis_results['group_results']
    confidence_levels = group_analysis_results['confidence_levels']
    
    summary_data = []
    
    for group_name, group_data in group_results.items():
        if 'optimal_weeks' in group_data:
            row = {
                'BMI_Group': group_name,
                'n_mothers': group_data['n_mothers'],
                'representative_BMI': group_data['representative_bmi'],
                'BMI_range': group_data['bmi_range']
            }
            
            # Add optimal weeks for each confidence level
            for conf in confidence_levels:
                week = group_data['optimal_weeks'][conf]
                week_str = f"{week:.1f}" if week != np.inf else "Never"
                row[f'optimal_week_{int(conf*100)}'] = week_str
            
            summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    
    if verbose and not summary_df.empty:
        print("📋 Group-Specific Optimal Testing Weeks Summary:")
        print("="*70)
        print(summary_df.to_string(index=False))
        
        # Additional insights
        total_mothers = summary_df['n_mothers'].sum()
        print(f"\n📊 Summary Statistics:")
        print(f"  Total mothers: {total_mothers}")
        print(f"  Number of BMI groups: {len(summary_df)}")
        print(f"  Grouping method: {group_analysis_results['best_grouping_method']}")
    
    return summary_df
