"""
Data preprocessing functions for Problem 2: Y-chromosome threshold analysis.

This module contains functions for cleaning and preparing the male fetus data
for both interval-censored survival analysis and simple time-to-event analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from pathlib import Path
from typing import Tuple, Dict, Any, Optional


def parse_gestational_weeks(week_str: str) -> float:
    """
    Convert gestational weeks from format "11w+6" to decimal weeks (11.857).
    
    Args:
        week_str: String in format like "11w+6" or "11w" or numeric string
        
    Returns:
        Decimal weeks as float, or NaN if parsing fails
    """
    if pd.isna(week_str):
        return np.nan
    
    week_str = str(week_str).strip()
    
    if 'w' in week_str.lower():
        parts = re.split('[wW]', week_str)
        if len(parts) == 2:
            weeks = int(parts[0])
            days_part = parts[1].strip()
            
            if '+' in days_part:
                days = int(days_part.split('+')[1])
            else:
                days = 0
            
            return weeks + days / 7.0
    
    try:
        return float(week_str)
    except:
        return np.nan


def remove_outliers_iqr(data: pd.DataFrame, column: str, factor: float = 1.5) -> pd.DataFrame:
    """
    Remove outliers using IQR method.
    
    Args:
        data: DataFrame to filter
        column: Column name to check for outliers
        factor: IQR multiplier factor (default 1.5)
        
    Returns:
        Filtered DataFrame with outliers removed
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]


def apply_qc_filters(df: pd.DataFrame, verbose: bool = True) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Apply quality control filters to the male fetus dataset.
    
    Args:
        df: Raw DataFrame with male fetus data
        verbose: Whether to print detailed progress information
        
    Returns:
        Tuple of (filtered_df, filter_stats_dict)
    """
    initial_count = len(df)
    filter_stats = {'initial': initial_count}
    
    if verbose:
        print("🔧 Parsing variables and applying QC filters...")
    
    # Parse variables
    df = df.copy()
    df['gestational_weeks'] = df['检测孕周'].apply(parse_gestational_weeks)  
    df['bmi'] = pd.to_numeric(df['孕妇BMI'], errors='coerce')   
    df['y_concentration'] = pd.to_numeric(df['Y染色体浓度'], errors='coerce')  
    df['maternal_id'] = df['孕妇代码']  
    df['gc_content'] = pd.to_numeric(df['GC含量'], errors='coerce')  
    df['aneuploidy'] = df['染色体的非整倍体']  
    
    if verbose:
        print(f"✅ Variable parsing completed:")
        print(f"  Gestational weeks: {df['gestational_weeks'].notna().sum()}/{len(df)} valid")
        print(f"  BMI: {df['bmi'].notna().sum()}/{len(df)} valid") 
        print(f"  Y concentration: {df['y_concentration'].notna().sum()}/{len(df)} valid")
    
    # 1. Gestational weeks: 10-25 weeks
    df = df[(df['gestational_weeks'] >= 10) & (df['gestational_weeks'] <= 25)]
    filter_stats['after_gestational_weeks'] = len(df)
    if verbose:
        print(f"  After gestational weeks filter (10-25): {len(df)} records")
    
    # 2. GC content: 40-60%  
    df = df[(df['gc_content'] >= 0.40) & (df['gc_content'] <= 0.60)]
    filter_stats['after_gc_content'] = len(df)
    if verbose:
        print(f"  After GC content filter (40-60%): {len(df)} records")
    
    # 3. Remove chromosome abnormalities
    df = df[df['aneuploidy'].isna()]
    filter_stats['after_aneuploidy'] = len(df)
    if verbose:
        print(f"  After chromosome abnormality filter: {len(df)} records")
    
    # 4. Remove missing key variables
    df = df.dropna(subset=['gestational_weeks', 'bmi', 'y_concentration'])
    filter_stats['after_missing_data'] = len(df)
    if verbose:
        print(f"  After missing data filter: {len(df)} records")
    
    # 5. IQR outlier filters
    if verbose:
        print("🎯 Applying IQR outlier detection...")
    
    for col in ['gestational_weeks', 'bmi', 'y_concentration']:
        df_before = len(df)
        
        if verbose:
            # Show outlier detection details
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            print(f"  {col}:")
            print(f"    IQR outliers: {outliers_mask.sum()} ({100*outliers_mask.mean():.2f}%) - bounds: [{lower_bound:.3f}, {upper_bound:.3f}]")
        
        df = remove_outliers_iqr(df, col)
        filter_stats[f'after_iqr_{col}'] = len(df)
        if verbose:
            print(f"    After IQR filtering: {len(df)} records (removed {df_before - len(df)})")
    
    # Keep only required columns for interval construction  
    df_clean = df[['maternal_id', 'gestational_weeks', 'bmi', 'y_concentration']].copy()
    df_clean = df_clean.sort_values(['maternal_id', 'gestational_weeks']).reset_index(drop=True)
    
    filter_stats['final'] = len(df_clean)
    filter_stats['retention_rate'] = len(df_clean) / initial_count
    
    if verbose:
        print(f"\n✅ QC filtering completed: {len(df_clean)} records remaining ({100*len(df_clean)/initial_count:.1f}% retention)")
        print(f"  Unique mothers: {df_clean['maternal_id'].nunique()}")
        print(f"  Tests per mother: {len(df_clean) / df_clean['maternal_id'].nunique():.1f} average")
    
    return df_clean, filter_stats


def construct_intervals(df_tests: pd.DataFrame, threshold: float = 0.04, verbose: bool = True) -> pd.DataFrame:
    """
    Convert individual tests to interval-censored format.
    
    For each maternal_id, determine:
    1. Left-censored: First observation already ≥4% → L=0, R=first_week
    2. Interval-censored: Threshold crossed between visits → L=weeks[j-1], R=weeks[j]  
    3. Right-censored: Never reached threshold → L=last_week, R=inf
    
    Args:
        df_tests: DataFrame with individual test records
        threshold: Y-chromosome concentration threshold (default 0.04 = 4%)
        verbose: Whether to print progress information
        
    Returns:
        DataFrame with interval-censored observations
    """
    intervals = []
    
    if verbose:
        print("🔄 Constructing interval-censored observations...")
    
    for maternal_id, group in df_tests.groupby('maternal_id'):
        # Sort by gestational weeks
        group = group.sort_values('gestational_weeks').reset_index(drop=True)
        bmi = group['bmi'].iloc[0]  # Use first BMI value (should be consistent)
        
        # Find threshold crossings
        threshold_mask = group['y_concentration'] >= threshold
        
        if threshold_mask.iloc[0]:
            # Left-censored: First observation already ≥4%
            L, R = 0, group['gestational_weeks'].iloc[0]
            censor_type = 'left'
        elif threshold_mask.any():
            # Interval-censored: Threshold crossed between visits
            cross_idx = threshold_mask.idxmax()  # First True index
            cross_pos = group.index[group.index == cross_idx][0] - group.index[0]  # Position in group
            
            if cross_pos > 0:
                L = group['gestational_weeks'].iloc[cross_pos - 1]
            else:
                L = 0  # Edge case: first observation is crossing
            R = group['gestational_weeks'].iloc[cross_pos]
            censor_type = 'interval'
        else:
            # Right-censored: Never reached threshold
            L = group['gestational_weeks'].iloc[-1]
            R = np.inf
            censor_type = 'right'
        
        intervals.append({
            'maternal_id': maternal_id,
            'bmi': bmi,
            'L': L, 
            'R': R,
            'censor_type': censor_type,
            'n_tests': len(group),
            'weeks_range': f"{group['gestational_weeks'].min():.1f}-{group['gestational_weeks'].max():.1f}",
            'max_y_concentration': group['y_concentration'].max()
        })
    
    df_intervals = pd.DataFrame(intervals)
    
    if verbose:
        print(f"✅ Interval construction completed: {len(df_intervals)} mothers")
        print(f"\n📊 Censoring type distribution:")
        censoring_counts = df_intervals['censor_type'].value_counts()
        for censor_type, count in censoring_counts.items():
            print(f"  {censor_type}: {count} ({100*count/len(df_intervals):.1f}%)")
    
    return df_intervals


def prepare_feature_matrix(df_intervals: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Prepare feature matrix for AFT modeling with standardization and quality checks.
    
    Args:
        df_intervals: DataFrame with interval-censored data
        verbose: Whether to print detailed progress information
        
    Returns:
        DataFrame with standardized features and valid intervals
    """
    df_X = df_intervals.copy()
    
    if verbose:
        print("🔧 Preparing feature matrix for AFT modeling...")
    
    # Standardize BMI for modeling
    bmi_mean = df_X['bmi'].mean()
    bmi_std = df_X['bmi'].std()
    df_X['bmi_z'] = (df_X['bmi'] - bmi_mean) / bmi_std
    
    # Store standardization parameters
    df_X.attrs['bmi_mean'] = bmi_mean
    df_X.attrs['bmi_std'] = bmi_std
    
    if verbose:
        print(f"✅ BMI standardization completed:")
        print(f"  Mean BMI: {bmi_mean:.2f}")
        print(f"  Std BMI: {bmi_std:.2f}")
        print(f"  BMI range: {df_X['bmi'].min():.1f} - {df_X['bmi'].max():.1f}")
    
    # Verify intervals are valid (L < R for finite intervals)
    finite_intervals = df_X[df_X['R'] != np.inf]
    invalid_intervals = finite_intervals[finite_intervals['L'] >= finite_intervals['R']]
    
    if len(invalid_intervals) > 0:
        if verbose:
            print(f"⚠️  Found {len(invalid_intervals)} invalid intervals (L >= R)")
            print("Fixing invalid intervals...")
        # Fix invalid intervals by setting L = R - 0.1 (small interval)
        mask = df_X['L'] >= df_X['R']
        df_X.loc[mask, 'L'] = df_X.loc[mask, 'R'] - 0.1
    else:
        if verbose:
            print("✅ All intervals are valid (L < R)")
    
    if verbose:
        print(f"\n📊 Feature matrix summary:")
        print(f"  Observations: {len(df_X)}")
        print(f"  Features: bmi, bmi_z")
        print(f"  Interval bounds: L, R")
        print(f"  Censoring types: {df_X['censor_type'].unique()}")
        
        # Display basic statistics
        print(f"\n📈 Dataset statistics:")
        print(df_X[['bmi', 'bmi_z', 'L', 'R']].describe())
        
        # Quality checks
        print("\n🔍 Quality checks:")
        missing_counts = df_X[['bmi', 'bmi_z', 'L', 'R']].isnull().sum()
        print(f"  Missing values: {missing_counts.sum()} total")
        if missing_counts.sum() > 0:
            print(missing_counts[missing_counts > 0])
        
        # Check interval validity
        print(f"  Interval validity check:")
        print(f"    Finite intervals: {len(df_X[df_X['R'] != np.inf])}")
        print(f"    Right-censored: {len(df_X[df_X['R'] == np.inf])}")
        print(f"    Left-censored: {len(df_X[df_X['L'] == 0])}")
        
        # Summary by censoring type
        print(f"\n📊 Summary by censoring type:")
        censoring_summary = df_X.groupby('censor_type').agg({
            'bmi': ['count', 'mean', 'std'],
            'L': 'mean',
            'R': lambda x: x[x != np.inf].mean() if len(x[x != np.inf]) > 0 else np.nan
        }).round(2)
        print(censoring_summary)
        
        print(f"\n✅ Feature matrix preparation completed - ready for AFT modeling")
    
    return df_X


def create_simple_time_to_event_data(df_tests: pd.DataFrame, threshold: float = 0.04) -> pd.DataFrame:
    """
    Create simple time-to-event dataset (one row per mother) for standard survival analysis.
    
    Args:
        df_tests: DataFrame with individual test records
        threshold: Y-chromosome concentration threshold (default 0.04 = 4%)
        
    Returns:
        DataFrame with time-to-event format (one row per mother)
    """
    def process_mother(mother_group):
        """Process data for one mother to extract time-to-event information"""
        # Sort by gestational weeks
        mother_group = mother_group.sort_values('gestational_weeks')
        
        # Get consistent BMI (should be same across tests)
        bmi = mother_group['bmi'].iloc[0]
        
        # Find first crossing of threshold
        crossed_tests = mother_group[mother_group['y_concentration'] >= threshold]
        
        if len(crossed_tests) > 0:
            # Event occurred - use time of first crossing
            time_to_event = crossed_tests.iloc[0]['gestational_weeks']
            event = 1
            status = 'event'
        else:
            # Censored - use time of last test
            time_to_event = mother_group.iloc[-1]['gestational_weeks']
            event = 0
            status = 'censored'
        
        # Additional information
        n_tests = len(mother_group)
        first_test_week = mother_group.iloc[0]['gestational_weeks']
        last_test_week = mother_group.iloc[-1]['gestational_weeks']
        max_y_concentration = mother_group['y_concentration'].max()
        
        return pd.Series({
            'maternal_id': mother_group.iloc[0]['maternal_id'],
            'bmi': bmi,
            'time_to_event': time_to_event,
            'event': event,
            'status': status,
            'n_tests': n_tests,
            'first_test_week': first_test_week,
            'last_test_week': last_test_week,
            'max_y_concentration': max_y_concentration,
            'threshold_pct': threshold * 100,
            'analysis_type': 'simple_time_to_event'
        })
    
    # Process all mothers
    survival_df = df_tests.groupby('maternal_id').apply(process_mother).reset_index(drop=True)
    
    # Add BMI quartiles for analysis
    survival_df['bmi_quartile'] = pd.qcut(survival_df['bmi'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    
    return survival_df


def visualize_preprocessing_results(df_tests: pd.DataFrame, 
                                  df_intervals: pd.DataFrame,
                                  survival_df: Optional[pd.DataFrame] = None,
                                  threshold: float = 0.04,
                                  output_path: Optional[Path] = None) -> plt.Figure:
    """
    Create comprehensive visualization of preprocessing results.
    
    Args:
        df_tests: Individual test records DataFrame
        df_intervals: Interval-censored DataFrame
        survival_df: Simple time-to-event DataFrame (optional)
        threshold: Y-chromosome concentration threshold
        output_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure object
    """
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Y-concentration distribution with threshold
    axes[0,0].hist(df_tests['y_concentration'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0,0].axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'{threshold*100}% threshold')
    axes[0,0].set_xlabel('Y-Chromosome Concentration')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].set_title('Y-Chromosome Concentration Distribution')
    axes[0,0].legend()
    
    # 2. BMI vs gestational weeks scatter
    colors = ['green' if conc >= threshold else 'red' for conc in df_tests['y_concentration']]
    axes[0,1].scatter(df_tests['bmi'], df_tests['gestational_weeks'], c=colors, alpha=0.6)
    axes[0,1].set_xlabel('BMI')
    axes[0,1].set_ylabel('Gestational Weeks')
    axes[0,1].set_title('BMI vs Gestational Weeks\n(Green: Y≥4%, Red: Y<4%)')
    
    # 3. Censoring type distribution (interval-censored)
    censor_counts = df_intervals['censor_type'].value_counts()
    colors_censor = ['lightcoral', 'lightblue', 'lightgreen']
    axes[0,2].pie(censor_counts.values, labels=censor_counts.index, autopct='%1.1f%%', 
                  colors=colors_censor, startangle=90)
    axes[0,2].set_title(f'Censoring Types\n(n={len(df_intervals)} mothers)')
    
    # 4. Number of tests per mother
    n_tests_per_mother = df_tests.groupby('maternal_id').size()
    axes[1,0].hist(n_tests_per_mother, bins=range(1, n_tests_per_mother.max()+2), 
                   alpha=0.7, color='orange', edgecolor='black')
    axes[1,0].set_xlabel('Number of Tests per Mother')
    axes[1,0].set_ylabel('Frequency')
    axes[1,0].set_title('Tests per Mother Distribution')
    
    # 5. Time-to-event analysis (if available)
    if survival_df is not None:
        # Event status distribution
        event_counts = survival_df['status'].value_counts()
        axes[1,1].bar(event_counts.index, event_counts.values, alpha=0.7, color=['lightcoral', 'lightgreen'])
        axes[1,1].set_ylabel('Count')
        axes[1,1].set_title('Event Status Distribution')
        for i, v in enumerate(event_counts.values):
            axes[1,1].text(i, v + 1, str(v), ha='center', va='bottom')
        
        # Event rate by BMI quartiles
        event_rate_by_quartile = survival_df.groupby('bmi_quartile')['event'].agg(['mean', 'count']).reset_index()
        event_rate_by_quartile.columns = ['BMI_Quartile', 'Event_Rate', 'Count']
        
        bars = axes[1,2].bar(event_rate_by_quartile['BMI_Quartile'], event_rate_by_quartile['Event_Rate'], 
                            alpha=0.7, color='purple')
        axes[1,2].set_ylabel('Event Rate')
        axes[1,2].set_title('Event Rate by BMI Quartile')
        axes[1,2].set_ylim(0, 1)
        
        # Add count labels on bars
        for bar, count in zip(bars, event_rate_by_quartile['Count']):
            height = bar.get_height()
            axes[1,2].text(bar.get_x() + bar.get_width()/2., height + 0.01, f'n={count}', 
                           ha='center', va='bottom', fontsize=10)
    else:
        # BMI distribution
        axes[1,1].hist(df_intervals['bmi'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[1,1].set_xlabel('BMI')
        axes[1,1].set_ylabel('Frequency')
        axes[1,1].set_title('BMI Distribution')
        
        # Interval lengths (for finite intervals)
        finite_intervals = df_intervals[df_intervals['R'] != np.inf]
        interval_lengths = finite_intervals['R'] - finite_intervals['L']
        axes[1,2].hist(interval_lengths, bins=20, alpha=0.7, color='gold', edgecolor='black')
        axes[1,2].set_xlabel('Interval Length (weeks)')
        axes[1,2].set_ylabel('Frequency')
        axes[1,2].set_title('Interval Lengths\n(Finite intervals only)')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Preprocessing visualization saved to: {output_path}")
    
    return fig


def load_and_preprocess_data(data_path: Path, 
                           verbose: bool = True,
                           create_simple_format: bool = False,
                           visualize: bool = False,
                           output_path: Optional[Path] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Complete data loading and preprocessing pipeline with enhanced options.
    
    Args:
        data_path: Path to the Excel file with male fetus data
        verbose: Whether to print detailed progress information
        create_simple_format: Whether to also create simple time-to-event format
        visualize: Whether to create preprocessing visualizations
        output_path: Optional output directory for files and figures
        
    Returns:
        Tuple of (processed_intervals_df, preprocessing_info)
    """
    if verbose:
        print("📂 Loading and preprocessing male fetus data...")
    
    # Load original data
    male_data = pd.read_excel(data_path, sheet_name='男胎检测数据')
    
    if verbose:
        print(f"✅ Loaded {len(male_data)} rows and {len(male_data.columns)} columns")
        print(f"  Maternal IDs: {male_data['孕妇代码'].nunique()} unique mothers")
        print(f"  Total tests: {len(male_data)} test records")
    
    # Apply QC filters
    df_tests, filter_stats = apply_qc_filters(male_data, verbose=verbose)
    
    # Construct intervals
    if verbose:
        print("\n🔄 Constructing interval-censored observations...")
    df_intervals = construct_intervals(df_tests)
    
    # Prepare feature matrix
    df_X = prepare_feature_matrix(df_intervals)
    
    # Create simple time-to-event format if requested
    survival_df = None
    if create_simple_format:
        if verbose:
            print("📊 Creating simple time-to-event format...")
        survival_df = create_simple_time_to_event_data(df_tests)
        
        if verbose:
            print(f"✅ Simple format created:")
            print(f"  Events (threshold reached): {survival_df['event'].sum()}")
            print(f"  Censored (threshold not reached): {(survival_df['event'] == 0).sum()}")
            print(f"  Event rate: {100 * survival_df['event'].mean():.1f}%")
    
    # Create visualizations if requested
    if visualize and output_path:
        if verbose:
            print("📈 Creating preprocessing visualizations...")
        fig = visualize_preprocessing_results(
            df_tests, df_intervals, survival_df, 
            output_path=output_path / 'p2_preprocessing_analysis.png'
        )
    
    # Save datasets if output path provided
    if output_path and create_simple_format:
        simple_file = output_path / 'p2_simple_time_to_event.csv'
        survival_df.to_csv(simple_file, index=False)
        if verbose:
            print(f"💾 Simple time-to-event dataset saved to: {simple_file}")
    
    # Compile preprocessing info
    preprocessing_info = {
        'filter_stats': filter_stats,
        'n_mothers': df_intervals['maternal_id'].nunique(),
        'avg_tests_per_mother': len(df_tests) / df_intervals['maternal_id'].nunique(),
        'censoring_distribution': df_intervals['censor_type'].value_counts().to_dict(),
        'bmi_stats': {
            'mean': df_X.attrs['bmi_mean'],
            'std': df_X.attrs['bmi_std'],
            'range': (df_X['bmi'].min(), df_X['bmi'].max())
        },
        'simple_format_available': create_simple_format,
        'visualization_created': visualize and output_path is not None
    }
    
    if create_simple_format:
        preprocessing_info['simple_stats'] = {
            'n_events': survival_df['event'].sum(),
            'n_censored': (survival_df['event'] == 0).sum(),
            'event_rate': survival_df['event'].mean(),
            'median_time_to_event': survival_df['time_to_event'].median()
        }
    
    if verbose:
        print(f"\n✅ Data preprocessing pipeline completed!")
        
    return df_X, preprocessing_info
