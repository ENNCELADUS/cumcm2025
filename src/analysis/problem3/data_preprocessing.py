"""
Extended data preprocessing for Problem 3.

This module handles the expanded covariate set (BMI, age, height, weight)
with collinearity control, standardization, and missing value handling.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import warnings
import re

from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrix
# Enable experimental IterativeImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

# Import Problem 2 preprocessing functions for reuse
from ..problem2.data_preprocessing import construct_intervals, apply_qc_filters


def parse_gestational_weeks(week_str: str) -> float:
    """
    Convert gestational weeks from format "11w+6" to decimal weeks (11.857).
    Copied from Problem 2 for consistency.
    
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
    Copied from Problem 2 for consistency.
    
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


def apply_extended_qc_filters(df: pd.DataFrame, verbose: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply comprehensive quality control filters adapted from Problem 2 for extended covariates.
    Handles both Chinese and English column names.
    
    Filters applied:
    1. Gestational weeks: 10-25 weeks
    2. GC content: 40-60% (if available)
    3. Remove chromosome abnormalities (if available)
    4. Remove missing key variables
    5. IQR outlier filters for continuous variables
    6. Extended covariate range filters (BMI, age, height, weight)
    
    Args:
        df: DataFrame with extended covariates (Chinese or English column names)
        verbose: Whether to print detailed progress information
        
    Returns:
        Tuple of (filtered_df, filter_stats_dict)
    """
    initial_count = len(df)
    filter_stats = {'initial': initial_count}
    
    if verbose:
        print("🔧 Applying comprehensive QC filters (Problem 3 extended)...")
    
    df = df.copy()
    
    # Ensure we have canonical English column names for consistent processing
    # (This function should be called after _extract_canonical_variables, but be defensive)
    expected_columns = ['gestational_weeks', 'bmi', 'age', 'height', 'weight', 'y_concentration', 'gc_content']
    available_columns = [col for col in expected_columns if col in df.columns]
    
    if verbose:
        print(f"   📊 Available columns for QC: {available_columns}")
    
    # Parse and validate core variables
    if 'gestational_weeks' in df.columns:
        if df['gestational_weeks'].dtype == 'object':
            df['gestational_weeks'] = df['gestational_weeks'].apply(parse_gestational_weeks)
    
    # Ensure numeric types for all continuous variables
    continuous_vars = ['bmi', 'age', 'height', 'weight', 'y_concentration', 'gc_content']
    for var in continuous_vars:
        if var in df.columns:
            df[var] = pd.to_numeric(df[var], errors='coerce')
    
    if verbose:
        print("✅ Variable parsing completed")
        for var in ['gestational_weeks', 'bmi', 'age', 'y_concentration']:
            if var in df.columns:
                valid_count = df[var].notna().sum()
                print(f"  {var}: {valid_count}/{len(df)} valid ({100*valid_count/len(df):.1f}%)")
    
    # 1. Gestational weeks: 10-25 weeks (standard obstetric range for NIPT)
    if 'gestational_weeks' in df.columns:
        df = df[(df['gestational_weeks'] >= 10) & (df['gestational_weeks'] <= 25)]
        filter_stats['after_gestational_weeks'] = len(df)
        if verbose:
            print(f"  After gestational weeks filter (10-25): {len(df)} records")
    
    # 2. GC content: 40-60% (sequencing quality control)
    if 'gc_content' in df.columns:
        df = df[(df['gc_content'] >= 0.40) & (df['gc_content'] <= 0.60)]
        filter_stats['after_gc_content'] = len(df)
        if verbose:
            print(f"  After GC content filter (40-60%): {len(df)} records")
    
    # 3. Remove chromosome abnormalities (if column exists)
    if 'aneuploidy' in df.columns:
        df = df[df['aneuploidy'].isna()]
        filter_stats['after_aneuploidy'] = len(df)
        if verbose:
            print(f"  After chromosome abnormality filter: {len(df)} records")
    
    # 4. Remove missing key variables (required canonicals only)
    required_vars = ['maternal_id', 'gestational_weeks', 'y_concentration']
    available_required = [var for var in required_vars if var in df.columns]
    if available_required:
        df = df.dropna(subset=available_required)
        filter_stats['after_missing_required'] = len(df)
        if verbose:
            print(f"  After missing required data filter: {len(df)} records")
    
    # 5. Extended covariate range filters (height/weight excluded due to BMI collinearity)
    if verbose:
        print("🎯 Applying extended covariate range filters...")
    
    # BMI: reasonable range 15-50
    if 'bmi' in df.columns:
        before_bmi = len(df)
        df = df[(df['bmi'].isna()) | ((df['bmi'] >= 15) & (df['bmi'] <= 50))]
        filter_stats['after_bmi_range'] = len(df)
        if verbose:
            print(f"  BMI range filter (15-50): {len(df)} records (removed {before_bmi - len(df)})")
    
    # Age: reproductive age range 15-50
    if 'age' in df.columns:
        before_age = len(df)
        df = df[(df['age'].isna()) | ((df['age'] >= 15) & (df['age'] <= 50))]
        filter_stats['after_age_range'] = len(df)
        if verbose:
            print(f"  Age range filter (15-50): {len(df)} records (removed {before_age - len(df)})")
    
    # Note: Height and weight filters removed - these variables excluded due to BMI multicollinearity
    
    # 6. IQR outlier filters for all continuous variables
    if verbose:
        print("🎯 Applying IQR outlier detection...")
    
    iqr_vars = ['gestational_weeks', 'bmi', 'age', 'y_concentration']
    # Note: height and weight excluded from IQR filtering due to BMI multicollinearity
    
    for col in iqr_vars:
        if col in df.columns:
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
                print(f"    IQR bounds: [{lower_bound:.3f}, {upper_bound:.3f}]")
                print(f"    Outliers: {outliers_mask.sum()} ({100*outliers_mask.mean():.2f}%)")
            
            df = remove_outliers_iqr(df, col)
            filter_stats[f'after_iqr_{col}'] = len(df)
            if verbose:
                print(f"    After IQR filtering: {len(df)} records (removed {df_before - len(df)})")
    
    # Final cleanup and sorting
    if 'maternal_id' in df.columns and 'gestational_weeks' in df.columns:
        df = df.sort_values(['maternal_id', 'gestational_weeks']).reset_index(drop=True)
    
    filter_stats['final'] = len(df)
    filter_stats['retention_rate'] = len(df) / initial_count
    
    if verbose:
        print(f"\n✅ Extended QC filtering completed: {len(df)} records remaining")
        print(f"   📊 Retention rate: {100*len(df)/initial_count:.1f}%")
        if 'maternal_id' in df.columns:
            print(f"   👥 Unique mothers: {df['maternal_id'].nunique()}")
            print(f"   📈 Tests per mother: {len(df) / df['maternal_id'].nunique():.1f} average")
    
    return df, filter_stats


def comprehensive_data_preprocessing(data_path: Union[str, Path], 
                                   verbose: bool = True,
                                   sheet_name: Union[str, List[str]] = "男胎检测数据") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Comprehensive data preprocessing for Problem 3 with extended covariates.
    Handles raw Excel data with Chinese column names.
    
    Implements the full preprocessing pipeline:
    1) Inclusion/canonical variables: {t_ij, Y_ij, BMI_i, age_i} required, {height_i, weight_i} optional
    2) De-duplication & quality flags
    3) Missingness handling (MICE/RF-impute covariates, never outcomes)
    4) Standardization (center/scale continuous covariates)
    
    Args:
        data_path: Path to raw Excel data file with Chinese column names
        verbose: Whether to print progress messages
        sheet_name: Excel sheet name(s) to read. Options:
                   - "男胎检测数据" (male fetus data, default)
                   - "女胎检测数据" (female fetus data)  
                   - ["男胎检测数据", "女胎检测数据"] (combine both)
        
    Returns:
        Tuple of (processed_dataframe, preprocessing_metadata)
    """
    if verbose:
        print("🔄 Starting comprehensive data preprocessing for Problem 3...")
        print("📂 Reading raw Excel data with Chinese column names...")
        print(f"   🎯 Target sheet(s): {sheet_name}")
    
    # Step 1: Load raw Excel data directly (handle single sheet or multiple sheets)
    if isinstance(sheet_name, list):
        # Load and combine multiple sheets
        raw_data_list = []
        for sheet in sheet_name:
            try:
                sheet_data = pd.read_excel(data_path, sheet_name=sheet)
                sheet_data['data_source'] = sheet  # Track data source
                raw_data_list.append(sheet_data)
                if verbose:
                    print(f"   ✅ Loaded sheet '{sheet}': {sheet_data.shape}")
            except Exception as e:
                if verbose:
                    print(f"   ⚠️  Failed to load sheet '{sheet}': {e}")
        
        if not raw_data_list:
            raise ValueError(f"Could not load any sheets from {sheet_name}")
        
        # Combine all sheets
        raw_data = pd.concat(raw_data_list, ignore_index=True, sort=False)
        if verbose:
            print(f"   ✅ Combined data: {raw_data.shape}")
            print(f"   📊 Data sources: {raw_data['data_source'].value_counts().to_dict()}")
    else:
        # Load single sheet
        try:
            raw_data = pd.read_excel(data_path, sheet_name=sheet_name)
            if verbose:
                print(f"   ✅ Loaded raw data: {raw_data.shape}")
                print(f"   📊 Sample columns: {list(raw_data.columns[:5])}...")
        except Exception as e:
            if verbose:
                print(f"   ❌ Failed to load Excel file: {e}")
                print(f"   🔄 Trying alternative sheet names...")
            # Try common Chinese sheet names
            for alt_sheet in ["男胎检测数据", "女胎检测数据", "数据", "Data", "Sheet1", 0]:
                try:
                    raw_data = pd.read_excel(data_path, sheet_name=alt_sheet)
                    if verbose:
                        print(f"   ✅ Successfully loaded using sheet: '{alt_sheet}'")
                    break
                except:
                    continue
            else:
                raise ValueError(f"Could not load data from {data_path} with any common sheet names")
    
    # Step 2: Extract canonical variables with Problem 3 requirements (handles Chinese mapping)
    canonical_data, inclusion_stats = _extract_canonical_variables(raw_data, verbose)
    
    # Step 3: De-duplication and quality control
    deduplicated_data, quality_stats = _apply_quality_control(canonical_data, verbose)
    
    # Step 4: Handle missingness (covariates only, never outcomes)
    imputed_data, missingness_stats = _handle_missingness_problem3(deduplicated_data, verbose)
    
    # Step 5: Create engineered features
    engineered_data, feature_stats = _create_engineered_features(imputed_data, verbose)
    
    # Step 6: Standardization of continuous covariates (including engineered features)
    standardized_data, standardization_stats = _apply_standardization(engineered_data, verbose)
    
    # Compile preprocessing metadata (height/weight excluded due to BMI multicollinearity)
    all_original_covariates = ['bmi', 'age'] + inclusion_stats['available_extended']
    available_covariates = [col for col in all_original_covariates if col in standardized_data.columns]
    standardized_covariates = [col for col in standardized_data.columns if col.endswith('_std')]
    engineered_features = feature_stats.get('engineered_features_created', [])
    
    preprocessing_metadata = {
        'inclusion_stats': inclusion_stats,
        'quality_stats': quality_stats,
        'missingness_stats': missingness_stats,
        'feature_engineering_stats': feature_stats,
        'standardization_stats': standardization_stats,
        'final_shape': standardized_data.shape,
        'available_covariates': available_covariates,
        'standardized_covariates': standardized_covariates,
        'engineered_features': engineered_features
    }
    
    if verbose:
        print(f"\n✅ Comprehensive preprocessing complete!")
        print(f"📊 Final data shape: {standardized_data.shape}")
        print(f"🎯 Available covariates: {preprocessing_metadata['available_covariates']}")
        print(f"📈 Standardized variables: {preprocessing_metadata['standardized_covariates']}")
    
    return standardized_data, preprocessing_metadata


def _extract_canonical_variables(df: pd.DataFrame, verbose: bool) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Step 1: Extract canonical variables {t_ij, Y_ij, BMI_i, age_i} + optional {height_i, weight_i}
    Parse original Chinese column names to canonical English names.
    """
    if verbose:
        print("📋 Step 1: Extracting canonical variables from Chinese column names...")
    
    # Chinese to English column mapping based on ACTUAL Excel column names
    chinese_column_mapping = {
        # Core required variables
        '孕妇代码': 'maternal_id',                    # Maternal ID
        '检测孕周': 'gestational_weeks',              # Gestational weeks at testing  
        'Y染色体浓度': 'y_concentration',             # Y chromosome concentration
        '孕妇BMI': 'bmi',                          # Maternal BMI
        '年龄': 'age',                             # Maternal age
        
        # Optional extended covariates
        '身高': 'height',                           # Maternal height
        '体重': 'weight',                           # Maternal weight
        
        # Quality control variables
        'GC含量': 'gc_content',                     # GC content
        '染色体的非整倍体': 'aneuploidy',              # Aneuploidy detection
        
        # Additional useful variables for QC
        '序号': 'sample_id',                        # Sample sequence number
        '检测日期': 'test_date',                     # Detection date
        '检测抽血次数': 'blood_draw_count',            # Blood draw count
        '末次月经': 'last_menstrual_period',          # Last menstrual period
        'IVF妊娠': 'ivf_pregnancy',                 # IVF pregnancy method
        '怀孕次数': 'pregnancy_count',               # Pregnancy count
        '生产次数': 'birth_count',                   # Birth count
        '胎儿是否健康': 'fetal_health',               # Fetal health
        
        # Additional sequencing QC variables
        '原始读段数': 'raw_read_count',               # Raw read count
        '在参考基因组上比对的比例': 'mapping_ratio',      # Mapping ratio to reference genome
        '重复读段的比例': 'duplicate_ratio',           # Duplicate read ratio
        '唯一比对的读段数  ': 'unique_mapped_reads',   # Unique mapped reads (note: has extra spaces)
        '被过滤掉读段数的比例': 'filtered_reads_ratio', # Filtered reads ratio
        
        # Chromosome-specific variables
        '13号染色体的Z值': 'chr13_z_value',           # Chr13 Z-value
        '18号染色体的Z值': 'chr18_z_value',           # Chr18 Z-value  
        '21号染色体的Z值': 'chr21_z_value',           # Chr21 Z-value
        'X染色体的Z值': 'x_chr_z_value',             # X chromosome Z-value
        'Y染色体的Z值': 'y_chr_z_value',             # Y chromosome Z-value
        'X染色体浓度': 'x_chr_concentration',         # X chromosome concentration
        '13号染色体的GC含量': 'chr13_gc_content',      # Chr13 GC content
        '18号染色体的GC含量': 'chr18_gc_content',      # Chr18 GC content
        '21号染色体的GC含量': 'chr21_gc_content'       # Chr21 GC content
    }
    
    # Alternative English column names (in case data is already processed)
    english_column_mapping = {
        'patient_code': 'maternal_id',
        'y_chr_concentration': 'y_concentration', 
        'gestational_weeks': 'gestational_weeks',
        'bmi': 'bmi',
        'age': 'age',
        'height': 'height',
        'weight': 'weight',
        'gc_content': 'gc_content',
        'aneuploidy': 'aneuploidy'
    }
    
    # Combined mapping
    full_column_mapping = {**chinese_column_mapping, **english_column_mapping}
    
    # Core required variables (canonical)
    required_vars = ['maternal_id', 'gestational_weeks', 'y_concentration', 'bmi', 'age']
    
    # Extended original covariates (exclude height/weight due to BMI collinearity)
    extended_original_vars = [
        # Note: height and weight excluded as BMI = weight/(height^2) creates severe multicollinearity
        'ivf_pregnancy', 'pregnancy_count', 'birth_count',  # Pregnancy history
        'raw_read_count', 'unique_mapped_reads', 'mapping_ratio',  # Sequencing metrics
        'duplicate_ratio', 'filtered_reads_ratio', 'gc_content'  # Quality metrics
    ]
    
    # Apply column mapping
    df_canonical = df.copy()
    
    if verbose:
        print(f"   📊 Original columns: {list(df.columns)}")
    
    # Apply all mappings
    for old_col, new_col in full_column_mapping.items():
        if old_col in df_canonical.columns and new_col not in df_canonical.columns:
            df_canonical[new_col] = df_canonical[old_col]
            if verbose and old_col in chinese_column_mapping:
                print(f"   🔄 Mapped '{old_col}' -> '{new_col}'")
    
    # Special handling for gestational weeks parsing (if it's in string format)
    if 'gestational_weeks' in df_canonical.columns:
        if df_canonical['gestational_weeks'].dtype == 'object':
            if verbose:
                print("   🔧 Parsing gestational weeks from string format...")
            df_canonical['gestational_weeks'] = df_canonical['gestational_weeks'].apply(parse_gestational_weeks)
    
    # Check availability across all variable types
    available_required = [var for var in required_vars if var in df_canonical.columns]
    missing_required = [var for var in required_vars if var not in df_canonical.columns]
    available_extended = [var for var in extended_original_vars if var in df_canonical.columns]
    missing_extended = [var for var in extended_original_vars if var not in df_canonical.columns]
    
    # Record unique patients and repeated measures
    n_unique_patients = df_canonical['maternal_id'].nunique() if 'maternal_id' in df_canonical.columns else 0
    n_total_records = len(df_canonical)
    
    inclusion_stats = {
        'available_required': available_required,
        'missing_required': missing_required,
        'available_extended': available_extended,
        'missing_extended': missing_extended,
        'n_unique_patients': n_unique_patients,
        'n_total_records': n_total_records,
        'repeated_measures_ratio': n_total_records / n_unique_patients if n_unique_patients > 0 else 0,
        'column_mapping_applied': len([col for col in chinese_column_mapping.keys() if col in df.columns])
    }
    
    # Keep all mapped variables including QC and identification variables
    all_useful_vars = required_vars + available_extended + ['aneuploidy', 'sample_id', 'test_date']
    keep_cols = [col for col in all_useful_vars if col in df_canonical.columns]
    
    df_result = df_canonical[keep_cols].copy()
    
    if verbose:
        print(f"   ✅ Required variables: {len(available_required)}/{len(required_vars)} - {available_required}")
        print(f"   📊 Extended variables: {len(available_extended)}/{len(extended_original_vars)} - {available_extended}")
        print(f"   📊 Unique patients: {n_unique_patients}, Total records: {n_total_records}")
        print(f"   🔄 Chinese columns mapped: {inclusion_stats['column_mapping_applied']}")
    if missing_required:
        print(f"   ⚠️  Missing required: {missing_required}")
    if missing_extended:
        print(f"   📝 Missing extended: {missing_extended}")
    
    return df_result, inclusion_stats


def _apply_quality_control(df: pd.DataFrame, verbose: bool) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Step 2: De-duplication and comprehensive quality control using Problem 2 methods
    """
    if verbose:
        print("🔍 Step 2: Quality control and de-duplication...")
    
    initial_count = len(df)
    
    # Remove exact duplicates first
    df_dedup = df.drop_duplicates()
    n_duplicates_removed = initial_count - len(df_dedup)
    
    # Apply comprehensive quality control filters from Problem 2
    df_filtered, qc_filter_stats = apply_extended_qc_filters(df_dedup, verbose=verbose)
    
    # Additional quality flags for outcome variable (never filter these out)
    quality_flags = {}
    if 'y_concentration' in df_filtered.columns:
        # Flag extremely low values (potential assay failures) for sensitivity analysis
        low_threshold = df_filtered['y_concentration'].quantile(0.01)
        quality_flags['low_y_concentration'] = df_filtered['y_concentration'] < low_threshold
        
        # Flag extremely high values (potential contamination) for sensitivity analysis  
        high_threshold = df_filtered['y_concentration'].quantile(0.99)
        quality_flags['high_y_concentration'] = df_filtered['y_concentration'] > high_threshold
        
        # Overall quality flag (keep all records, just flag for sensitivity)
        df_filtered['quality_flag'] = ~(quality_flags['low_y_concentration'] | quality_flags['high_y_concentration'])
    
    # Combine all quality statistics
    quality_stats = {
        'n_duplicates_removed': n_duplicates_removed,
        'n_after_dedup': len(df_dedup),
        'qc_filter_stats': qc_filter_stats,
        'quality_flags_created': list(quality_flags.keys()),
        'n_quality_flagged': sum(df_filtered['quality_flag']) if 'quality_flag' in df_filtered.columns else 0,
        'final_after_qc': len(df_filtered)
    }
    
    if verbose:
        print(f"\n🔍 Step 2 Summary:")
        print(f"   🗑️  Removed {n_duplicates_removed} duplicate records")
        print(f"   🔧 Applied {len(qc_filter_stats)} QC filter stages")
        print(f"   📊 Final retention: {len(df_filtered)}/{initial_count} ({100*len(df_filtered)/initial_count:.1f}%)")
        if quality_flags:
            print(f"   🏷️  Quality flags: {len(quality_flags)} created for sensitivity analysis")
    
    return df_filtered, quality_stats


def _handle_missingness_problem3(df: pd.DataFrame, verbose: bool) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Step 3: Handle missingness - impute covariates, never outcomes
    """
    if verbose:
        print("🔧 Step 3: Handling missingness (covariates only)...")
    
    # Identify covariate vs outcome columns
    outcome_cols = ['y_concentration']
    covariate_cols = ['bmi', 'age', 'height', 'weight']
    available_covariates = [col for col in covariate_cols if col in df.columns]
    
    # Check missingness patterns
    missing_stats = {}
    for col in df.columns:
        n_missing = df[col].isnull().sum()
        missing_stats[col] = {
            'n_missing': n_missing,
            'pct_missing': n_missing / len(df) * 100
        }
    
    df_result = df.copy()
    
    # Impute covariates only (never outcomes)
    if available_covariates:
        covariates_to_impute = [col for col in available_covariates if df[col].isnull().any()]
        
        if covariates_to_impute:
            if verbose:
                print(f"   🔄 Imputing covariates: {covariates_to_impute}")
            
            # Use IterativeImputer (MICE) for covariates
            imputer = IterativeImputer(
                estimator=RandomForestRegressor(n_estimators=10, random_state=42),
                max_iter=10,
                random_state=42
            )
            
            # Fit and transform only the covariates
            df_covariates = df[available_covariates].copy()
            df_imputed_covariates = pd.DataFrame(
                imputer.fit_transform(df_covariates),
                columns=available_covariates,
                index=df_covariates.index
            )
            
            # Replace imputed covariates in result
            for col in available_covariates:
                df_result[col] = df_imputed_covariates[col]
    
    # Compile missingness statistics
    missingness_stats = {
        'original_missing_stats': missing_stats,
        'covariates_imputed': covariates_to_impute if 'covariates_to_impute' in locals() else [],
        'outcomes_never_imputed': outcome_cols,
        'final_missing_outcomes': df_result[outcome_cols].isnull().sum().to_dict() if outcome_cols[0] in df_result.columns else {}
    }
    
    if verbose:
        n_imputed = len(covariates_to_impute) if 'covariates_to_impute' in locals() else 0
        print(f"   ✅ Imputed {n_imputed} covariate columns")
        print(f"   🚫 Never impute outcomes: {outcome_cols}")
    
    return df_result, missingness_stats


def _apply_standardization(df: pd.DataFrame, verbose: bool) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Step 6: Standardization - center/scale continuous covariates including engineered features
    """
    if verbose:
        print("📏 Step 6: Standardizing continuous covariates and engineered features...")
    
    # Basic continuous variables (height/weight excluded due to BMI collinearity)
    basic_continuous_vars = ['bmi', 'age']
    
    # Extended original continuous variables
    extended_continuous_vars = ['raw_read_count', 'unique_mapped_reads', 'mapping_ratio', 
                               'duplicate_ratio', 'filtered_reads_ratio', 'gc_content']
    
    # Engineered continuous features (identify automatically)
    engineered_continuous_vars = [
        'log_unique_reads', 'seq_quality_score', 'prior_y_conc', 
        'slope_y_conc', 'bmi_weeks_interaction'
    ]
    
    # Gestational spline features (identify automatically)
    spline_vars = [col for col in df.columns if col.startswith('gest_week_spline_')]
    
    # Combine all continuous variables
    all_continuous_vars = basic_continuous_vars + extended_continuous_vars + engineered_continuous_vars + spline_vars
    available_continuous = [var for var in all_continuous_vars if var in df.columns and df[var].dtype in ['float64', 'int64']]
    
    df_result = df.copy()
    standardization_stats = {}
    
    if verbose:
        print(f"   📊 Found {len(available_continuous)} continuous variables to standardize:")
        print(f"      Basic: {[v for v in basic_continuous_vars if v in available_continuous]}")
        print(f"      Extended: {[v for v in extended_continuous_vars if v in available_continuous]}")
        print(f"      Engineered: {[v for v in engineered_continuous_vars if v in available_continuous]}")
        print(f"      Splines: {[v for v in spline_vars if v in available_continuous]}")
    
    for var in available_continuous:
        if df[var].notna().sum() > 1 and df[var].std() > 1e-10:  # Only standardize non-constant variables with data
            mean_val = df[var].mean()
            std_val = df[var].std()
            
            # Apply standardization: x_std = (x - mean) / std
            df_result[f'{var}_std'] = (df[var] - mean_val) / std_val
            
            standardization_stats[var] = {
                'original_mean': mean_val,
                'original_std': std_val,
                'standardized_mean': df_result[f'{var}_std'].mean(),
                'standardized_std': df_result[f'{var}_std'].std()
            }
    
    # Verify standardization
    standardized_cols = [col for col in df_result.columns if col.endswith('_std')]
    if standardized_cols:
        verification = {
            'means_near_zero': np.allclose([df_result[col].mean() for col in standardized_cols], 0, atol=1e-10),
            'stds_near_one': np.allclose([df_result[col].std() for col in standardized_cols], 1, atol=1e-10)
        }
    else:
        verification = {'means_near_zero': True, 'stds_near_one': True}
    
    standardization_stats['verification'] = verification
    standardization_stats['standardized_variables'] = standardized_cols
    
    if verbose:
        print(f"   📈 Successfully standardized {len(standardized_cols)} variables")
        print(f"   ✅ Verification - Means≈0: {verification['means_near_zero']}, Stds≈1: {verification['stds_near_one']}")
    
    return df_result, standardization_stats


def _create_engineered_features(df: pd.DataFrame, verbose: bool) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Step 5: Create engineered features as specified in implementation guide
    """
    if verbose:
        print("🔧 Step 5: Creating engineered features...")
    
    df_result = df.copy()
    feature_stats = {}
    
    # 1. BMI category variable
    if 'bmi' in df.columns:
        if verbose:
            print("   📊 Creating BMI category variable...")
        df_result['bmi_cat'] = pd.cut(
            df['bmi'], 
            bins=[0, 25, 30, 35, 40, float('inf')],
            labels=[0, 1, 2, 3, 4],
            right=False
        ).astype(int)
        feature_stats['bmi_cat_distribution'] = df_result['bmi_cat'].value_counts().to_dict()
    
    # 2. Spline basis of gestational weeks
    if 'gestational_weeks' in df.columns:
        if verbose:
            print("   🌊 Creating gestational weeks spline basis...")
        try:
            # Create cubic spline with 4 degrees of freedom
            from patsy import dmatrix
            spline_formula = "bs(gestational_weeks, df=4, include_intercept=False)"
            spline_matrix = dmatrix(spline_formula, df_result)
            
            # Add spline columns
            for i in range(spline_matrix.shape[1]):
                df_result[f'gest_week_spline_{i+1}'] = spline_matrix[:, i]
            
            feature_stats['gestational_splines_created'] = spline_matrix.shape[1]
        except Exception as e:
            if verbose:
                print(f"   ⚠️  Spline creation failed: {e}")
            feature_stats['gestational_splines_created'] = 0
    
    # 3. Log-transformed unique mapped reads
    if 'unique_mapped_reads' in df.columns:
        if verbose:
            print("   📈 Creating log-transformed unique reads...")
        df_result['log_unique_reads'] = np.log(df['unique_mapped_reads'] + 1)
        feature_stats['log_unique_reads_range'] = (
            df_result['log_unique_reads'].min(),
            df_result['log_unique_reads'].max()
        )
    
    # 4. Sequencing quality score
    seq_quality_vars = ['mapping_ratio', 'duplicate_ratio', 'filtered_reads_ratio']
    if all(var in df.columns for var in seq_quality_vars):
        if verbose:
            print("   🔬 Creating sequencing quality score...")
        # Quality score = mapping_ratio + (1 - duplicate_ratio) + (1 - filtered_reads_ratio)
        df_result['seq_quality_score'] = (
            df['mapping_ratio'] + 
            (1 - df['duplicate_ratio']) + 
            (1 - df['filtered_reads_ratio'])
        )
        feature_stats['seq_quality_score_range'] = (
            df_result['seq_quality_score'].min(),
            df_result['seq_quality_score'].max()
        )
    
    # 5. Previous Y-concentration (lag feature) - requires sorting by patient and time
    if all(var in df.columns for var in ['maternal_id', 'gestational_weeks', 'y_concentration']):
        if verbose:
            print("   ⏮️  Creating previous Y-concentration lag feature...")
        
        # Sort by patient and gestational weeks
        df_sorted = df_result.sort_values(['maternal_id', 'gestational_weeks'])
        df_sorted['prior_y_conc'] = df_sorted.groupby('maternal_id')['y_concentration'].shift(1)
        
        # Merge back to original order
        df_result['prior_y_conc'] = df_sorted.set_index(df_result.index)['prior_y_conc']
        
        feature_stats['prior_y_conc_available'] = df_result['prior_y_conc'].notna().sum()
    
    # 6. Slope of Y-concentration (within-patient trend)
    if all(var in df.columns for var in ['maternal_id', 'gestational_weeks', 'y_concentration']):
        if verbose:
            print("   📈 Creating Y-concentration slope (within-patient trend)...")
        
        def calculate_patient_slope(group):
            if len(group) < 2:
                return np.nan
            try:
                from scipy.stats import linregress
                slope, _, _, _, _ = linregress(group['gestational_weeks'], group['y_concentration'])
                return slope
            except:
                return np.nan
        
        # Calculate slope for each patient
        patient_slopes = df_result.groupby('maternal_id').apply(calculate_patient_slope)
        df_result['slope_y_conc'] = df_result['maternal_id'].map(patient_slopes)
        
        feature_stats['slope_y_conc_available'] = df_result['slope_y_conc'].notna().sum()
    
    # 7. BMI × Gestational Weeks interaction
    if all(var in df.columns for var in ['bmi', 'gestational_weeks']):
        if verbose:
            print("   🤝 Creating BMI × Gestational Weeks interaction...")
        df_result['bmi_weeks_interaction'] = df['bmi'] * df['gestational_weeks']
        feature_stats['bmi_weeks_interaction_range'] = (
            df_result['bmi_weeks_interaction'].min(),
            df_result['bmi_weeks_interaction'].max()
        )
    
    # Summary
    engineered_features = [col for col in df_result.columns if col not in df.columns]
    feature_stats['engineered_features_created'] = engineered_features
    feature_stats['n_engineered_features'] = len(engineered_features)
    
    if verbose:
        print(f"   ✅ Created {len(engineered_features)} engineered features:")
        for feat in engineered_features:
            print(f"      • {feat}")
    
    return df_result, feature_stats


def calculate_vif(df_covariates: pd.DataFrame, threshold: float = 5.0) -> pd.DataFrame:
    """
    Calculate Variance Inflation Factor for multicollinearity assessment.
    
    Args:
        df_covariates: DataFrame with continuous covariates
        threshold: VIF threshold for concern (default: 5.0)
        
    Returns:
        DataFrame with Variable and VIF columns
    """
    # Remove any constant columns
    df_clean = df_covariates.loc[:, df_covariates.std() > 0]
    
    # Handle missing values
    if df_clean.isnull().any().any():
        warnings.warn("Missing values detected in VIF calculation. Using mean imputation.")
        df_clean = df_clean.fillna(df_clean.mean())
    
    vif_data = pd.DataFrame()
    vif_data["Variable"] = df_clean.columns
    vif_data["VIF"] = [
        variance_inflation_factor(df_clean.values, i) 
        for i in range(len(df_clean.columns))
    ]
    
    # Add interpretation
    vif_data["Interpretation"] = vif_data["VIF"].apply(
        lambda x: "Severe (>10)" if x > 10 else ("High (>5)" if x > threshold else "Low (<5)")
    )
    
    print(f"📊 VIF Analysis on Final Model Covariates:")
    print(f"   Variables: {list(df_clean.columns)}")
    print(f"   VIF Results: {len(vif_data)} variables assessed")
    print(vif_data.to_string(index=False))
    
    # Check for problematic VIF values
    high_vif_vars = vif_data[vif_data['VIF'] > threshold]
    if len(high_vif_vars) > 0:
        print(f"   ⚠️  High VIF variables (>{threshold}): {list(high_vif_vars['Variable'])}")
    else:
        print(f"   ✅ All VIF values < {threshold} (acceptable)")
    
    return vif_data


def standardize_covariates_extended(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Standardize continuous covariates with z-score transformation.
    
    Args:
        df: DataFrame with covariates
        
    Returns:
        Tuple of (standardized_df, standardization_params)
    """
    # Include all continuous variables that might be used as covariates
    continuous_vars = [
        # Basic maternal characteristics
        'bmi', 'age', 'height', 'weight',
        # Sequencing quality variables
        'raw_read_count', 'unique_mapped_reads', 'mapping_ratio', 
        'duplicate_ratio', 'filtered_reads_ratio', 'gc_content',
        # Engineered features
        'log_unique_reads', 'seq_quality_score', 'prior_y_conc', 
        'slope_y_conc', 'bmi_weeks_interaction'
    ]
    available_vars = [var for var in continuous_vars if var in df.columns]
    
    df_standardized = df.copy()
    standardization_params = {}
    
    for var in available_vars:
        mean_val = df[var].mean()
        std_val = df[var].std()
        
        if std_val > 0:  # Avoid division by zero
            df_standardized[f'{var}_std'] = (df[var] - mean_val) / std_val
        standardization_params[var] = {'mean': mean_val, 'std': std_val}
    
    print(f"✅ Standardized variables: {available_vars}")
    
    return df_standardized, standardization_params


def assess_multicollinearity(df: pd.DataFrame, 
                           vif_threshold: float = 5.0) -> Tuple[List[str], Dict]:
    """
    Assess multicollinearity and select appropriate covariate set.
    
    Args:
        df: DataFrame with standardized covariates
        vif_threshold: VIF threshold for variable selection
        
    Returns:
        Tuple of (selected_variables, assessment_report)
    """
    # Core covariates (always included)
    core_vars = ['bmi_std', 'age_std']
    
    # Optional covariates (height/weight)
    optional_vars = [var for var in ['height_std', 'weight_std'] if var in df.columns]
    
    assessment_report = {
        'vif_results': {},
        'selected_set': [],
        'excluded_vars': [],
        'recommendation': ''
    }
    
    # Test different covariate combinations
    test_combinations = [
        ('core_only', core_vars),
        ('with_height_weight', core_vars + optional_vars)
    ]
    
    if optional_vars:
        test_combinations.append(('height_weight_only', ['age_std'] + optional_vars))
    
    for combo_name, var_list in test_combinations:
        if not all(var in df.columns for var in var_list):
            continue
            
        vif_df = calculate_vif(df[var_list], threshold=vif_threshold)
        assessment_report['vif_results'][combo_name] = vif_df
        
        max_vif = vif_df['VIF'].max()
        if max_vif < vif_threshold:
            assessment_report['selected_set'] = var_list
            assessment_report['recommendation'] = f"Use {combo_name} (max VIF: {max_vif:.2f})"
            break
    
    # Default to core variables if no combination passes
    if not assessment_report['selected_set']:
        assessment_report['selected_set'] = core_vars
        assessment_report['recommendation'] = "Use core variables only due to multicollinearity"
        assessment_report['excluded_vars'] = optional_vars
    
    print(f"🔍 Multicollinearity Assessment:")
    print(f"   Selected variables: {assessment_report['selected_set']}")
    print(f"   Recommendation: {assessment_report['recommendation']}")
    
    return assessment_report['selected_set'], assessment_report


def handle_missing_covariates(df: pd.DataFrame, 
                            method: str = 'iterative',
                            random_state: int = 42) -> pd.DataFrame:
    """
    Handle missing values in covariates using principled imputation.
    Never imputes outcome variables (Y_ij).
    
    Args:
        df: DataFrame with potential missing values
        method: 'iterative' (MICE) or 'rf' (Random Forest)
        random_state: Random seed for reproducibility
        
    Returns:
        DataFrame with imputed covariates
    """
    covariate_cols = [col for col in df.columns 
                     if col in ['bmi', 'age', 'height', 'weight', 'maternal_id']]
    
    # Check for missing values
    missing_counts = df[covariate_cols].isnull().sum()
    if missing_counts.sum() == 0:
        print("✅ No missing values in covariates")
        return df
    
    print(f"📊 Missing value counts:")
    print(missing_counts[missing_counts > 0])
    
    df_imputed = df.copy()
    
    if method == 'iterative':
        # MICE imputation
        imputer = IterativeImputer(
            estimator=RandomForestRegressor(n_estimators=10, random_state=random_state),
            random_state=random_state,
            max_iter=10
        )
        
        numeric_cols = ['bmi', 'age', 'height', 'weight']
        available_numeric = [col for col in numeric_cols if col in df.columns]
        
        if available_numeric:
            df_imputed[available_numeric] = imputer.fit_transform(df[available_numeric])
    
    elif method == 'rf':
        # Random Forest imputation (simplified)
        for col in covariate_cols:
            if col == 'maternal_id' or df[col].isnull().sum() == 0:
                continue
                
            missing_mask = df[col].isnull()
            if missing_mask.sum() > 0:
                # Simple mean imputation for now (can be enhanced)
                df_imputed.loc[missing_mask, col] = df[col].mean()
    
    print(f"✅ Imputation completed using {method} method")
    
    return df_imputed


def create_spline_basis(bmi_values: np.ndarray, 
                       n_knots: int = 3,
                       degree: int = 3) -> np.ndarray:
    """
    Create restricted cubic spline basis for BMI nonlinearity.
    
    Args:
        bmi_values: Standardized BMI values
        n_knots: Number of interior knots
        degree: Spline degree
        
    Returns:
        Spline basis matrix
    """
    # Place knots at quantiles
    knot_positions = np.linspace(0.1, 0.9, n_knots)
    knots = np.quantile(bmi_values, knot_positions)
    
    # Create spline basis using patsy
    data_dict = {'bmi_std': bmi_values}
    spline_formula = f"bs(bmi_std, knots={list(knots)}, degree={degree})"
    
    try:
        basis_matrix = dmatrix(spline_formula, data=data_dict, return_type='matrix')
        
        # Convert to numpy array (excluding intercept column if present)
        # Remove the first column if it's all 1s (intercept)
        if basis_matrix.shape[1] > 1 and np.allclose(basis_matrix[:, 0], 1):
            basis_array = np.asarray(basis_matrix[:, 1:])  # Skip intercept
        else:
            basis_array = np.asarray(basis_matrix)
        
        print(f"✅ Created spline basis with {n_knots} knots")
        print(f"   Basis dimensions: {basis_array.shape}")
        
        return basis_array
        
    except Exception as e:
        print(f"❌ Spline creation error: {e}")
        warnings.warn(f"Spline creation failed: {e}. Using linear BMI.")
        return None


def comprehensive_vif_assessment(df: pd.DataFrame, 
                                 preprocessing_metadata: Dict[str, Any],
                                 pca_results: Dict[str, Any] = None,  # FIXED: Accept PCA results
                                 vif_threshold: float = 5.0,
                                 verbose: bool = True) -> Tuple[List[str], Dict[str, Any]]:
    """
    Comprehensive multicollinearity assessment and covariate selection for Problem 3.
    
    This function systematically tests different covariate combinations, calculates VIF
    for each set, and selects the optimal covariate set that meets the VIF threshold.
    
    Args:
        df: Preprocessed DataFrame with standardized variables
        preprocessing_metadata: Metadata from comprehensive_data_preprocessing
        vif_threshold: Maximum acceptable VIF value (default: 5.0)
        verbose: Whether to print detailed progress information
        
    Returns:
        Tuple of (selected_covariates_list, assessment_results_dict)
    """
    if verbose:
        print("🔍 Performing comprehensive multicollinearity assessment...")
        print(f"📋 Goal: Select optimal covariate set with VIF ≤ {vif_threshold} constraint")
    
    # Step 1: Identify standardized variables available for VIF assessment
    available_covariates = preprocessing_metadata['available_covariates']
    all_standardized_cols = [col for col in df.columns if col.endswith('_std')]
    
    if verbose:
        print(f"\n📊 Standardized variables available: {len(all_standardized_cols)}")
        print(f"   Original covariates: {available_covariates}")
        print(f"   All standardized columns: {all_standardized_cols[:10]}{'...' if len(all_standardized_cols) > 10 else ''}")
    
    assessment_results = {
        'available_covariates': available_covariates,
        'all_standardized_cols': all_standardized_cols,
        'covariate_categories': {},
        'vif_tests': {},
        'selected_covariates': [],
        'selection_strategy': '',
        'final_vif_results': None
    }
    
    if not all_standardized_cols:
        if verbose:
            print("⚠️  No standardized covariates available for VIF assessment")
        assessment_results['selected_covariates'] = ['bmi', 'age']  # Fallback
        assessment_results['selection_strategy'] = 'fallback_to_original'
        return assessment_results['selected_covariates'], assessment_results
    
    # Step 2: Categorize covariates for systematic assessment
    core_standardized = ['bmi_std', 'age_std']
    available_core = [col for col in core_standardized if col in df.columns]
    
    height_weight_std = ['height_std', 'weight_std'] 
    available_hw = [col for col in height_weight_std if col in df.columns]
    
    sequencing_std = [col for col in all_standardized_cols if any(seq in col for seq in ['read', 'mapping', 'ratio', 'gc'])]
    
    engineered_std = [col for col in all_standardized_cols if any(eng in col for eng in ['spline', 'log', 'quality', 'interaction'])]
    
    assessment_results['covariate_categories'] = {
        'core': available_core,
        'height_weight': available_hw,
        'sequencing': sequencing_std,
        'engineered': engineered_std
    }
    
    if verbose:
        print(f"\n🎯 Covariate Categories:")
        print(f"   Core (BMI, age): {available_core}")
        print(f"   Height/Weight: {available_hw} (expect high VIF with BMI)")
        print(f"   Sequencing quality: {sequencing_std[:5]}{'...' if len(sequencing_std) > 5 else ''}")
        print(f"   Engineered features: {engineered_std[:3]}{'...' if len(engineered_std) > 3 else ''}")
    
    # Step 3: Test different covariate combinations (FIXED: Include PCA alternatives)
    if verbose:
        print(f"\n🔬 Testing covariate combinations for multicollinearity:")
    
    covariate_sets = {
        "Core Only (BMI + Age)": available_core,
        "Core + Sequencing": available_core + sequencing_std[:3] if sequencing_std else available_core,
        "All Available": [col for col in all_standardized_cols if df[col].notna().sum() > 10][:10]  # Limit for efficiency
    }
    
    # FIXED: Add PCA-based alternatives if available
    if pca_results is not None and 'alternative_modeling_covariates' in pca_results:
        pca_alternatives = pca_results['alternative_modeling_covariates']
        for pca_set_name, pca_covs in pca_alternatives.items():
            # Check if PCA covariates exist in dataframe
            available_pca_covs = [cov for cov in pca_covs if cov in df.columns]
            if len(available_pca_covs) >= 2:  # At least BMI + Age or BMI + PC
                covariate_sets[f"PCA_{pca_set_name}"] = available_pca_covs
                if verbose:
                    print(f"   📊 Added PCA alternative: {pca_set_name} -> {available_pca_covs}")
    
    if verbose:
        print(f"   📋 Total covariate sets to test: {len(covariate_sets)}")
    
    # Test each covariate set
    for set_name, covariate_list in covariate_sets.items():
        if not covariate_list:
            continue
            
        if verbose:
            print(f"\n   📋 Testing: {set_name}")
            print(f"      Variables: {covariate_list}")
        
        test_results = {
            'variables': covariate_list,
            'vif_results': None,
            'acceptable_vars': [],
            'high_vif_vars': [],
            'max_vif': None,
            'error': None
        }
        
        try:
            # Calculate VIF for this set
            test_data = df[covariate_list].dropna()
            if len(test_data) > len(covariate_list) and test_data.shape[1] > 1:
                vif_results = calculate_vif(test_data, threshold=vif_threshold)
                
                # Analyze VIF results
                acceptable_vars = vif_results[vif_results['VIF'] <= vif_threshold]['Variable'].tolist()
                high_vif_vars = vif_results[vif_results['VIF'] > vif_threshold]['Variable'].tolist()
                
                test_results.update({
                    'vif_results': vif_results,
                    'acceptable_vars': acceptable_vars,
                    'high_vif_vars': high_vif_vars,
                    'max_vif': vif_results['VIF'].max()
                })
                
                if verbose:
                    print(f"      ✅ Acceptable VIF (≤{vif_threshold}): {len(acceptable_vars)} variables")
                    print(f"      ⚠️  High VIF (>{vif_threshold}): {len(high_vif_vars)} variables")
                    if high_vif_vars:
                        print(f"         High VIF variables: {high_vif_vars}")
                        
            else:
                test_results['error'] = "Insufficient data for VIF assessment"
                if verbose:
                    print(f"      ⚠️  Insufficient data for VIF assessment")
                    
        except Exception as e:
            test_results['error'] = str(e)
            if verbose:
                print(f"      ❌ VIF calculation failed: {e}")
        
        assessment_results['vif_tests'][set_name] = test_results
    
    # Step 4: Select optimal covariate set using strategy
    if verbose:
        print(f"\n🎯 FINAL COVARIATE SELECTION:")
    
    selected_covariates = []
    selection_strategy = "none"
    
    # Strategy 1: Start with core variables if they pass VIF
    if "Core Only (BMI + Age)" in assessment_results['vif_tests']:
        core_test = assessment_results['vif_tests']["Core Only (BMI + Age)"]
        if core_test['acceptable_vars']:
            selected_covariates = core_test['acceptable_vars'].copy()
            selection_strategy = "core_acceptable"
            
            # Strategy 2: Add other acceptable variables (excluding height/weight due to BMI collinearity)
            for set_name, test_results in assessment_results['vif_tests'].items():
                if set_name != "Core Only (BMI + Age)" and test_results['acceptable_vars']:
                    additional_acceptable = [
                        var for var in test_results['acceptable_vars'] 
                        if var not in selected_covariates and 
                        'height' not in var and 'weight' not in var  # Exclude due to BMI collinearity
                    ]
                    selected_covariates.extend(additional_acceptable[:3])  # Limit additions
            
            # Remove duplicates while preserving order
            selected_covariates = list(dict.fromkeys(selected_covariates))
            selection_strategy = "core_plus_acceptable"
    
    # Fallback strategy: Use minimal core set
    if not selected_covariates:
        selected_covariates = available_core if available_core else ['bmi', 'age']
        selection_strategy = "fallback_core"
    
    assessment_results['selected_covariates'] = selected_covariates
    assessment_results['selection_strategy'] = selection_strategy
    
    if verbose:
        print(f"   Selected covariates: {selected_covariates}")
        print(f"   Total covariates: {len(selected_covariates)}")
        print(f"   Selection strategy: {selection_strategy}")
    
    # Step 5: Final VIF verification on selected set
    if len(selected_covariates) > 1:
        try:
            if verbose:
                print(f"\n📊 Final VIF verification on selected covariate set:")
            final_vif = calculate_vif(df[selected_covariates].dropna(), threshold=vif_threshold)
            assessment_results['final_vif_results'] = final_vif
            
            if verbose:
                print(final_vif)
                
                # Check for remaining issues
                remaining_high_vif = final_vif[final_vif['VIF'] > vif_threshold]
                if len(remaining_high_vif) > 0:
                    print(f"\n⚠️  Warning: {len(remaining_high_vif)} variables still have high VIF")
                    print("🔧 Consider additional variable selection or regularization")
                    
        except Exception as e:
            if verbose:
                print(f"⚠️  Final VIF verification failed: {e}")
    
    if verbose:
        print(f"\n✅ Multicollinearity assessment completed!")
        print(f"🎯 Final modeling covariates: {selected_covariates}")
        print(f"📊 Ready for AFT model fitting with {len(selected_covariates)} covariates")
    
    return selected_covariates, assessment_results


def compute_actual_first_visit_weeks(df_preprocessed: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Compute actual first visit weeks per patient from the raw longitudinal data.
    
    This fixes the left-censoring analysis bug where L=0 was used instead of 
    actual first measurement weeks.
    
    Args:
        df_preprocessed: Preprocessed longitudinal data with maternal_id and gestational_weeks
        verbose: Whether to print progress information
        
    Returns:
        DataFrame with maternal_id and actual_first_visit_week columns
    """
    if verbose:
        print("🔧 Computing actual first visit weeks (fixing left-censoring bug)...")
    
    if 'maternal_id' not in df_preprocessed.columns or 'gestational_weeks' not in df_preprocessed.columns:
        raise ValueError("Required columns missing: maternal_id, gestational_weeks")
    
    # Compute actual first visit week per patient
    first_visit_weeks = df_preprocessed.groupby('maternal_id')['gestational_weeks'].min().reset_index()
    first_visit_weeks.columns = ['maternal_id', 'actual_first_visit_week']
    
    if verbose:
        n_patients = len(first_visit_weeks)
        mean_first_week = first_visit_weeks['actual_first_visit_week'].mean()
        print(f"   ✅ Computed first visit weeks for {n_patients} patients")
        print(f"   📊 Mean actual first visit: {mean_first_week:.1f} weeks")
        print(f"   📊 Range: {first_visit_weeks['actual_first_visit_week'].min():.1f} - {first_visit_weeks['actual_first_visit_week'].max():.1f} weeks")
    
    return first_visit_weeks


def create_parsimonious_model_specifications(df: pd.DataFrame,
                                           pca_results: Dict[str, Any] = None,
                                           verbose: bool = True) -> Dict[str, List[str]]:
    """
    Create parsimonious model specifications to address low events per covariate.
    
    FIXED: Implements biological core and tech-adjusted models as recommended.
    
    Args:
        df: DataFrame with standardized covariates
        pca_results: PCA consolidation results (if available)
        verbose: Whether to print model specifications
        
    Returns:
        Dictionary of model specifications with covariate lists
    """
    if verbose:
        print("🎯 Creating Parsimonious Model Specifications (FIXED):")
        print("   Addressing low events per covariate (~3.7 events/cov)")
    
    model_specs = {}
    
    # 1. Biological core model: BMI + Age only (2 covariates)
    core_covariates = []
    if 'bmi_std' in df.columns:
        core_covariates.append('bmi_std')
    if 'age_std' in df.columns:
        core_covariates.append('age_std')
    
    if core_covariates:
        model_specs['biological_core'] = core_covariates
        if verbose:
            print(f"   1️⃣  Biological Core: {core_covariates}")
    
    # 2. Tech-adjusted model: BMI + Age + 1-2 QC PCs (3-4 covariates max)
    if pca_results is not None and 'alternative_modeling_covariates' in pca_results:
        pca_alternatives = pca_results['alternative_modeling_covariates']
        
        # Use core + top PC (conservative)
        if 'core_plus_top_pc' in pca_alternatives:
            tech_adjusted = pca_alternatives['core_plus_top_pc']
            available_tech_adjusted = [cov for cov in tech_adjusted if cov in df.columns]
            if len(available_tech_adjusted) >= 2:
                model_specs['tech_adjusted_1pc'] = available_tech_adjusted
                if verbose:
                    print(f"   2️⃣  Tech-Adjusted (1 PC): {available_tech_adjusted}")
        
        # Use core + top 2 PCs (moderate)
        if 'core_plus_pca' in pca_alternatives:
            tech_adjusted_2pc = pca_alternatives['core_plus_pca']
            available_tech_adjusted_2pc = [cov for cov in tech_adjusted_2pc if cov in df.columns]
            if len(available_tech_adjusted_2pc) >= 2 and len(available_tech_adjusted_2pc) <= 4:
                model_specs['tech_adjusted_2pc'] = available_tech_adjusted_2pc
                if verbose:
                    print(f"   3️⃣  Tech-Adjusted (2 PCs): {available_tech_adjusted_2pc}")
    
    # 3. Extended model (if events permit): All available standardized (for comparison only)
    all_std_covs = [col for col in df.columns if col.endswith('_std') and df[col].notna().sum() > 10]
    # Limit to 6 covariates max (for ~22 events = 3.7 events/cov)
    limited_extended = all_std_covs[:6]
    if len(limited_extended) >= 2:
        model_specs['extended_limited'] = limited_extended
        if verbose:
            print(f"   4️⃣  Extended (Limited): {limited_extended[:3]}{'...' if len(limited_extended) > 3 else ''}")
    
    # 4. Add interpretability notes
    model_interpretation = {
        'biological_core': 'Primary clinical drivers (BMI, age) - interpret as biological effects',
        'tech_adjusted_1pc': 'Biological + measurement process adjustment (1 QC PC)',
        'tech_adjusted_2pc': 'Biological + measurement process adjustment (2 QC PCs)', 
        'extended_limited': 'All available covariates (for comparison, may be overfitted)'
    }
    
    if verbose:
        print(f"\n   💡 Model Selection Strategy:")
        print(f"      • Start with biological_core for primary interpretation")
        print(f"      • Use tech_adjusted_1pc if AIC improves by >2 points")
        print(f"      • Interpret QC PCs as measurement-process adjusters, NOT biology")
        print(f"   📊 Created {len(model_specs)} parsimonious model specifications")
    
    return {
        'model_specifications': model_specs,
        'model_interpretation': model_interpretation,
        'selection_strategy': 'Start with biological_core, add QC adjustment if AIC improves'
    }


def analyze_heavy_left_censoring(df_intervals: pd.DataFrame, 
                                threshold: float = 0.04,  # FIXED: Restore original parameter order
                                df_preprocessed: pd.DataFrame = None,
                                verbose: bool = True) -> Dict[str, Any]:
    """
    Analyze and document heavy left-censoring patterns in detail.
    
    FIXED: Now uses actual first visit weeks instead of L=0 artifact.
    
    Args:
        df_intervals: Interval-censored data with censor_type column
        df_preprocessed: Original preprocessed data for computing actual first visits
        threshold: Y-chromosome concentration threshold used (default 0.04)
        verbose: Whether to print detailed analysis
        
    Returns:
        Dictionary with left-censoring analysis results
    """
    if verbose:
        print("🔍 HEAVY LEFT-CENSORING ANALYSIS (FIXED):")
        print(f"   Investigating censoring patterns for threshold = {threshold*100}%")
    
    analysis_results = {
        'threshold': threshold,
        'censoring_distribution': {},
        'left_censored_characteristics': {},
        'recommendations': [],
        'sensitivity_needed': False
    }
    
    # Basic censoring distribution
    if 'censor_type' in df_intervals.columns:
        censor_counts = df_intervals['censor_type'].value_counts()
        censor_pcts = (censor_counts / len(df_intervals) * 100).round(1)
        
        analysis_results['censoring_distribution'] = {
            'counts': censor_counts.to_dict(),
            'percentages': censor_pcts.to_dict(),
            'total_observations': len(df_intervals)
        }
        
        left_censored_pct = censor_pcts.get('left', 0)
        
        if verbose:
            print(f"\n📊 Censoring Distribution:")
            for ctype, count in censor_counts.items():
                pct = censor_pcts[ctype]
                print(f"   {ctype.title()}: {count} ({pct}%)")
    
    # Analyze left-censored observations in detail with ACTUAL first visit weeks
    if 'censor_type' in df_intervals.columns:
        left_censored = df_intervals[df_intervals['censor_type'] == 'left']
        
        if len(left_censored) > 0:
            analysis_results['left_censored_characteristics'] = {
                'n_left_censored': len(left_censored),
                'first_measurement_weeks': {},
                'bmi_distribution': {},
                'age_distribution': {}
            }
            
            # FIXED: Get actual first visit weeks instead of using L=0
            if df_preprocessed is not None and 'maternal_id' in df_intervals.columns:
                try:
                    actual_first_visits = compute_actual_first_visit_weeks(df_preprocessed, verbose=False)
                    left_with_first_visits = left_censored.merge(
                        actual_first_visits, on='maternal_id', how='left'
                    )
                    
                    if 'actual_first_visit_week' in left_with_first_visits.columns:
                        first_weeks = left_with_first_visits['actual_first_visit_week'].dropna()
                        
                        analysis_results['left_censored_characteristics']['first_measurement_weeks'] = {
                            'mean': first_weeks.mean(),
                            'median': first_weeks.median(),
                            'min': first_weeks.min(),
                            'max': first_weeks.max(),
                            'std': first_weeks.std(),
                            'data_source': 'actual_first_visits'  # Flag for corrected data
                        }
                        
                        if verbose:
                            print(f"\n🕐 Left-Censored ACTUAL First Measurement Timing (FIXED):")
                            print(f"   Mean actual first measurement: {first_weeks.mean():.1f} weeks")
                            print(f"   Median actual first measurement: {first_weeks.median():.1f} weeks")
                            print(f"   Range: {first_weeks.min():.1f} - {first_weeks.max():.1f} weeks")
                            
                            # Check if early measurements explain left-censoring
                            early_measurements = (first_weeks < 12).sum()
                            if early_measurements > 0:
                                early_pct = (early_measurements / len(first_weeks)) * 100
                                print(f"   📊 {early_measurements} ({early_pct:.1f}%) first measurements < 12 weeks")
                                print(f"      These patients reached ≥4% Y-concentration at their first visit")
                            
                            late_measurements = (first_weeks >= 16).sum()
                            if late_measurements > 0:
                                late_pct = (late_measurements / len(first_weeks)) * 100
                                print(f"   📊 {late_measurements} ({late_pct:.1f}%) first measurements ≥ 16 weeks")
                                print(f"      Later testing may have missed early threshold crossing")
                    
                except Exception as e:
                    if verbose:
                        print(f"   ⚠️  Could not compute actual first visits: {e}")
                        print(f"   📊 Fallback: Using interval bounds (may include L=0 artifact)")
                    # Fallback to original method with warning
                    first_weeks = left_censored['L'] if 'L' in left_censored.columns else None
            else:
                if verbose:
                    print(f"   ⚠️  No preprocessed data provided - using interval bounds")
                    print(f"   ⚠️  WARNING: This may include the L=0 artifact for left-censored")
                # Fallback to original method
                first_weeks = left_censored['L'] if 'L' in left_censored.columns else None
            
            # Store first weeks analysis (either corrected or fallback) - FIXED logic
            if first_weeks is not None and len(first_weeks) > 0:
                # Only set if not already set by the corrected method above
                if 'first_measurement_weeks' not in analysis_results['left_censored_characteristics']:
                    analysis_results['left_censored_characteristics']['first_measurement_weeks'] = {
                        'mean': first_weeks.mean(),
                        'median': first_weeks.median(),
                        'min': first_weeks.min(),
                        'max': first_weeks.max(),
                        'std': first_weeks.std(),
                        'data_source': 'interval_bounds_fallback'
                    }
            elif 'first_measurement_weeks' not in analysis_results['left_censored_characteristics']:
                # FIXED: Ensure the key always exists even if no data
                analysis_results['left_censored_characteristics']['first_measurement_weeks'] = {
                    'mean': np.nan,
                    'median': np.nan,
                    'min': np.nan,
                    'max': np.nan,
                    'std': np.nan,
                    'data_source': 'no_data_available'
                }
            
            # BMI distribution among left-censored
            if 'bmi' in left_censored.columns:
                bmi_values = left_censored['bmi'].dropna()
                if len(bmi_values) > 0:
                    analysis_results['left_censored_characteristics']['bmi_distribution'] = {
                        'mean': bmi_values.mean(),
                        'median': bmi_values.median(),
                        'min': bmi_values.min(),
                        'max': bmi_values.max(),
                        'std': bmi_values.std()
                    }
                    
                    if verbose:
                        print(f"\n📏 BMI Distribution (Left-Censored):")
                        print(f"   Mean BMI: {bmi_values.mean():.1f}")
                        print(f"   Median BMI: {bmi_values.median():.1f}")
            
            # Age distribution among left-censored
            if 'age' in left_censored.columns:
                age_values = left_censored['age'].dropna()
                if len(age_values) > 0:
                    analysis_results['left_censored_characteristics']['age_distribution'] = {
                        'mean': age_values.mean(),
                        'median': age_values.median(),
                        'min': age_values.min(),
                        'max': age_values.max(),
                        'std': age_values.std()
                    }
                    
                    if verbose:
                        print(f"\n🎂 Age Distribution (Left-Censored):")
                        print(f"   Mean age: {age_values.mean():.1f} years")
                        print(f"   Median age: {age_values.median():.1f} years")
    
    # Generate recommendations based on analysis
    recommendations = []
    
    if left_censored_pct >= 80:
        recommendations.append("High left-censoring (≥80%) detected - consider sensitivity analysis")
        recommendations.append("Investigate measurement schedule: are first visits too late in pregnancy?")
        recommendations.append("Consider alternative thresholds (e.g., 3%, 5%) for robustness")
        analysis_results['sensitivity_needed'] = True
    
    # FIXED: Properly handle mean_first_week access with error checking
    if 'first_measurement_weeks' in analysis_results['left_censored_characteristics']:
        first_weeks_data = analysis_results['left_censored_characteristics']['first_measurement_weeks']
        if 'mean' in first_weeks_data:
            mean_first_week = first_weeks_data['mean']
            if not np.isnan(mean_first_week) and mean_first_week > 15:
                recommendations.append(f"Late first measurements (mean={mean_first_week:.1f} weeks) may contribute to left-censoring")
                recommendations.append("Consider recommending earlier initial NIPT testing")
    
    analysis_results['recommendations'] = recommendations
    
    if verbose:
        print(f"\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
        
        if analysis_results['sensitivity_needed']:
            print(f"\n⚠️  CAUTION: Heavy left-censoring affects AFT interpretation")
            print(f"   Time ratios represent acceleration/delay conditional on late threshold attainment")
            print(f"   Results may not generalize to mothers who reach threshold early")
    
    return analysis_results


def apply_pca_qc_consolidation(df: pd.DataFrame, 
                             qc_variables: List[str] = None,
                             variance_threshold: float = 0.85,  # FIXED: Lower threshold for parsimony
                             verbose: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply PCA to consolidate highly correlated QC variables into principal components.
    
    This addresses the multicollinearity issue where raw_read_count, unique_mapped_reads, 
    mapping_ratio, duplicate_ratio, etc. are highly correlated.
    
    Args:
        df: DataFrame with QC variables
        qc_variables: List of QC variable names (if None, auto-detect)
        variance_threshold: Minimum cumulative variance to retain (default 0.95)
        verbose: Whether to print detailed analysis
        
    Returns:
        Tuple of (df_with_pcs, pca_results_dict)
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    if verbose:
        print("🔬 PCA-based QC Variable Consolidation:")
    
    # Auto-detect QC variables if not provided
    if qc_variables is None:
        potential_qc_vars = [
            'raw_read_count', 'unique_mapped_reads', 'mapping_ratio',
            'duplicate_ratio', 'filtered_reads_ratio', 'gc_content'
        ]
        # Include standardized versions
        potential_qc_vars_std = [var + '_std' for var in potential_qc_vars]
        
        all_potential = potential_qc_vars + potential_qc_vars_std
        qc_variables = [var for var in all_potential if var in df.columns]
    
    available_qc_vars = [var for var in qc_variables if var in df.columns and df[var].notna().sum() > 10]
    
    if len(available_qc_vars) < 2:
        if verbose:
            print(f"   ⚠️  Insufficient QC variables for PCA: {available_qc_vars}")
        return df, {'pca_applied': False, 'reason': 'insufficient_variables'}
    
    if verbose:
        print(f"   📊 Available QC variables: {len(available_qc_vars)}")
        print(f"      Variables: {available_qc_vars}")
    
    # Extract and clean QC data
    df_qc = df[available_qc_vars].copy()
    
    # Handle missing values
    missing_counts = df_qc.isnull().sum()
    if missing_counts.sum() > 0:
        if verbose:
            print(f"   🔧 Handling missing values...")
            for var, missing in missing_counts[missing_counts > 0].items():
                print(f"      {var}: {missing} missing values")
        
        # Use mean imputation for QC variables
        df_qc = df_qc.fillna(df_qc.mean())
    
    # Check for correlation among QC variables
    correlation_matrix = df_qc.corr()
    high_corr_pairs = []
    
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_val = abs(correlation_matrix.iloc[i, j])
            if corr_val > 0.7:  # High correlation threshold
                var1, var2 = correlation_matrix.columns[i], correlation_matrix.columns[j]
                high_corr_pairs.append((var1, var2, corr_val))
    
    if verbose:
        print(f"   🔍 High correlation pairs (>0.7): {len(high_corr_pairs)}")
        for var1, var2, corr in high_corr_pairs[:3]:  # Show first 3
            print(f"      {var1} ↔ {var2}: r={corr:.3f}")
        if len(high_corr_pairs) > 3:
            print(f"      ... and {len(high_corr_pairs) - 3} more")
    
    # Apply PCA
    # Standardize if variables are not already standardized
    if not all(var.endswith('_std') for var in available_qc_vars):
        if verbose:
            print(f"   📏 Standardizing QC variables for PCA...")
        scaler = StandardScaler()
        qc_data_scaled = scaler.fit_transform(df_qc)
        scaling_applied = True
        scaling_params = {
            'means': scaler.mean_,
            'stds': scaler.scale_
        }
    else:
        qc_data_scaled = df_qc.values
        scaling_applied = False
        scaling_params = None
    
    # Fit PCA
    pca = PCA()
    pca_data = pca.fit_transform(qc_data_scaled)
    
    # Determine number of components to retain
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
    
    # Ensure at least 1 component
    n_components = max(1, n_components)
    
    pca_results = {
        'pca_applied': True,
        'original_variables': available_qc_vars,
        'n_original_variables': len(available_qc_vars),
        'n_components_retained': n_components,
        'explained_variance_ratio': pca.explained_variance_ratio_[:n_components],
        'cumulative_variance': cumulative_variance[:n_components],
        'loadings': pca.components_[:n_components],
        'high_correlation_pairs': high_corr_pairs,
        'scaling_applied': scaling_applied,
        'scaling_params': scaling_params
    }
    
    if verbose:
        print(f"   📈 PCA Results:")
        print(f"      Components retained: {n_components}/{len(available_qc_vars)}")
        print(f"      Cumulative variance explained: {cumulative_variance[n_components-1]:.3f}")
        print(f"      Individual variance explained:")
        for i in range(n_components):
            print(f"         PC{i+1}: {pca.explained_variance_ratio_[i]:.3f} ({pca.explained_variance_ratio_[i]*100:.1f}%)")
    
    # Add principal components to dataframe
    df_result = df.copy()
    
    for i in range(n_components):
        pc_name = f'qc_pc{i+1}'
        df_result[pc_name] = pca_data[:, i]
        
        if verbose and i < 3:  # Show loadings for first 3 PCs
            print(f"      {pc_name} loadings:")
            loadings = pca.components_[i]
            for j, var in enumerate(available_qc_vars):
                if abs(loadings[j]) > 0.3:  # Only show substantial loadings
                    print(f"         {var}: {loadings[j]:+.3f}")
    
    # Calculate VIF reduction potential
    if len(available_qc_vars) > 1:
        try:
            original_vif = calculate_vif(df[available_qc_vars])
            pc_vars = [f'qc_pc{i+1}' for i in range(n_components)]
            pc_vif = calculate_vif(df_result[pc_vars])
            
            pca_results['vif_comparison'] = {
                'original_max_vif': original_vif['VIF'].max(),
                'pc_max_vif': pc_vif['VIF'].max(),
                'vif_reduction': original_vif['VIF'].max() - pc_vif['VIF'].max()
            }
            
            if verbose:
                print(f"   📊 VIF Comparison:")
                print(f"      Original max VIF: {original_vif['VIF'].max():.2f}")
                print(f"      PC max VIF: {pc_vif['VIF'].max():.2f}")
                print(f"      Reduction: {pca_results['vif_comparison']['vif_reduction']:.2f}")
                
        except Exception as e:
            if verbose:
                print(f"   ⚠️  VIF comparison failed: {e}")
    
    # FIXED: Create alternative modeling covariates that actually use PCA components
    pc_vars = [f'qc_pc{i+1}' for i in range(n_components)]
    
    # Create alternative covariate specifications
    basic_covariates = ['bmi_std', 'age_std']  # Core always included
    
    # Create PCA-based covariate alternatives
    alternative_modeling_covariates = {
        'core_only': basic_covariates,
        'core_plus_top_pc': basic_covariates + [pc_vars[0]] if pc_vars else basic_covariates,
        'core_plus_pca': basic_covariates + pc_vars[:min(2, len(pc_vars))],  # Limit to top 2 PCs
        'full_pca': basic_covariates + pc_vars  # All PCs (for comparison)
    }
    
    pca_results['recommended_usage'] = {
        'replace_variables': available_qc_vars,
        'with_components': pc_vars,
        'modeling_benefit': f'Reduces multicollinearity while preserving {cumulative_variance[n_components-1]:.1%} variance',
        'alternative_covariate_sets': alternative_modeling_covariates,  # FIXED: Actually provide alternatives
        'recommended_set': 'core_plus_top_pc'  # Conservative recommendation
    }
    
    # FIXED: Add the alternative covariates to the results for actual use
    pca_results['alternative_modeling_covariates'] = alternative_modeling_covariates
    
    if verbose:
        print(f"   💡 FIXED Recommendation:")
        print(f"      Replace {len(available_qc_vars)} QC variables with {len(pc_vars)} principal components")
        print(f"      Variance explained: {cumulative_variance[n_components-1]:.1%}")
        print(f"      📋 Alternative covariate sets created:")
        for set_name, covs in alternative_modeling_covariates.items():
            print(f"         {set_name}: {covs}")
        print(f"      🎯 Recommended for modeling: {alternative_modeling_covariates['core_plus_top_pc']}")
        print(f"   ✅ PCA consolidation completed successfully!")
    
    return df_result, pca_results


def construct_intervals_extended(df_preprocessed: pd.DataFrame, 
                               threshold: float = 0.04, 
                               verbose: bool = True) -> pd.DataFrame:
    """
    Section 2.1: Event Interval Construction (Extended for Problem 3).
    
    Reuses construct_intervals() from Problem 2 but ensures compatibility with
    extended covariates (BMI, age, height, weight) and preserves them in output.
    
    For each maternal_id, determine:
    1. Left-censored: First observation already ≥4% → L=0, R=first_week
    2. Interval-censored: Threshold crossed between visits → L=weeks[j-1], R=weeks[j]  
    3. Right-censored: Never reached threshold → L=last_week, R=inf
    
    Args:
        df_preprocessed: Preprocessed DataFrame with extended covariates
        threshold: Y-chromosome concentration threshold (default 0.04 = 4%)
        verbose: Whether to print progress information
        
    Returns:
        DataFrame with interval-censored observations and extended covariates
    """
    if verbose:
        print("🔄 Section 2.1: Constructing interval-censored observations (extended)...")
        print(f"   📊 Input data: {df_preprocessed.shape}")
        print(f"   🎯 Threshold: {threshold * 100}%")
    
    # Import construct_intervals from Problem 2
    from ..problem2.data_preprocessing import construct_intervals
    
    # Prepare basic required columns for interval construction
    required_cols = ['maternal_id', 'gestational_weeks', 'bmi', 'y_concentration']
    missing_required = [col for col in required_cols if col not in df_preprocessed.columns]
    
    if missing_required:
        raise ValueError(f"Missing required columns for interval construction: {missing_required}")
    
    # Extract basic data for interval construction
    df_basic = df_preprocessed[required_cols].copy()
    
    if verbose:
        print(f"   📋 Core variables for intervals: {required_cols}")
        print(f"   👥 Unique mothers: {df_basic['maternal_id'].nunique()}")
        print(f"   📈 Total test records: {len(df_basic)}")
    
    # Construct intervals using Problem 2 method
    df_intervals_basic = construct_intervals(df_basic, threshold=threshold, verbose=verbose)
    
    if verbose:
        print(f"   ✅ Basic intervals created: {df_intervals_basic.shape}")
        censor_counts = df_intervals_basic['censor_type'].value_counts()
        print(f"   📊 Censoring types: {censor_counts.to_dict()}")
    
    # Merge back extended covariates
    # Get patient-level covariates (take first observation per patient)
    extended_patient_covariates = df_preprocessed.groupby('maternal_id').first().reset_index()
    
    # Extended covariates to preserve (exclude those already in intervals)
    extended_cols = [col for col in df_preprocessed.columns 
                    if col not in ['maternal_id', 'gestational_weeks', 'y_concentration'] 
                    and col not in df_intervals_basic.columns]
    
    # Available extended covariates
    available_extended = [col for col in extended_cols if col in extended_patient_covariates.columns]
    
    if verbose:
        print(f"   📊 Extended covariates to merge: {len(available_extended)}")
        print(f"      Covariates: {available_extended[:10]}{'...' if len(available_extended) > 10 else ''}")
    
    # Merge extended covariates with intervals
    merge_cols = ['maternal_id'] + available_extended
    df_intervals_extended = df_intervals_basic.merge(
        extended_patient_covariates[merge_cols],
        on='maternal_id',
        how='left'
    )
    
    if verbose:
        print(f"   ✅ Extended intervals created: {df_intervals_extended.shape}")
        print(f"   📊 Final columns: {len(df_intervals_extended.columns)}")
        print(f"      Interval cols: ['maternal_id', 'L', 'R', 'censor_type', 'bmi']")
        print(f"      Extended cols: {len(available_extended)} additional covariates")
    
    # DETAILED SAMPLE DERIVATION DOCUMENTATION
    if verbose:
        print(f"\n📋 SAMPLE DERIVATION DOCUMENTATION:")
        print(f"   Original preprocessed data: {df_preprocessed.shape[0]} rows")
        print(f"   Unique mothers in preprocessed: {df_preprocessed['maternal_id'].nunique()}")
        print(f"   After interval construction: {df_intervals_extended.shape[0]} intervals (one per mother)")
        print(f"   Reduction reason: Aggregated multiple visits per mother to single interval")
    
    # Quality validation with improved missing data reporting
    invalid_intervals = (df_intervals_extended['L'] >= df_intervals_extended['R']).sum()
    if invalid_intervals > 0:
        warnings.warn(f"⚠️  {invalid_intervals} invalid intervals detected (L >= R)")
    
    # Comprehensive missing data assessment
    missing_by_covariate = df_intervals_extended[available_extended].isnull().sum()
    total_missing_values = missing_by_covariate.sum()
    
    if verbose:
        print(f"\n📊 MISSING DATA ASSESSMENT (Post-Interval Construction):")
        print(f"   Total intervals: {len(df_intervals_extended)}")
        if total_missing_values > 0:
            print(f"   Total missing values across all extended covariates: {total_missing_values}")
            print(f"   Covariates with missing values:")
            for col, missing_count in missing_by_covariate[missing_by_covariate > 0].items():
                pct_missing = (missing_count / len(df_intervals_extended)) * 100
                print(f"      {col}: {missing_count}/{len(df_intervals_extended)} ({pct_missing:.1f}%)")
        else:
            print(f"   ✅ No missing values in extended covariates")
    
    return df_intervals_extended


def prepare_extended_feature_matrix(df_intervals: pd.DataFrame,
                                  selected_covariates: List[str],
                                  include_splines: bool = False,
                                  verbose: bool = True) -> pd.DataFrame:
    """
    Section 2.2: Extended Feature Matrix Creation.
    
    Creates df_X with interval bounds and standardized covariates for AFT modeling.
    Includes VIF-approved covariate set only and validates completeness.
    
    Args:
        df_intervals: Interval-censored data from construct_intervals_extended
        selected_covariates: VIF-approved covariate list (standardized)
        include_splines: Whether to include BMI splines
        verbose: Whether to print progress information
        
    Returns:
        Feature matrix ready for AFT fitting (df_X)
    """
    if verbose:
        print("📊 Section 2.2: Creating extended feature matrix (df_X)...")
        print(f"   📋 Input intervals: {df_intervals.shape}")
        print(f"   🎯 Selected covariates: {selected_covariates}")
        print(f"   🌊 Include splines: {include_splines}")
    
    # Verify required interval columns
    required_interval_cols = ['maternal_id', 'L', 'R', 'censor_type']
    missing_interval_cols = [col for col in required_interval_cols if col not in df_intervals.columns]
    
    if missing_interval_cols:
        raise ValueError(f"Missing required interval columns: {missing_interval_cols}")
    
    # Verify selected covariates availability
    missing_covariates = [col for col in selected_covariates if col not in df_intervals.columns]
    if missing_covariates:
        available_alternatives = [col for col in df_intervals.columns if any(missing in col for missing in missing_covariates)]
        error_msg = f"Missing selected covariates: {missing_covariates}"
        if available_alternatives:
            error_msg += f"\nAvailable alternatives: {available_alternatives}"
        raise ValueError(error_msg)
    
    # Start with core interval columns and selected covariates
    feature_cols = required_interval_cols + selected_covariates
    
    # Add original covariates for reference and debugging
    original_covariates = ['bmi', 'age', 'height', 'weight']
    available_original = [col for col in original_covariates if col in df_intervals.columns]
    
    df_features = df_intervals[feature_cols + available_original].copy()
    
    if verbose:
        print(f"   📊 Core feature matrix: {df_features.shape}")
        print(f"   📋 Interval columns: {required_interval_cols}")
        print(f"   🎯 Selected covariates: {selected_covariates}")
        print(f"   📚 Original references: {available_original}")
    
    # Add spline features if requested and BMI is standardized
    spline_features_added = 0
    if include_splines and 'bmi_std' in selected_covariates:
        if verbose:
            print("   🌊 Adding BMI spline features...")
        
        spline_basis = create_spline_basis(df_intervals['bmi_std'].values)
        
        if spline_basis is not None:
            # Add spline columns
            spline_cols = [f'bmi_spline_{i}' for i in range(spline_basis.shape[1])]
            df_splines = pd.DataFrame(spline_basis.values, 
                                     columns=spline_cols,
                                     index=df_intervals.index)
            df_features = pd.concat([df_features, df_splines], axis=1)
            spline_features_added = len(spline_cols)
            
            if verbose:
                print(f"      ✅ Added {spline_features_added} spline features: {spline_cols}")
        else:
            if verbose:
                print("      ⚠️  Spline creation failed, using linear BMI only")
    
    # Interval validity checks
    if verbose:
        print("   🔍 Performing quality validation...")
    
    # Check interval validity (L < R)
    invalid_intervals = (df_features['L'] >= df_features['R']).sum()
    if invalid_intervals > 0:
        warnings.warn(f"⚠️  {invalid_intervals} invalid intervals detected (L >= R)")
        if verbose:
            print(f"      ⚠️  Invalid intervals: {invalid_intervals}/{len(df_features)} ({100*invalid_intervals/len(df_features):.1f}%)")
    
    # Check for infinite R values (right-censored)
    infinite_R = np.isinf(df_features['R']).sum()
    if verbose:
        print(f"      📊 Right-censored (R=∞): {infinite_R}/{len(df_features)} ({100*infinite_R/len(df_features):.1f}%)")
    
    # Check covariate completeness
    covariate_missing = df_features[selected_covariates].isnull().sum()
    total_missing = covariate_missing.sum()
    
    if total_missing > 0:
        if verbose:
            print(f"      ⚠️  Missing covariate values: {total_missing} total")
            for cov, missing_count in covariate_missing[covariate_missing > 0].items():
                print(f"         {cov}: {missing_count}/{len(df_features)} ({100*missing_count/len(df_features):.1f}%)")
    else:
        if verbose:
            print(f"      ✅ No missing values in selected covariates")
    
    # Final feature matrix summary
    if verbose:
        print(f"\n   ✅ Extended feature matrix (df_X) prepared successfully!")
        print(f"      📊 Final shape: {df_features.shape}")
        print(f"      📋 Interval columns: {len(required_interval_cols)}")
        print(f"      🎯 Selected covariates: {len(selected_covariates)}")
        print(f"      📚 Original references: {len(available_original)}")
        print(f"      🌊 Spline features: {spline_features_added}")
        print(f"      🔢 Total features: {len(df_features.columns)}")
        print(f"      📈 Ready for AFT model fitting with {len(selected_covariates) + spline_features_added} modeling covariates")
    
    return df_features


def validate_feature_matrix_completeness(df_X: pd.DataFrame, 
                                       selected_covariates: List[str],
                                       verbose: bool = True) -> Dict[str, Any]:
    """
    Validate completeness and quality of extended feature matrix for AFT modeling.
    
    Args:
        df_X: Extended feature matrix from prepare_extended_feature_matrix
        selected_covariates: List of selected covariates for modeling
        verbose: Whether to print validation results
        
    Returns:
        Dictionary with validation results and recommendations
    """
    if verbose:
        print("🔍 Validating extended feature matrix completeness...")
    
    validation_results = {
        'matrix_shape': df_X.shape,
        'interval_validation': {},
        'covariate_validation': {},
        'data_quality': {},
        'modeling_readiness': False,
        'recommendations': []
    }
    
    # 1. Interval validation
    required_cols = ['maternal_id', 'L', 'R', 'censor_type']
    missing_interval_cols = [col for col in required_cols if col not in df_X.columns]
    
    if missing_interval_cols:
        validation_results['interval_validation']['missing_columns'] = missing_interval_cols
        validation_results['recommendations'].append(f"Add missing interval columns: {missing_interval_cols}")
    else:
        validation_results['interval_validation']['all_columns_present'] = True
    
    # Interval bounds validation
    invalid_intervals = (df_X['L'] >= df_X['R']).sum()
    validation_results['interval_validation']['invalid_intervals'] = invalid_intervals
    validation_results['interval_validation']['invalid_percentage'] = invalid_intervals / len(df_X) * 100
    
    if invalid_intervals > 0:
        validation_results['recommendations'].append(f"Fix {invalid_intervals} invalid intervals (L >= R)")
    
    # 2. Covariate validation
    missing_covariates = [col for col in selected_covariates if col not in df_X.columns]
    validation_results['covariate_validation']['missing_covariates'] = missing_covariates
    validation_results['covariate_validation']['available_covariates'] = [col for col in selected_covariates if col in df_X.columns]
    
    if missing_covariates:
        validation_results['recommendations'].append(f"Add missing covariates: {missing_covariates}")
    
    # Covariate completeness
    available_covariates = [col for col in selected_covariates if col in df_X.columns]
    covariate_missing = df_X[available_covariates].isnull().sum()
    validation_results['covariate_validation']['missing_values'] = covariate_missing.to_dict()
    validation_results['covariate_validation']['total_missing'] = covariate_missing.sum()
    
    # 3. Data quality assessment
    validation_results['data_quality']['n_patients'] = df_X['maternal_id'].nunique()
    validation_results['data_quality']['n_observations'] = len(df_X)
    validation_results['data_quality']['censoring_distribution'] = df_X['censor_type'].value_counts().to_dict()
    
    # Check for sufficient sample size
    min_sample_size = len(selected_covariates) * 10  # Rule of thumb: 10 events per covariate
    validation_results['data_quality']['sufficient_sample_size'] = len(df_X) >= min_sample_size
    validation_results['data_quality']['recommended_min_size'] = min_sample_size
    
    if len(df_X) < min_sample_size:
        validation_results['recommendations'].append(f"Sample size {len(df_X)} may be too small for {len(selected_covariates)} covariates (recommend ≥{min_sample_size})")
    
    # 4. Overall readiness assessment
    validation_results['modeling_readiness'] = (
        len(missing_interval_cols) == 0 and
        len(missing_covariates) == 0 and
        invalid_intervals == 0 and
        covariate_missing.sum() == 0
    )
    
    if verbose:
        print(f"   📊 Matrix shape: {validation_results['matrix_shape']}")
        print(f"   📋 Interval columns: {'✅ Complete' if not missing_interval_cols else f'❌ Missing {missing_interval_cols}'}")
        print(f"   📊 Invalid intervals: {invalid_intervals} ({invalid_intervals/len(df_X)*100:.1f}%)")
        print(f"   🎯 Covariates: {len(available_covariates)}/{len(selected_covariates)} available")
        print(f"   📈 Missing values: {covariate_missing.sum()} total")
        print(f"   👥 Patients: {validation_results['data_quality']['n_patients']}")
        print(f"   📊 Sample size: {'✅ Adequate' if validation_results['data_quality']['sufficient_sample_size'] else '⚠️  Potentially insufficient'}")
        print(f"   🎯 Modeling ready: {'✅ Yes' if validation_results['modeling_readiness'] else '❌ Issues found'}")
        
        if validation_results['recommendations']:
            print(f"   📝 Recommendations:")
            for i, rec in enumerate(validation_results['recommendations'], 1):
                print(f"      {i}. {rec}")
    
    return validation_results
