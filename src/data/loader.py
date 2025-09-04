"""
Data loading and preprocessing module for NIPT analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import warnings

from ..config.settings import DATA_FILE, COLUMN_MAPPING, AnalysisConfig


class NIPTDataLoader:
    """
    Data loader and preprocessor for NIPT analysis.
    
    This class handles loading the Excel data, cleaning, and basic preprocessing
    for subsequent analysis modules.
    """
    
    def __init__(self, data_file: Optional[Path] = None):
        """
        Initialize the data loader.
        
        Args:
            data_file: Path to the data file. If None, uses default from config.
        """
        self.data_file = data_file or DATA_FILE
        self.raw_data = None
        self.processed_data = None
        self.male_data = None
        self.female_data = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load raw data from Excel file.
        
        Returns:
            DataFrame with raw data
        """
        try:
            self.raw_data = pd.read_excel(self.data_file)
            print(f"Loaded data with shape: {self.raw_data.shape}")
            return self.raw_data
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {self.data_file}")
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def preprocess_data(self) -> pd.DataFrame:
        """
        Clean and preprocess the raw data.
        
        Returns:
            DataFrame with preprocessed data
        """
        if self.raw_data is None:
            self.load_data()
            
        df = self.raw_data.copy()
        
        # Rename columns to English for easier handling
        df = df.rename(columns=COLUMN_MAPPING)
        
        # Convert data types
        df = self._convert_data_types(df)
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Add derived features
        df = self._add_derived_features(df)
        
        # Filter valid samples
        df = self._filter_valid_samples(df)
        
        self.processed_data = df
        print(f"Processed data shape: {df.shape}")
        
        return df
    
    def _convert_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert data types appropriately."""
        
        # Numeric columns that should be float
        numeric_cols = ['age', 'height', 'weight', 'bmi', 'gestational_weeks',
                       'y_chr_concentration', 'x_chr_concentration',
                       'chr13_z_value', 'chr18_z_value', 'chr21_z_value',
                       'x_chr_z_value', 'y_chr_z_value', 'gc_content',
                       'chr13_gc_content', 'chr18_gc_content', 'chr21_gc_content',
                       'mapping_ratio', 'duplicate_ratio', 'filtered_reads_ratio']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Integer columns
        int_cols = ['sample_id', 'blood_draw_count', 'raw_read_count', 
                   'unique_mapped_reads', 'pregnancy_count', 'birth_count']
        
        for col in int_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
        
        # Convert dates
        if 'test_date' in df.columns:
            df['test_date'] = pd.to_datetime(df['test_date'], errors='coerce')
        
        if 'last_menstrual_period' in df.columns:
            df['last_menstrual_period'] = pd.to_datetime(df['last_menstrual_period'], errors='coerce')
            
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values appropriately."""
        
        # For critical analysis columns, we'll keep NaN and handle in specific analyses
        # Log missing value statistics
        missing_stats = df.isnull().sum()
        if missing_stats.sum() > 0:
            print("Missing values summary:")
            print(missing_stats[missing_stats > 0])
            
        return df
    
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features for analysis."""
        
        # BMI calculation (if height and weight are available)
        if 'height' in df.columns and 'weight' in df.columns:
            df['calculated_bmi'] = df['weight'] / (df['height'] / 100) ** 2
        
        # Fetal sex determination based on Y chromosome concentration
        if 'y_chr_concentration' in df.columns:
            df['fetal_sex'] = df['y_chr_concentration'].apply(
                lambda x: 'male' if pd.notna(x) and x >= AnalysisConfig.Y_CHROMOSOME_THRESHOLD 
                else 'female' if pd.notna(x) else 'unknown'
            )
        
        # Risk category based on gestational weeks
        if 'gestational_weeks' in df.columns:
            df['risk_category'] = df['gestational_weeks'].apply(self._categorize_risk)
        
        # BMI group assignment
        if 'bmi' in df.columns:
            df['bmi_group'] = df['bmi'].apply(self._assign_bmi_group)
            
        return df
    
    def _categorize_risk(self, weeks: float) -> str:
        """Categorize pregnancy risk based on gestational weeks."""
        if pd.isna(weeks):
            return 'unknown'
        elif weeks <= AnalysisConfig.LOW_RISK_WEEKS:
            return 'low'
        elif weeks <= AnalysisConfig.MEDIUM_RISK_WEEKS:
            return 'medium'
        else:
            return 'high'
    
    def _assign_bmi_group(self, bmi: float) -> int:
        """Assign BMI group based on configured ranges."""
        if pd.isna(bmi):
            return -1
        
        for i, (lower, upper) in enumerate(AnalysisConfig.BMI_GROUPS):
            if lower <= bmi < upper:
                return i + 1
        return -1
    
    def _filter_valid_samples(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter out clearly invalid samples."""
        
        initial_count = len(df)
        
        # Remove samples with invalid BMI (< 10 or > 60)
        if 'bmi' in df.columns:
            df = df[(df['bmi'].isna()) | ((df['bmi'] >= 10) & (df['bmi'] <= 60))]
        
        # Remove samples with invalid gestational weeks (< 5 or > 45)
        if 'gestational_weeks' in df.columns:
            df = df[(df['gestational_weeks'].isna()) | 
                   ((df['gestational_weeks'] >= 5) & (df['gestational_weeks'] <= 45))]
        
        # Remove samples with invalid age (< 15 or > 55)
        if 'age' in df.columns:
            df = df[(df['age'].isna()) | ((df['age'] >= 15) & (df['age'] <= 55))]
        
        filtered_count = initial_count - len(df)
        if filtered_count > 0:
            print(f"Filtered out {filtered_count} invalid samples")
            
        return df
    
    def split_by_fetal_sex(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into male and female fetus groups.
        
        Returns:
            Tuple of (male_data, female_data)
        """
        if self.processed_data is None:
            self.preprocess_data()
            
        male_data = self.processed_data[
            self.processed_data['fetal_sex'] == 'male'
        ].copy()
        
        female_data = self.processed_data[
            self.processed_data['fetal_sex'] == 'female'
        ].copy()
        
        self.male_data = male_data
        self.female_data = female_data
        
        print(f"Male fetus samples: {len(male_data)}")
        print(f"Female fetus samples: {len(female_data)}")
        
        return male_data, female_data
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics of the processed data.
        
        Returns:
            Dictionary with summary statistics
        """
        if self.processed_data is None:
            self.preprocess_data()
            
        df = self.processed_data
        
        summary = {
            'total_samples': len(df),
            'unique_patients': df['patient_code'].nunique() if 'patient_code' in df.columns else 'N/A',
            'fetal_sex_distribution': df['fetal_sex'].value_counts().to_dict() if 'fetal_sex' in df.columns else {},
            'bmi_group_distribution': df['bmi_group'].value_counts().to_dict() if 'bmi_group' in df.columns else {},
            'risk_category_distribution': df['risk_category'].value_counts().to_dict() if 'risk_category' in df.columns else {},
            'gestational_weeks_range': {
                'min': df['gestational_weeks'].min(),
                'max': df['gestational_weeks'].max(),
                'mean': df['gestational_weeks'].mean()
            } if 'gestational_weeks' in df.columns else {},
            'bmi_range': {
                'min': df['bmi'].min(),
                'max': df['bmi'].max(),
                'mean': df['bmi'].mean()
            } if 'bmi' in df.columns else {}
        }
        
        return summary
