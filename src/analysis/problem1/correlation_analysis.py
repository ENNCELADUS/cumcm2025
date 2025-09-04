"""
Y chromosome concentration correlation analysis for Problem 1.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from ...utils.statistics import StatisticalAnalyzer
from ...utils.visualization import NIPTVisualizer
from ...config.settings import AnalysisConfig


class YChromosomeCorrelationAnalyzer:
    """
    Analyzer for Y chromosome concentration relationships with maternal factors.
    
    This class implements analysis for Problem 1: analyzing the correlation
    between fetal Y chromosome concentration and maternal gestational weeks,
    BMI, and other indicators, building relationship models and testing significance.
    """
    
    def __init__(self):
        """Initialize the analyzer."""
        self.stats_analyzer = StatisticalAnalyzer()
        self.visualizer = NIPTVisualizer()
        self.results = {}
        
    def analyze_correlations(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze correlations between Y chromosome concentration and maternal factors.
        
        Args:
            data: DataFrame with NIPT data (should be filtered for male fetuses)
            
        Returns:
            Dictionary with correlation analysis results
        """
        # Filter for male fetuses with valid Y chromosome concentration
        male_data = data[
            (data['fetal_sex'] == 'male') & 
            (data['y_chr_concentration'].notna())
        ].copy()
        
        if len(male_data) < 10:
            raise ValueError("Insufficient male fetus data for analysis")
        
        # Define features to analyze
        features = [
            'gestational_weeks', 'bmi', 'age', 'height', 'weight',
            'x_chr_concentration', 'gc_content', 'chr13_z_value',
            'chr18_z_value', 'chr21_z_value', 'x_chr_z_value'
        ]
        
        # Filter features that exist in data
        available_features = [f for f in features if f in male_data.columns]
        
        # Perform correlation analysis
        correlation_results = self.stats_analyzer.correlation_analysis(
            male_data, 'y_chr_concentration', available_features
        )
        
        # Store results
        self.results['correlation_analysis'] = {
            'sample_size': len(male_data),
            'correlations': correlation_results['correlations'],
            'p_values': correlation_results['p_values'],
            'significant_features': correlation_results['significant_features'],
            'features_analyzed': available_features
        }
        
        return self.results['correlation_analysis']
    
    def build_regression_model(self, data: pd.DataFrame, 
                              features: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Build regression model for Y chromosome concentration.
        
        Args:
            data: DataFrame with NIPT data
            features: List of features to include. If None, uses significant features.
            
        Returns:
            Dictionary with regression model results
        """
        # Filter for male fetuses
        male_data = data[
            (data['fetal_sex'] == 'male') & 
            (data['y_chr_concentration'].notna())
        ].copy()
        
        if features is None:
            # Use primary features from problem description
            features = ['gestational_weeks', 'bmi']
            
            # Add other significant features if available
            if 'correlation_analysis' in self.results:
                sig_features = self.results['correlation_analysis']['significant_features']
                additional_features = [f for f in sig_features 
                                     if f not in features and f in male_data.columns]
                features.extend(additional_features[:3])  # Limit to avoid overfitting
        
        # Ensure features exist in data
        features = [f for f in features if f in male_data.columns]
        
        if len(features) == 0:
            raise ValueError("No valid features available for regression")
        
        # Perform regression analysis
        regression_results = self.stats_analyzer.linear_regression_analysis(
            male_data, 'y_chr_concentration', features
        )
        
        # Store results
        self.results['regression_model'] = regression_results
        self.results['regression_model']['features_used'] = features
        
        return self.results['regression_model']
    
    def test_significance(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive significance tests.
        
        Args:
            data: DataFrame with NIPT data
            
        Returns:
            Dictionary with significance test results
        """
        male_data = data[
            (data['fetal_sex'] == 'male') & 
            (data['y_chr_concentration'].notna())
        ].copy()
        
        significance_results = {}
        
        # Test correlation significance for key variables
        key_variables = ['gestational_weeks', 'bmi']
        
        for var in key_variables:
            if var in male_data.columns:
                clean_data = male_data[[var, 'y_chr_concentration']].dropna()
                
                if len(clean_data) >= 3:
                    from scipy import stats
                    corr, p_value = stats.pearsonr(
                        clean_data[var], 
                        clean_data['y_chr_concentration']
                    )
                    
                    significance_results[f'{var}_correlation'] = {
                        'correlation': corr,
                        'p_value': p_value,
                        'significant': p_value < AnalysisConfig.ALPHA,
                        'sample_size': len(clean_data)
                    }
        
        # Test for differences across BMI groups
        if 'bmi_group' in male_data.columns:
            try:
                anova_result = self.stats_analyzer.anova_test(
                    male_data, 'y_chr_concentration', 'bmi_group'
                )
                significance_results['bmi_group_anova'] = anova_result
            except Exception as e:
                significance_results['bmi_group_anova'] = {'error': str(e)}
        
        # Test normality of Y chromosome concentration
        normality_result = self.stats_analyzer.normality_test(
            male_data['y_chr_concentration']
        )
        significance_results['y_chr_normality'] = normality_result
        
        self.results['significance_tests'] = significance_results
        
        return significance_results
    
    def create_visualizations(self, data: pd.DataFrame, save_figures: bool = True) -> Dict[str, Any]:
        """
        Create visualizations for correlation analysis.
        
        Args:
            data: DataFrame with NIPT data
            save_figures: Whether to save figures to disk
            
        Returns:
            Dictionary with figure objects
        """
        male_data = data[
            (data['fetal_sex'] == 'male') & 
            (data['y_chr_concentration'].notna())
        ].copy()
        
        figures = {}
        
        # 1. Scatter plot: Y concentration vs Gestational weeks
        if 'gestational_weeks' in male_data.columns:
            fig1 = self.visualizer.scatter_plot(
                male_data, 'gestational_weeks', 'y_chr_concentration',
                title='Y Chromosome Concentration vs Gestational Weeks',
                save_name='y_chr_vs_gestational_weeks' if save_figures else None
            )
            figures['gestational_weeks_scatter'] = fig1
        
        # 2. Scatter plot: Y concentration vs BMI
        if 'bmi' in male_data.columns:
            fig2 = self.visualizer.scatter_plot(
                male_data, 'bmi', 'y_chr_concentration',
                title='Y Chromosome Concentration vs BMI',
                save_name='y_chr_vs_bmi' if save_figures else None
            )
            figures['bmi_scatter'] = fig2
        
        # 3. Regression plot: Y concentration vs Gestational weeks
        if 'gestational_weeks' in male_data.columns:
            fig3 = self.visualizer.regression_plot(
                male_data, 'gestational_weeks', 'y_chr_concentration',
                title='Y Chromosome Concentration vs Gestational Weeks (with Regression)',
                save_name='y_chr_gestational_regression' if save_figures else None
            )
            figures['gestational_regression'] = fig3
        
        # 4. Regression plot: Y concentration vs BMI
        if 'bmi' in male_data.columns:
            fig4 = self.visualizer.regression_plot(
                male_data, 'bmi', 'y_chr_concentration',
                title='Y Chromosome Concentration vs BMI (with Regression)',
                save_name='y_chr_bmi_regression' if save_figures else None
            )
            figures['bmi_regression'] = fig4
        
        # 5. Distribution of Y chromosome concentration
        fig5 = self.visualizer.distribution_plot(
            male_data, 'y_chr_concentration',
            title='Distribution of Y Chromosome Concentration (Male Fetuses)',
            save_name='y_chr_distribution' if save_figures else None
        )
        figures['y_chr_distribution'] = fig5
        
        # 6. Box plot: Y concentration by BMI group
        if 'bmi_group' in male_data.columns:
            fig6 = self.visualizer.box_plot(
                male_data, 'bmi_group', 'y_chr_concentration',
                title='Y Chromosome Concentration by BMI Group',
                save_name='y_chr_by_bmi_group' if save_figures else None
            )
            figures['bmi_group_boxplot'] = fig6
        
        # 7. Correlation heatmap
        numeric_cols = ['y_chr_concentration', 'gestational_weeks', 'bmi', 'age']
        available_cols = [col for col in numeric_cols if col in male_data.columns]
        
        if len(available_cols) >= 2:
            fig7 = self.visualizer.correlation_heatmap(
                male_data[available_cols],
                title='Correlation Matrix: Y Chromosome Concentration and Maternal Factors',
                save_name='correlation_heatmap' if save_figures else None
            )
            figures['correlation_heatmap'] = fig7
        
        self.results['visualizations'] = figures
        
        return figures
    
    def generate_summary_report(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive summary report for Problem 1.
        
        Args:
            data: DataFrame with NIPT data
            
        Returns:
            Dictionary with complete analysis summary
        """
        # Run all analyses
        correlation_results = self.analyze_correlations(data)
        regression_results = self.build_regression_model(data)
        significance_results = self.test_significance(data)
        
        # Compile summary
        summary = {
            'problem': 'Problem 1: Y Chromosome Concentration Relationship Analysis',
            'objective': 'Analyze correlation between Y chromosome concentration and maternal factors',
            'sample_size': correlation_results['sample_size'],
            'key_findings': {
                'significant_correlations': correlation_results['significant_features'],
                'model_performance': {
                    'r2_score': regression_results['test_r2'],
                    'features_used': regression_results['features_used']
                },
                'model_significance': regression_results.get('model_significant', 'Unknown')
            },
            'detailed_results': {
                'correlations': correlation_results,
                'regression': regression_results,
                'significance_tests': significance_results
            },
            'recommendations': self._generate_recommendations()
        }
        
        self.results['summary'] = summary
        
        return summary
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on analysis results."""
        recommendations = []
        
        if 'correlation_analysis' in self.results:
            sig_features = self.results['correlation_analysis']['significant_features']
            
            if 'gestational_weeks' in sig_features:
                recommendations.append(
                    "Gestational weeks show significant correlation with Y chromosome concentration."
                )
            
            if 'bmi' in sig_features:
                recommendations.append(
                    "BMI shows significant correlation with Y chromosome concentration."
                )
            
            if len(sig_features) == 0:
                recommendations.append(
                    "No significant correlations found. Consider additional factors or larger sample size."
                )
        
        if 'regression_model' in self.results:
            r2 = self.results['regression_model'].get('test_r2', 0)
            if r2 > 0.5:
                recommendations.append(
                    f"Regression model shows good predictive power (R² = {r2:.3f})."
                )
            elif r2 > 0.3:
                recommendations.append(
                    f"Regression model shows moderate predictive power (R² = {r2:.3f})."
                )
            else:
                recommendations.append(
                    f"Regression model shows limited predictive power (R² = {r2:.3f}). "
                    "Consider non-linear relationships or additional variables."
                )
        
        return recommendations
