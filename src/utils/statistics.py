"""
Statistical utilities for NIPT analysis.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from typing import Tuple, Dict, Any, Optional, List
import warnings

from ..config.settings import AnalysisConfig


class StatisticalAnalyzer:
    """
    Statistical analysis utilities for NIPT data.
    
    Provides methods for correlation analysis, regression diagnostics,
    hypothesis testing, and other statistical operations.
    """
    
    def __init__(self, alpha: float = AnalysisConfig.ALPHA):
        """
        Initialize the statistical analyzer.
        
        Args:
            alpha: Significance level for hypothesis tests
        """
        self.alpha = alpha
        
    def correlation_analysis(self, data: pd.DataFrame, target: str, 
                           features: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Perform correlation analysis between target and features.
        
        Args:
            data: DataFrame containing the data
            target: Target variable name
            features: List of feature names. If None, uses all numeric columns.
            
        Returns:
            Dictionary with correlation results
        """
        if features is None:
            features = data.select_dtypes(include=[np.number]).columns.tolist()
            if target in features:
                features.remove(target)
        
        results = {
            'correlations': {},
            'p_values': {},
            'significant_features': []
        }
        
        for feature in features:
            if feature in data.columns and target in data.columns:
                # Remove missing values for this pair
                clean_data = data[[target, feature]].dropna()
                
                if len(clean_data) < 3:  # Need at least 3 points for correlation
                    continue
                    
                # Calculate Pearson correlation
                corr, p_value = stats.pearsonr(clean_data[target], clean_data[feature])
                
                results['correlations'][feature] = corr
                results['p_values'][feature] = p_value
                
                if p_value < self.alpha:
                    results['significant_features'].append(feature)
        
        return results
    
    def linear_regression_analysis(self, data: pd.DataFrame, target: str, 
                                 features: List[str]) -> Dict[str, Any]:
        """
        Perform linear regression analysis.
        
        Args:
            data: DataFrame containing the data
            target: Target variable name
            features: List of feature names
            
        Returns:
            Dictionary with regression results
        """
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import StandardScaler
        
        # Prepare data
        clean_data = data[features + [target]].dropna()
        
        if len(clean_data) < len(features) + 5:  # Need enough samples
            raise ValueError("Insufficient data for regression analysis")
        
        X = clean_data[features]
        y = clean_data[target]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=AnalysisConfig.RANDOM_SEED
        )
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Fit model
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        results = {
            'model': model,
            'scaler': scaler,
            'coefficients': dict(zip(features, model.coef_)),
            'intercept': model.intercept_,
            'train_r2': r2_score(y_train, y_train_pred),
            'test_r2': r2_score(y_test, y_test_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'test_mae': mean_absolute_error(y_test, y_test_pred),
            'feature_importance': dict(zip(features, np.abs(model.coef_))),
            'data_shape': {
                'train': X_train.shape,
                'test': X_test.shape
            }
        }
        
        # Statistical significance tests
        results.update(self._regression_significance_tests(X_train_scaled, y_train, model))
        
        return results
    
    def _regression_significance_tests(self, X: np.ndarray, y: np.ndarray, 
                                     model) -> Dict[str, Any]:
        """
        Perform significance tests for regression coefficients.
        
        Args:
            X: Feature matrix
            y: Target vector
            model: Fitted linear regression model
            
        Returns:
            Dictionary with significance test results
        """
        n = X.shape[0]
        k = X.shape[1]
        
        # Predictions and residuals
        y_pred = model.predict(X)
        residuals = y - y_pred
        
        # Calculate standard errors
        mse = np.sum(residuals**2) / (n - k - 1)
        var_coef = mse * np.linalg.inv(X.T @ X)
        se_coef = np.sqrt(np.diag(var_coef))
        
        # t-statistics and p-values
        t_stats = model.coef_ / se_coef
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - k - 1))
        
        # F-statistic for overall model significance
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2 = 1 - (ss_res / ss_tot)
        f_stat = (r2 / k) / ((1 - r2) / (n - k - 1))
        f_p_value = 1 - stats.f.cdf(f_stat, k, n - k - 1)
        
        return {
            'coefficient_std_errors': se_coef,
            'coefficient_t_stats': t_stats,
            'coefficient_p_values': p_values,
            'significant_coefficients': p_values < self.alpha,
            'f_statistic': f_stat,
            'f_p_value': f_p_value,
            'model_significant': f_p_value < self.alpha
        }
    
    def anova_test(self, data: pd.DataFrame, dependent: str, 
                   independent: str) -> Dict[str, Any]:
        """
        Perform one-way ANOVA test.
        
        Args:
            data: DataFrame containing the data
            dependent: Dependent variable name
            independent: Independent variable name (categorical)
            
        Returns:
            Dictionary with ANOVA results
        """
        groups = []
        group_names = []
        
        for group_name in data[independent].unique():
            if pd.notna(group_name):
                group_data = data[data[independent] == group_name][dependent].dropna()
                if len(group_data) > 0:
                    groups.append(group_data)
                    group_names.append(group_name)
        
        if len(groups) < 2:
            raise ValueError("Need at least 2 groups for ANOVA")
        
        # Perform ANOVA
        f_stat, p_value = stats.f_oneway(*groups)
        
        # Calculate group statistics
        group_stats = {}
        for i, (group_name, group_data) in enumerate(zip(group_names, groups)):
            group_stats[group_name] = {
                'count': len(group_data),
                'mean': np.mean(group_data),
                'std': np.std(group_data, ddof=1),
                'median': np.median(group_data)
            }
        
        return {
            'f_statistic': f_stat,
            'p_value': p_value,
            'significant': p_value < self.alpha,
            'group_statistics': group_stats,
            'groups_tested': group_names
        }
    
    def t_test(self, group1: pd.Series, group2: pd.Series, 
               paired: bool = False) -> Dict[str, Any]:
        """
        Perform t-test between two groups.
        
        Args:
            group1: First group data
            group2: Second group data
            paired: Whether to perform paired t-test
            
        Returns:
            Dictionary with t-test results
        """
        # Clean data
        if paired:
            clean_data = pd.DataFrame({'group1': group1, 'group2': group2}).dropna()
            g1, g2 = clean_data['group1'], clean_data['group2']
            t_stat, p_value = stats.ttest_rel(g1, g2)
        else:
            g1, g2 = group1.dropna(), group2.dropna()
            
            # Check for equal variances
            _, levene_p = stats.levene(g1, g2)
            equal_var = levene_p > self.alpha
            
            t_stat, p_value = stats.ttest_ind(g1, g2, equal_var=equal_var)
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < self.alpha,
            'group1_stats': {
                'count': len(g1),
                'mean': np.mean(g1),
                'std': np.std(g1, ddof=1)
            },
            'group2_stats': {
                'count': len(g2),
                'mean': np.mean(g2),
                'std': np.std(g2, ddof=1)
            },
            'test_type': 'paired' if paired else 'independent'
        }
    
    def chi_square_test(self, data: pd.DataFrame, var1: str, 
                       var2: str) -> Dict[str, Any]:
        """
        Perform chi-square test of independence.
        
        Args:
            data: DataFrame containing the data
            var1: First categorical variable
            var2: Second categorical variable
            
        Returns:
            Dictionary with chi-square test results
        """
        # Create contingency table
        contingency_table = pd.crosstab(data[var1], data[var2])
        
        # Perform chi-square test
        chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        
        return {
            'chi2_statistic': chi2_stat,
            'p_value': p_value,
            'degrees_of_freedom': dof,
            'significant': p_value < self.alpha,
            'contingency_table': contingency_table,
            'expected_frequencies': expected,
            'cramer_v': self._calculate_cramers_v(chi2_stat, contingency_table)
        }
    
    def _calculate_cramers_v(self, chi2: float, contingency_table: pd.DataFrame) -> float:
        """Calculate Cramer's V effect size measure."""
        n = contingency_table.sum().sum()
        min_dim = min(contingency_table.shape) - 1
        return np.sqrt(chi2 / (n * min_dim))
    
    def normality_test(self, data: pd.Series) -> Dict[str, Any]:
        """
        Test for normality using Shapiro-Wilk test (for small samples) 
        or Kolmogorov-Smirnov test (for large samples).
        
        Args:
            data: Data series to test
            
        Returns:
            Dictionary with normality test results
        """
        clean_data = data.dropna()
        n = len(clean_data)
        
        results = {
            'sample_size': n,
            'mean': np.mean(clean_data),
            'std': np.std(clean_data, ddof=1),
            'skewness': stats.skew(clean_data),
            'kurtosis': stats.kurtosis(clean_data)
        }
        
        if n < 5000:  # Use Shapiro-Wilk for smaller samples
            stat, p_value = stats.shapiro(clean_data)
            results.update({
                'test_used': 'Shapiro-Wilk',
                'statistic': stat,
                'p_value': p_value,
                'is_normal': p_value > self.alpha
            })
        else:  # Use Kolmogorov-Smirnov for larger samples
            stat, p_value = stats.kstest(clean_data, 'norm')
            results.update({
                'test_used': 'Kolmogorov-Smirnov',
                'statistic': stat,
                'p_value': p_value,
                'is_normal': p_value > self.alpha
            })
        
        return results
    
    def bootstrap_confidence_interval(self, data: pd.Series, statistic: str = 'mean',
                                    n_bootstrap: int = 1000, 
                                    confidence_level: float = 0.95) -> Dict[str, Any]:
        """
        Calculate bootstrap confidence interval for a statistic.
        
        Args:
            data: Data series
            statistic: Type of statistic ('mean', 'median', 'std')
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            
        Returns:
            Dictionary with confidence interval results
        """
        clean_data = data.dropna().values
        n = len(clean_data)
        
        if n < 2:
            raise ValueError("Need at least 2 data points for bootstrap")
        
        # Define statistic function
        stat_func = {
            'mean': np.mean,
            'median': np.median,
            'std': lambda x: np.std(x, ddof=1)
        }[statistic]
        
        # Bootstrap sampling
        np.random.seed(AnalysisConfig.RANDOM_SEED)
        bootstrap_stats = []
        
        for _ in range(n_bootstrap):
            sample = np.random.choice(clean_data, size=n, replace=True)
            bootstrap_stats.append(stat_func(sample))
        
        bootstrap_stats = np.array(bootstrap_stats)
        
        # Calculate confidence interval
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ci_lower = np.percentile(bootstrap_stats, lower_percentile)
        ci_upper = np.percentile(bootstrap_stats, upper_percentile)
        
        return {
            'statistic': statistic,
            'original_value': stat_func(clean_data),
            'bootstrap_mean': np.mean(bootstrap_stats),
            'bootstrap_std': np.std(bootstrap_stats),
            'confidence_level': confidence_level,
            'confidence_interval': (ci_lower, ci_upper),
            'n_bootstrap': n_bootstrap,
            'sample_size': n
        }
