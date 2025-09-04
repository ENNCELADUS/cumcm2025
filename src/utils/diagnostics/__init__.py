"""
Data Quality Diagnostics Module

This module provides tools for diagnosing the impact of data filtering on statistical relationships.
Useful for understanding whether strict quality filters introduce bias or improve signal quality.

Main functions:
- filter_impact_analysis: Test correlations at each filtering stage
- selection_bias_analysis: Compare kept vs removed samples
- gc_correlation_analysis: Test if GC content correlates with key variables
"""

from .filter_impact import test_filter_impact
from .selection_bias import test_selection_bias  
from .gc_correlation import test_gc_correlations

__all__ = [
    'test_filter_impact',
    'test_selection_bias', 
    'test_gc_correlations'
]
