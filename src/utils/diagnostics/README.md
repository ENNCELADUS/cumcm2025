# Data Quality Diagnostics Module

This module provides comprehensive tools for diagnosing the impact of data filtering on statistical relationships. It helps determine whether strict quality filters introduce bias or improve signal quality.

## Quick Start

### Run All Diagnostics (Recommended)

```bash
# From project root
python run_diagnostics.py

# With options
python run_diagnostics.py --quiet      # Minimal output
python run_diagnostics.py --no-save    # Don't save plots/results
```

### Run Individual Tests

```python
from src.utils.diagnostics import test_filter_impact, test_selection_bias, test_gc_correlations

# Test correlation changes at each filter stage
results_df, raw_data, clean_data = test_filter_impact()

# Test for selection bias in removed samples
original, final, bias_detected = test_selection_bias()

# Test if GC content correlates with key variables
corr_df, data, gc_bias = test_gc_correlations()
```

## Test Descriptions

### 1. Filter Impact Analysis (`filter_impact.py`)

**Purpose**: Tests how correlations change at each filtering stage to identify which filters cause signal loss or bias.

**Key Outputs**:
- Correlation coefficients at each filter stage
- Fisher z-tests for significant changes
- Sample retention rates
- Assessment of substantial vs minimal changes

**Interpretation**:
- Large correlation changes (>20%) may indicate filter bias
- Small changes suggest filters improve data quality
- Statistical significance tests help distinguish real vs random changes

### 2. Selection Bias Analysis (`selection_bias.py`)

**Purpose**: Compares characteristics of kept vs removed samples to detect systematic differences.

**Key Outputs**:
- T-tests and Mann-Whitney U tests comparing groups
- Distribution plots for visual comparison
- Bias detection flags for different variables
- Sample size impacts

**Interpretation**:
- Significant differences in weeks/BMI/Y_concentration indicate bias
- GC content differences are expected (filtering criterion)
- Visual plots help identify distribution shifts

### 3. GC Correlation Analysis (`gc_correlation.py`)

**Purpose**: Tests if GC content correlates with key variables (weeks, BMI, Y_concentration).

**Key Outputs**:
- Pearson and Spearman correlations
- Scatter plots with trend lines
- Bias assessment and recommendations
- Filter threshold evaluation

**Interpretation**:
- Significant GC correlations suggest filtering bias potential
- Non-significant correlations support filter validity
- Visualization helps assess relationship patterns

## Output Files

The diagnostics generate several output files in `output/`:

### Figures (`output/figures/`)
- `selection_bias_comparison.png` - Distribution comparisons
- `gc_correlations.png` - GC content relationship plots

### Results (`output/results/`)
- `filter_impact_results.csv` - Correlation changes by stage
- `gc_correlation_results.csv` - GC correlation matrix

## Understanding the Results

### Overall Assessment Framework

The comprehensive summary provides a final recommendation based on:

1. **Filter Impact**: How much correlations changed
2. **Selection Bias**: Whether removed samples differ systematically
3. **GC Correlations**: Whether GC filtering may introduce bias

### Possible Outcomes

**✅ FILTERING APPEARS SOUND**
- No major bias indicators detected
- Filters improve rather than harm analysis
- Weak correlations likely reflect true biology

**⚠️ CAUTION: Some bias indicators detected**
- Consider reporting both filtered and unfiltered results
- Use inverse probability weighting if bias is substantial
- Consider relaxing GC thresholds

## Advanced Usage

### Custom Data Files

```python
# Use different data file
test_filter_impact(data_file_path="path/to/your/data.xlsx")
```

### Programmatic Access

```python
from src.utils.diagnostics.run_all import run_all_diagnostics

# Get structured results
results = run_all_diagnostics(verbose=False, save_outputs=False)

# Access specific test results
filter_results = results['filter_impact']['results_df']
bias_detected = results['selection_bias']['bias_detected']
gc_bias = results['gc_correlation']['bias_detected']
```

## Technical Details

### Statistical Methods

- **Fisher z-transform**: Tests correlation coefficient differences
- **T-tests**: Compare means between groups
- **Mann-Whitney U**: Compare distributions (non-parametric)
- **Pearson correlation**: Linear relationships
- **Spearman correlation**: Monotonic relationships

### Filter Stages Tested

1. **Raw Data**: No filters applied
2. **Age Filter**: Gestational weeks 10-25
3. **GC Filter**: GC content 40-60% (most impactful)
4. **Aneuploidy Filter**: Remove chromosomal abnormalities

### Dependencies

- pandas: Data manipulation
- numpy: Numerical operations
- scipy.stats: Statistical tests
- matplotlib: Plotting
- pathlib: File handling

## Troubleshooting

### Import Errors
If you get import errors, ensure you're running from the project root:
```bash
cd /path/to/cumcm
python run_diagnostics.py
```

### Missing Data
The scripts expect male fetus data in Excel format with specific column names (Chinese). Ensure your data file matches the expected format.

### Plot Saving Issues
Plots are saved to `output/figures/`. The script creates directories automatically, but ensure you have write permissions.
