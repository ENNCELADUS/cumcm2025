# NIPT Analysis Project

A comprehensive statistical analysis framework for Non-invasive Prenatal Testing (NIPT) data, addressing four key problems related to NIPT timing optimization and fetal abnormality detection. This project implements advanced mixed-effects modeling, non-linear regression, and clinical decision support tools.

## Project Structure

```
src/
├── __init__.py                 # Main package initialization
├── main.py                     # Command-line interface
├── config/
│   └── settings.py            # Configuration and constants
├── data/
│   ├── __init__.py
│   ├── loader.py              # Data loading and preprocessing
│   └── attachment.xlsx        # Raw data file
├── models/                    # Statistical and ML models (future)
├── analysis/                  # Problem-specific analysis modules
│   ├── problem1/              # Y chromosome correlation analysis
│   │   ├── __init__.py
│   │   └── correlation_analysis.py
│   ├── problem2/              # BMI grouping for optimal timing
│   ├── problem3/              # Multi-factor optimization
│   └── problem4/              # Female fetus abnormality detection
├── utils/
│   ├── __init__.py
│   ├── visualization.py       # Plotting and visualization utilities
│   └── statistics.py          # Statistical analysis utilities
└── notebooks/
    └── 01_data_exploration.ipynb  # Jupyter notebooks for analysis

output/
├── figures/                   # Generated plots and visualizations
└── results/                   # Analysis results and reports
```

## Problems Addressed

### Problem 1: Y Chromosome Concentration Relationship Analysis ✅ **COMPLETED**
- **Objective**: Analyze correlation between fetal Y chromosome concentration and maternal factors
- **Implementation**: Advanced mixed-effects models with natural splines
- **Key Results**: 
  - Final model: `Y_concentration ~ bs(weeks, df=3) + BMI + (1|patient_id)`
  - Conditional R²: 82.9% (Marginal R²: 6.3%)
  - Strong evidence for non-linear gestational age effects
  - BMI shows consistent negative association
- **Clinical Impact**: Provides timing optimization recommendations for different BMI groups
- **Module**: `src/analysis/problem1/` + `src/notebooks/01_data_exploration.ipynb`

### Problem 2: BMI-based Grouping for Optimal NIPT Timing (Male Fetuses) 📋 **PLANNED**
- Group male fetus pregnancies by BMI for optimal NIPT timing
- Minimize potential risk while ensuring accuracy
- **Module**: `src/analysis/problem2/` (to be implemented)

### Problem 3: Multi-factor NIPT Timing Optimization 📋 **PLANNED**
- Consider multiple factors (height, weight, age, etc.) for NIPT timing
- Comprehensive risk minimization approach
- **Module**: `src/analysis/problem3/` (to be implemented)

### Problem 4: Female Fetus Abnormality Detection 📋 **PLANNED**
- Develop methods for detecting abnormalities in female fetuses
- Use chromosome Z-values, GC content, and other factors
- **Module**: `src/analysis/problem4/` (to be implemented)

## Environment Setup

### Using Conda (Recommended)

1. Create and activate the environment:
```bash
conda create -n cumcm-env python=3.11 pandas numpy scipy matplotlib seaborn scikit-learn statsmodels patsy openpyxl xlrd jupyterlab ipykernel -y
conda activate cumcm-env
```

2. Navigate to project directory:
```bash
cd /path/to/cumcm
```

3. Install additional dependencies via pip:
```bash
pip install -r requirements.txt
```

### Using pip Only

1. Create virtual environment:
```bash
python -m venv cumcm-env
source cumcm-env/bin/activate  # On Windows: cumcm-env\Scripts\activate
```

2. Install all dependencies:
```bash
pip install -r requirements.txt
```

### Verification

Test your installation:
```bash
conda activate cumcm-env
python -c "import pandas, numpy, statsmodels, matplotlib, seaborn; print('✅ All packages installed successfully')"
```

## Usage

### Jupyter Notebooks (Primary Interface)

```bash
# Activate environment
conda activate cumcm-env

# Start JupyterLab
jupyter lab

# Open the main analysis notebook
# Navigate to src/notebooks/01_data_exploration.ipynb
```

### Command Line Interface

```bash
# Activate environment
conda activate cumcm-env

# Run diagnostic utilities
python run_diagnostics.py

# Or run specific analysis modules
python src/main.py
```

### Key Analysis Files

1. **Data Preprocessing**: `src/notebooks/00_data_preprocessing.ipynb`
2. **Main Analysis**: `src/notebooks/01_data_exploration.ipynb` (⭐ **Primary Results**)
3. **Analysis Summary**: `summary/p1.md` (⭐ **Comprehensive Report**)
4. **Project Plan**: `plan/prob1.md`

## Data Description

The project uses NIPT data with the following key variables:

- **Maternal factors**: Age, height, weight, BMI, gestational weeks
- **Test data**: Y/X chromosome concentrations, Z-values for chromosomes 13/18/21
- **Quality metrics**: GC content, read counts, mapping ratios
- **Outcomes**: Fetal health status, chromosome aneuploidy

## Key Features

### 🔬 **Advanced Statistical Modeling**
- **Mixed-effects models** with random intercepts for patient clustering (ICC ≈ 0.71)
- **Non-linear regression** using natural splines for gestational age effects
- **Model diagnostics** with marginal/conditional R² decomposition
- **Clinical threshold analysis** with logistic regression for binary decision support

### 📊 **Comprehensive Data Analysis**
- **Exploratory Data Analysis** with correlation and distribution analysis
- **Effect visualization** with partial effects plots and confidence intervals
- **Clinical interpretation** with scenario-based prediction tables
- **Robust statistical inference** with cluster-robust standard errors

### 🎨 **Publication-Ready Visualizations**
- **Four-panel diagnostic plots** with residual analysis
- **Partial effects visualization** with BMI stratification
- **Clinical threshold heatmaps** for decision support
- **Automatic figure generation** with consistent styling

### 🔧 **Robust Implementation**
- **Comprehensive data preprocessing** with automatic type conversion
- **Fallback mechanisms** for model convergence issues
- **Extensive error handling** and diagnostic reporting
- **Modular design** for easy extension and maintenance

## Configuration

Key analysis parameters can be modified in `src/config/settings.py`:

- Y chromosome concentration threshold (default: 4.0%)
- BMI grouping ranges
- Statistical significance level (default: 0.05)
- Risk categorization thresholds

## Output

### 📊 **Generated Analyses (Problem 1)**

**Key Visualizations** (`output/figures/`):
- `p1_comprehensive_partial_effects.png` - Four-panel partial effects analysis ⭐
- `p1_model_diagnostics.png` - Mixed-effects model diagnostics
- `p1_scatter_*.png` - Exploratory scatter plots  
- `p1_distributions.png` - Variable distribution analysis

**Statistical Results** (`output/results/`):
- `p1_final_model_summary.csv` - Final mixed-effects model coefficients ⭐
- `p1_clinical_interpretation_*.csv` - Clinical scenario predictions ⭐
- `p1_model_comparison.csv` - Comprehensive model comparisons
- `p1_*_model_results.csv` - Detailed statistical outputs

**Comprehensive Reports** (`summary/`):
- `p1.md` - Complete analysis summary with interpretation ⭐
- `preprocess.md` - Data preprocessing documentation

### 🎯 **Key Results Summary**
- **Final Model**: Mixed-effects with natural splines (R² = 82.9%)
- **Clinical Impact**: BMI-stratified timing recommendations
- **Statistical Rigor**: Comprehensive diagnostics and validation
- **Reproducibility**: Complete analysis pipeline in Jupyter notebooks

## License

This project is developed for the CUMCM (China Undergraduate Mathematical Contest in Modeling) competition.
