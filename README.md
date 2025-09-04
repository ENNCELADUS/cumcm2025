# NIPT Analysis Project

A comprehensive analysis toolkit for Non-invasive Prenatal Testing (NIPT) data, addressing four key problems related to NIPT time point selection and fetal abnormality detection.

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

### Problem 1: Y Chromosome Concentration Relationship Analysis
- Analyze correlation between fetal Y chromosome concentration and maternal factors (gestational weeks, BMI, etc.)
- Build relationship models and test their significance
- **Module**: `src/analysis/problem1/`

### Problem 2: BMI-based Grouping for Optimal NIPT Timing (Male Fetuses)
- Group male fetus pregnancies by BMI for optimal NIPT timing
- Minimize potential risk while ensuring accuracy
- **Module**: `src/analysis/problem2/` (planned)

### Problem 3: Multi-factor NIPT Timing Optimization
- Consider multiple factors (height, weight, age, etc.) for NIPT timing
- Comprehensive risk minimization approach
- **Module**: `src/analysis/problem3/` (planned)

### Problem 4: Female Fetus Abnormality Detection
- Develop methods for detecting abnormalities in female fetuses
- Use chromosome Z-values, GC content, and other factors
- **Module**: `src/analysis/problem4/` (planned)

## Environment Setup

### Using Conda (Recommended)

1. Create and activate the environment:
```bash
conda create -n cumcm-env python=3.11 pandas numpy scipy matplotlib seaborn scikit-learn openpyxl xlrd jupyter -y
conda activate cumcm-env
```

2. Navigate to project directory:
```bash
cd /path/to/cumcm
```

### Using pip

1. Create virtual environment:
```bash
python -m venv cumcm-env
source cumcm-env/bin/activate  # On Windows: cumcm-env\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface

```bash
# Activate environment
conda activate cumcm-env

# Run data exploration
python src/main.py --explore

# Run specific problem analysis
python src/main.py --problem 1

# Run all analyses
python src/main.py --all
```

### Jupyter Notebooks

```bash
# Start Jupyter
jupyter lab

# Open the data exploration notebook
# Navigate to src/notebooks/01_data_exploration.ipynb
```

### Python API

```python
from src.data import NIPTDataLoader
from src.analysis.problem1 import YChromosomeCorrelationAnalyzer

# Load and preprocess data
loader = NIPTDataLoader()
data = loader.preprocess_data()

# Run Problem 1 analysis
analyzer = YChromosomeCorrelationAnalyzer()
results = analyzer.generate_summary_report(data)
```

## Data Description

The project uses NIPT data with the following key variables:

- **Maternal factors**: Age, height, weight, BMI, gestational weeks
- **Test data**: Y/X chromosome concentrations, Z-values for chromosomes 13/18/21
- **Quality metrics**: GC content, read counts, mapping ratios
- **Outcomes**: Fetal health status, chromosome aneuploidy

## Key Features

- **Comprehensive data preprocessing** with automatic type conversion and validation
- **Statistical analysis utilities** including correlation, regression, and significance testing
- **Rich visualizations** with automatic saving and customization
- **Modular design** for easy extension and maintenance
- **Jupyter notebook integration** for interactive analysis
- **Command-line interface** for batch processing

## Configuration

Key analysis parameters can be modified in `src/config/settings.py`:

- Y chromosome concentration threshold (default: 4.0%)
- BMI grouping ranges
- Statistical significance level (default: 0.05)
- Risk categorization thresholds

## Output

- **Figures**: Saved to `output/figures/` in PNG format
- **Results**: Analysis summaries and statistics
- **Models**: Trained models and coefficients (future)

## Development Status

- ✅ **Data loading and preprocessing**
- ✅ **Visualization utilities**
- ✅ **Statistical analysis utilities**
- ✅ **Problem 1 implementation**
- 🔄 **Problem 2-4 modules** (in development)
- 🔄 **Advanced modeling** (planned)

## Requirements

- Python 3.10+
- pandas, numpy, scipy
- matplotlib, seaborn
- scikit-learn
- openpyxl, xlrd
- jupyter (for notebooks)

## Contributing

1. Follow the modular structure for new analyses
2. Add comprehensive docstrings and type hints
3. Include unit tests for new functionality
4. Update this README for new features

## License

This project is developed for the CUMCM (China Undergraduate Mathematical Contest in Modeling) competition.
