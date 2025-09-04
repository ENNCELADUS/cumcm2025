"""
Configuration settings for NIPT analysis project.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "src"
OUTPUT_DIR = PROJECT_ROOT / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
RESULTS_DIR = OUTPUT_DIR / "results"

# Data file paths
DATA_FILE = DATA_DIR / "attachment.xlsx"

# Create output directories if they don't exist
OUTPUT_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Analysis parameters
class AnalysisConfig:
    """Configuration parameters for analysis."""
    
    # Y chromosome concentration threshold for accuracy
    Y_CHROMOSOME_THRESHOLD = 4.0  # percentage
    
    # Gestational age ranges for risk assessment
    LOW_RISK_WEEKS = 12  # weeks
    MEDIUM_RISK_WEEKS = 27  # weeks
    HIGH_RISK_WEEKS = 28  # weeks and above
    
    # BMI grouping (default ranges from problem description)
    BMI_GROUPS = [
        (20, 28),   # Group 1: [20, 28)
        (28, 32),   # Group 2: [28, 32)
        (32, 36),   # Group 3: [32, 36)
        (36, 40),   # Group 4: [36, 40)
        (40, float('inf'))  # Group 5: 40+
    ]
    
    # Statistical significance level
    ALPHA = 0.05
    
    # Random seed for reproducibility
    RANDOM_SEED = 42

# Column names mapping (Chinese to English for easier coding)
COLUMN_MAPPING = {
    '序号': 'sample_id',
    '孕妇代码': 'patient_code', 
    '年龄': 'age',
    '身高': 'height',
    '体重': 'weight',
    '末次月经': 'last_menstrual_period',
    'IVF妊娠': 'ivf_pregnancy',
    '检测日期': 'test_date',
    '检测抽血次数': 'blood_draw_count',
    '检测孕周': 'gestational_weeks',
    '孕妇BMI': 'bmi',
    '原始读段数': 'raw_read_count',
    '在参考基因组上比对的比例': 'mapping_ratio',
    '重复读段的比例': 'duplicate_ratio',
    '唯一比对的读段数': 'unique_mapped_reads',
    'GC含量': 'gc_content',
    '13号染色体的Z值': 'chr13_z_value',
    '18号染色体的Z值': 'chr18_z_value', 
    '21号染色体的Z值': 'chr21_z_value',
    'X染色体的Z值': 'x_chr_z_value',
    'Y染色体的Z值': 'y_chr_z_value',
    'Y染色体浓度': 'y_chr_concentration',
    'X染色体浓度': 'x_chr_concentration',
    '13号染色体的GC含量': 'chr13_gc_content',
    '18号染色体的GC含量': 'chr18_gc_content',
    '21号染色体的GC含量': 'chr21_gc_content',
    '被过滤掉读段数的比例': 'filtered_reads_ratio',
    '染色体的非整倍体': 'chromosome_aneuploidy',
    '怀孕次数': 'pregnancy_count',
    '生产次数': 'birth_count',
    '胎儿是否健康': 'fetal_health'
}
