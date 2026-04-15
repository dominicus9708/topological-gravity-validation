"""
Data loading utilities
"""

from pathlib import Path
import pandas as pd

def load_rotation_curve_file(file_path: str | Path):
    return pd.read_csv(file_path)
