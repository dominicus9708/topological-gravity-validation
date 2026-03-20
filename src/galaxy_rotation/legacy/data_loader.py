"""
Data loading utilities
"""

import pandas as pd


def load_rotation_curve_file(file_path):
    import pandas as pd
    return pd.read_csv(file_path)
