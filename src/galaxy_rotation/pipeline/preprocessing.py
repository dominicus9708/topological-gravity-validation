import pandas as pd


def load_processed_sparc_table(path: str) -> pd.DataFrame:
    return pd.read_csv(path)
