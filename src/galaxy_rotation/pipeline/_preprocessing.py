from __future__ import annotations

from pathlib import Path

import pandas as pd



def load_processed_sparc_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Processed SPARC table not found: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"Loaded SPARC table is empty: {path}")

    required_columns = [
        "galaxy",
        "r_kpc",
        "v_obs_kmps",
        "v_err_kmps",
        "v_gas_kmps",
        "v_disk_kmps",
        "v_bul_kmps",
    ]

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in SPARC table: {missing}")

    return df
