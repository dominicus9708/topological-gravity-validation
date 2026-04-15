from __future__ import annotations

import pandas as pd

from galaxy_rotation_legacy_units import (
    to_kpc,
    to_kmps,
    velocity_squared_over_radius,
)


# ----- default SPARC unit map -----

DEFAULT_UNIT_MAP = {
    "r": "kpc",
    "v_obs": "km/s",
    "v_err": "km/s",
    "v_gas": "km/s",
    "v_disk": "km/s",
    "v_bul": "km/s",
}


# ----- required schema -----

REQUIRED_COLUMNS = [
    "r",
    "v_obs",
    "v_err",
    "v_gas",
    "v_disk",
    "v_bul",
]


def validate_input_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required SPARC columns: {missing}")


def normalize_sparc_dataframe(
    df: pd.DataFrame,
    unit_map: dict | None = None,
    galaxy_name: str | None = None,
) -> pd.DataFrame:
    validate_input_columns(df)

    if unit_map is None:
        unit_map = DEFAULT_UNIT_MAP.copy()

    out = pd.DataFrame()

    out["r_kpc"] = to_kpc(df["r"].values, unit_map["r"])
    out["v_obs_kmps"] = to_kmps(df["v_obs"].values, unit_map["v_obs"])
    out["v_err_kmps"] = to_kmps(df["v_err"].values, unit_map["v_err"])
    out["v_gas_kmps"] = to_kmps(df["v_gas"].values, unit_map["v_gas"])
    out["v_disk_kmps"] = to_kmps(df["v_disk"].values, unit_map["v_disk"])
    out["v_bul_kmps"] = to_kmps(df["v_bul"].values, unit_map["v_bul"])

    out["a_obs_kmps2_per_kpc"] = velocity_squared_over_radius(
        out["v_obs_kmps"].values,
        out["r_kpc"].values,
    )

    name = galaxy_name if galaxy_name is not None else "unknown"
    out["galaxy"] = [name] * len(out)

    out = out[
        [
            "galaxy",
            "r_kpc",
            "v_obs_kmps",
            "v_err_kmps",
            "v_gas_kmps",
            "v_disk_kmps",
            "v_bul_kmps",
            "a_obs_kmps2_per_kpc",
        ]
    ]

    return out


def validate_normalized_sparc(df_norm: pd.DataFrame) -> None:
    required = [
        "galaxy",
        "r_kpc",
        "v_obs_kmps",
        "v_err_kmps",
        "v_gas_kmps",
        "v_disk_kmps",
        "v_bul_kmps",
        "a_obs_kmps2_per_kpc",
    ]

    missing = [col for col in required if col not in df_norm.columns]
    if missing:
        raise ValueError(f"Missing normalized columns: {missing}")

    if df_norm.empty:
        raise ValueError("Normalized dataframe is empty.")

    if df_norm.isnull().any().any():
        print("\n[NaN column counts]")
        print(df_norm.isnull().sum())

        print("\n[Rows containing NaN]")
        print(df_norm[df_norm.isnull().any(axis=1)])

        raise ValueError("Normalized dataframe contains NaN values.")

    if (df_norm["r_kpc"] <= 0).any():
        bad_rows = df_norm[df_norm["r_kpc"] <= 0]
        print("\n[Rows with non-positive radius]")
        print(bad_rows)
        raise ValueError("Radius values must all be positive.")

    if (df_norm["v_err_kmps"] <= 0).any():
        bad_rows = df_norm[df_norm["v_err_kmps"] <= 0]
        print("\n[Rows with non-positive velocity error]")
        print(bad_rows)
        raise ValueError("Velocity errors must all be positive.")

    if (df_norm["v_obs_kmps"] < 0).any():
        bad_rows = df_norm[df_norm["v_obs_kmps"] < 0]
        print("\n[Rows with negative observed velocity]")
        print(bad_rows)
        raise ValueError("Observed velocities must be non-negative.")


def sort_by_radius(df_norm: pd.DataFrame) -> pd.DataFrame:
    return df_norm.sort_values("r_kpc").reset_index(drop=True)