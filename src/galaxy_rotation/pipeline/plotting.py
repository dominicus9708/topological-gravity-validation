import matplotlib.pyplot as plt
import pandas as pd


def save_rotation_curve_plot(
    df: pd.DataFrame,
    galaxy_name: str,
    output_path: str,
) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(df["r"], df["v_obs"], label="Observed")
    plt.plot(df["r"], df["v_model"], label="Model")
    plt.xlabel("r")
    plt.ylabel("v")
    plt.title(f"Rotation Curve: {galaxy_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
