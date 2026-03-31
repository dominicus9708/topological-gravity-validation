from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(".")
OBS_CANDIDATES = [
    BASE / r"data\derived\Echo of Light\input\T Pyxidis\t_pyx_standard_observation_epochs_original_paper.csv",
    BASE / "t_pyx_standard_observation_epochs_original_paper.csv",
    BASE / "t_pyx_standard_observation_epochs_processed.csv",
]
PATCH_CANDIDATES = [
    BASE / r"data\derived\Echo of Light\input\T Pyxidis\t_pyx_standard_echo_patches_original_paper.csv",
    BASE / "t_pyx_standard_echo_patches_original_paper.csv",
    BASE / "t_pyx_standard_echo_patches_processed.csv",
]

def first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    raise FileNotFoundError(f"No matching input file found among: {paths}")

def classify_z(row):
    z = row["z_1e17_cm"]
    err = row["z_err_1e17_cm"]
    if pd.isna(z) or pd.isna(err):
        return "undetermined"
    if abs(z) <= err:
        return "near_ring_plane"
    if z < -err:
        return "observer_side"
    if z > err:
        return "far_side"
    return "undetermined"

def classify_geometry(row):
    z_class = row["z_position_class"]
    if z_class == "near_ring_plane":
        return "ring_plane_patch"
    if z_class == "observer_side":
        return "inclined_observer_side_patch"
    if z_class == "far_side":
        return "inclined_far_side_patch"
    return "undetermined"

def standard_note(row):
    patch = row["echo_patch"]
    z_class = row["z_position_class"]
    theta = row["theta_arcsec"]
    delay = row["delay_time_days"]
    if z_class == "near_ring_plane":
        return (
            f"{patch}: delay={delay:.0f} d, theta={theta:.2f} arcsec. "
            "Standard geometric reading: patch lies close to the ring plane; "
            "the delayed illumination is interpreted as scattered light from the nova flash "
            "reaching material near the main ring surface."
        )
    if z_class == "observer_side":
        return (
            f"{patch}: delay={delay:.0f} d, theta={theta:.2f} arcsec. "
            "Standard geometric reading: patch is displaced toward the observer relative to the main ring plane; "
            "this supports the inclined-west-side interpretation rather than any real superluminal transport."
        )
    if z_class == "far_side":
        return (
            f"{patch}: delay={delay:.0f} d, theta={theta:.2f} arcsec. "
            "Standard geometric reading: patch is displaced away from the observer relative to the main ring plane."
        )
    return (
        f"{patch}: delay={delay:.0f} d, theta={theta:.2f} arcsec. "
        "Standard geometric reading remains uncertain from the current error bounds."
    )

def main():
    obs_path = first_existing(OBS_CANDIDATES)
    patch_path = first_existing(PATCH_CANDIDATES)

    obs = pd.read_csv(obs_path, encoding="utf-8-sig")
    patches = pd.read_csv(patch_path, encoding="utf-8-sig")

    for col in ["delay_time_days", "theta_arcsec", "z_1e17_cm", "z_err_1e17_cm"]:
        patches[col] = pd.to_numeric(patches[col], errors="coerce")

    patches["z_position_class"] = patches.apply(classify_z, axis=1)
    patches["standard_geometry_role"] = patches.apply(classify_geometry, axis=1)
    patches["theta_rank_small_to_large"] = patches["theta_arcsec"].rank(method="dense").astype("Int64")
    patches["delay_rank_small_to_large"] = patches["delay_time_days"].rank(method="dense").astype("Int64")
    patches["standard_interpretation_summary"] = patches.apply(standard_note, axis=1)

    obs["sequence_role"] = np.where(
        pd.to_numeric(obs["day_since_outburst"], errors="coerce") <= 282.8,
        "active_light_echo_epoch_or_followup",
        "late_followup_epoch",
    )

    interp_cols = [
        "echo_patch",
        "location_label",
        "delay_time_days",
        "delay_time_err_plus_days",
        "delay_time_err_minus_days",
        "theta_arcsec",
        "z_1e17_cm",
        "z_err_1e17_cm",
        "z_position_class",
        "standard_geometry_role",
        "theta_rank_small_to_large",
        "delay_rank_small_to_large",
        "hemisphere_group",
        "standard_interpretation_summary",
        "source_table",
    ]
    interp = patches[interp_cols].sort_values("delay_time_days").reset_index(drop=True)

    interp.to_csv("t_pyx_standard_interpretation_table.csv", index=False, encoding="utf-8-sig")
    obs.to_csv("t_pyx_standard_observation_timeline_table.csv", index=False, encoding="utf-8-sig")
    print("Created:")
    print("- t_pyx_standard_interpretation_table.csv")
    print("- t_pyx_standard_observation_timeline_table.csv")

if __name__ == "__main__":
    main()
