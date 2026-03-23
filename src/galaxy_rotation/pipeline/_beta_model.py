import numpy as np
import pandas as pd



def get_constant_beta(value: float = 200.0) -> float:
    return float(value)



def get_structural_beta(
    galaxy_data: pd.DataFrame,
    sigma_profile: np.ndarray,
) -> float:
    del galaxy_data
    sigma_profile = np.asarray(sigma_profile, dtype=float)
    sigma_mean = float(np.mean(np.abs(sigma_profile))) if sigma_profile.size else 0.0
    return max(1.0, sigma_mean)
