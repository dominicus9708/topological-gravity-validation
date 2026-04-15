from __future__ import annotations

import numpy as np


def _to_float_array(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return np.where(np.isfinite(arr), arr, np.nan)


def _valid_pair_mask(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.isfinite(y_true) & np.isfinite(y_pred)


def rmse(y_true, y_pred) -> float:
    """
    Root Mean Squared Error.
    단위는 입력과 동일합니다. (회전속도면 km/s)
    """
    yt = _to_float_array(y_true)
    yp = _to_float_array(y_pred)

    mask = _valid_pair_mask(yt, yp)
    if not np.any(mask):
        return float("nan")

    diff = yt[mask] - yp[mask]
    return float(np.sqrt(np.mean(diff ** 2)))


def mae(y_true, y_pred) -> float:
    """
    Mean Absolute Error.
    """
    yt = _to_float_array(y_true)
    yp = _to_float_array(y_pred)

    mask = _valid_pair_mask(yt, yp)
    if not np.any(mask):
        return float("nan")

    return float(np.mean(np.abs(yt[mask] - yp[mask])))


def mean_absolute_fractional_error(y_true, y_pred, floor: float = 1.0e-12) -> float:
    """
    Mean Absolute Fractional Error:
        mean( |y_true - y_pred| / max(|y_true|, floor) )

    y_true가 0에 가까운 경우를 대비해 floor를 둡니다.
    """
    yt = _to_float_array(y_true)
    yp = _to_float_array(y_pred)

    mask = _valid_pair_mask(yt, yp)
    if not np.any(mask):
        return float("nan")

    denom = np.maximum(np.abs(yt[mask]), float(floor))
    frac = np.abs(yt[mask] - yp[mask]) / denom
    return float(np.mean(frac))


def median_absolute_fractional_error(y_true, y_pred, floor: float = 1.0e-12) -> float:
    """
    Median Absolute Fractional Error:
        median( |y_true - y_pred| / max(|y_true|, floor) )
    """
    yt = _to_float_array(y_true)
    yp = _to_float_array(y_pred)

    mask = _valid_pair_mask(yt, yp)
    if not np.any(mask):
        return float("nan")

    denom = np.maximum(np.abs(yt[mask]), float(floor))
    frac = np.abs(yt[mask] - yp[mask]) / denom
    return float(np.median(frac))


def chi_square(y_true, y_pred, y_err, floor: float = 1.0e-12) -> float:
    """
    Chi-square:
        sum( ((y_true - y_pred) / sigma)^2 )

    여기서 sigma = max(y_err, floor).
    """
    yt = _to_float_array(y_true)
    yp = _to_float_array(y_pred)
    ye = _to_float_array(y_err)

    mask = np.isfinite(yt) & np.isfinite(yp) & np.isfinite(ye)
    if not np.any(mask):
        return float("nan")

    sigma = np.maximum(np.abs(ye[mask]), float(floor))
    chi2 = np.sum(((yt[mask] - yp[mask]) / sigma) ** 2)
    return float(chi2)


def reduced_chi_square(
    y_true,
    y_pred,
    y_err,
    n_params: int = 1,
    floor: float = 1.0e-12,
) -> float:
    """
    Reduced chi-square:
        chi2 / dof

    dof = N_valid - n_params
    n_params는 모델의 유효 자유 파라미터 수로 해석합니다.
    기본값은 beta 1개를 반영하여 1로 둡니다.
    """
    yt = _to_float_array(y_true)
    yp = _to_float_array(y_pred)
    ye = _to_float_array(y_err)

    mask = np.isfinite(yt) & np.isfinite(yp) & np.isfinite(ye)
    n_valid = int(np.sum(mask))
    dof = n_valid - int(n_params)

    if n_valid == 0 or dof <= 0:
        return float("nan")

    chi2 = chi_square(yt[mask], yp[mask], ye[mask], floor=floor)
    return float(chi2 / dof)


def valid_point_count(*arrays) -> int:
    """
    여러 배열에 대해 동시에 유효한 점 개수를 반환합니다.
    """
    if len(arrays) == 0:
        return 0

    masks = []
    for arr in arrays:
        a = _to_float_array(arr)
        masks.append(np.isfinite(a))

    mask = masks[0]
    for m in masks[1:]:
        mask = mask & m

    return int(np.sum(mask))
