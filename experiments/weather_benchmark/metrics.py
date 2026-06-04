import numpy as np


def calculate_metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    diff = pred - true
    mse = float(np.mean(diff**2))
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(mse))
    rse_denominator = np.sqrt(np.sum((true - true.mean()) ** 2))
    rse = float(np.sqrt(np.sum(diff**2)) / max(rse_denominator, 1e-12))

    denominator = np.where(np.abs(true) < 1e-6, np.nan, true)
    mape = float(np.nanmean(np.abs(diff / denominator)))
    mspe = float(np.nanmean(np.square(diff / denominator)))

    centered_true = true - true.mean(axis=0)
    centered_pred = pred - pred.mean(axis=0)
    corr_denominator = np.sqrt((centered_true**2 * centered_pred**2).sum(axis=0)) + 1e-12
    corr = float(np.nanmean(0.01 * ((centered_true * centered_pred).sum(axis=0) / corr_denominator).mean(axis=-1)))

    return {"MSE": mse, "MAE": mae, "RMSE": rmse, "RSE": rse, "MAPE": mape, "MSPE": mspe, "Corr mean": corr}

