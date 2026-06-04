import json
from pathlib import Path

import numpy as np
import torch

from exponential_smoothing import exponential_smooth, predict_from_smoothed
from gru import GRUForecast
from linear import LinearForecast
from models import PatchTST
from patchtst_weather import build_config as build_patchtst_config
from weather_experiment_common import (
    BATCH_SIZE,
    BASE_DIR,
    PRED_LEN,
    SEQ_LEN,
    calculate_metrics,
    collect_torch_predictions,
    format_metrics,
    load_patchtst_weather,
    make_torch_loaders,
    numpy_targets,
    set_seed,
    train_torch_model,
    window_starts,
)


def validation_mse(pred, true):
    return float(np.mean((pred - true) ** 2))


def repeat_last_prediction(info, starts):
    data = info["data"]
    pred = data[starts + SEQ_LEN - 1]
    return np.repeat(pred[:, None, :], PRED_LEN, axis=1)


def holt_smooth(data, alpha, beta, damping):
    levels = np.empty_like(data)
    trends = np.empty_like(data)
    levels[0] = data[0]
    trends[0] = data[1] - data[0]
    for index in range(1, len(data)):
        previous_level = levels[index - 1]
        levels[index] = alpha * data[index] + (1.0 - alpha) * (levels[index - 1] + damping * trends[index - 1])
        trends[index] = beta * (levels[index] - previous_level) + (1.0 - beta) * damping * trends[index - 1]
    return levels, trends


def holt_predict(levels, trends, starts, damping):
    base = levels[starts + SEQ_LEN - 1]
    trend = trends[starts + SEQ_LEN - 1]
    horizons = np.arange(1, PRED_LEN + 1, dtype=np.float32)
    if damping == 1.0:
        factors = horizons
    else:
        factors = damping * (1.0 - damping**horizons) / (1.0 - damping)
    return base[:, None, :] + factors[None, :, None] * trend[:, None, :]


def seasonal_naive_prediction(info, starts, period, trend=False):
    data = info["data"]
    horizon_offsets = np.arange(PRED_LEN)
    seasonal_index = starts[:, None] + SEQ_LEN - period + horizon_offsets[None, :]
    pred = data[seasonal_index]
    if trend:
        recent = data[starts + SEQ_LEN - 1]
        previous = data[starts + SEQ_LEN - period - 1]
        drift = (recent - previous) / float(period)
        pred = pred + (horizon_offsets[None, :, None] + 1.0) * drift[:, None, :]
    return pred


def tune_exponential_smoothing(info):
    val_starts = window_starts(info, "val", batch_size=BATCH_SIZE, drop_last=True)
    test_starts = window_starts(info, "test", batch_size=BATCH_SIZE, drop_last=True)
    val_true = numpy_targets(info, "val", batch_size=BATCH_SIZE, drop_last=True)
    test_true = numpy_targets(info, "test", batch_size=BATCH_SIZE, drop_last=True)
    data = info["data"]

    candidates = []
    candidates.append(("last", {}, repeat_last_prediction(info, val_starts)))

    for alpha in np.arange(0.1, 1.0, 0.1):
        smoothed = exponential_smooth(data, float(alpha))
        candidates.append(
            (
                "simple",
                {"alpha": float(alpha), "trend": None, "seasonal": None},
                predict_from_smoothed(smoothed, val_starts),
            )
        )

    for alpha in [0.2, 0.5, 0.8]:
        for beta in [0.05, 0.1, 0.2]:
            for damping in [1.0, 0.9, 0.8]:
                levels, trends = holt_smooth(data, alpha, beta, damping)
                candidates.append(
                    (
                        "holt",
                        {"alpha": alpha, "beta": beta, "damping": damping, "trend": "additive", "seasonal": None},
                        holt_predict(levels, trends, val_starts, damping),
                    )
                )

    for period in [24, 48, 72, 96]:
        candidates.append(
            (
                "seasonal_naive",
                {"period": period, "trend": None, "seasonal": "lag"},
                seasonal_naive_prediction(info, val_starts, period, trend=False),
            )
        )
        candidates.append(
            (
                "seasonal_naive_trend",
                {"period": period, "trend": "drift", "seasonal": "lag"},
                seasonal_naive_prediction(info, val_starts, period, trend=True),
            )
        )

    best = min(candidates, key=lambda item: validation_mse(item[2], val_true))
    name, params, _ = best
    if name == "last":
        test_pred = repeat_last_prediction(info, test_starts)
    elif name == "simple":
        test_pred = predict_from_smoothed(exponential_smooth(data, params["alpha"]), test_starts)
    elif name == "holt":
        levels, trends = holt_smooth(data, params["alpha"], params["beta"], params["damping"])
        test_pred = holt_predict(levels, trends, test_starts, params["damping"])
    else:
        test_pred = seasonal_naive_prediction(info, test_starts, params["period"], trend=params["trend"] == "drift")

    return {
        "model": "Exponential Smoothing",
        "best_params": {"method": name, **params},
        "best_val": validation_mse(best[2], val_true),
        "metrics": calculate_metrics(test_pred, test_true),
    }


def tune_linear(info, device):
    candidates = [
        {"epochs": 20, "learning_rate": 0.001, "weight_decay": 0.0, "batch_size": 512},
        {"epochs": 30, "learning_rate": 0.003, "weight_decay": 0.0, "batch_size": 512},
        {"epochs": 30, "learning_rate": 0.01, "weight_decay": 0.0, "batch_size": 512},
    ]
    return tune_torch_candidates("Linear", info, device, candidates, lambda _config: LinearForecast())


def tune_gru(info, device):
    candidates = [
        {"hidden_size": 32, "epochs": 10, "learning_rate": 0.003, "weight_decay": 0.0, "batch_size": 512},
        {"hidden_size": 64, "epochs": 12, "learning_rate": 0.003, "weight_decay": 0.0, "batch_size": 512},
        {"hidden_size": 128, "epochs": 10, "learning_rate": 0.001, "weight_decay": 0.0, "batch_size": 512},
    ]
    input_size = len(info["columns"])
    return tune_torch_candidates(
        "GRU",
        info,
        device,
        candidates,
        lambda config: GRUForecast(input_size=input_size, hidden_size=config["hidden_size"]),
    )


def tune_patchtst(info, device):
    candidates = [
        {
            "d_model": 32,
            "n_heads": 4,
            "e_layers": 1,
            "d_ff": 64,
            "dropout": 0.2,
            "fc_dropout": 0.2,
            "head_dropout": 0.0,
            "patch_len": 16,
            "stride": 8,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "epochs": 4,
            "batch_size": 512,
        },
        {
            "d_model": 64,
            "n_heads": 4,
            "e_layers": 1,
            "d_ff": 128,
            "dropout": 0.1,
            "fc_dropout": 0.1,
            "head_dropout": 0.0,
            "patch_len": 16,
            "stride": 8,
            "learning_rate": 0.0003,
            "weight_decay": 0.0,
            "epochs": 4,
            "batch_size": 512,
        },
    ]
    input_size = len(info["columns"])
    return tune_torch_candidates(
        "PatchTST",
        info,
        device,
        candidates,
        lambda config: PatchTST.Model(build_patchtst_config(input_size, config)).float(),
    )


def tune_torch_candidates(model_name, info, device, candidates, model_factory):
    best = None
    records = []

    for index, config in enumerate(candidates, start=1):
        print(f"{model_name} candidate {index} starting: {config}", flush=True)
        set_seed()
        train_loader, val_loader, test_loader = make_torch_loaders(info, batch_size=config["batch_size"])
        model = model_factory(config).to(device)
        result = train_torch_model(
            model,
            train_loader,
            val_loader,
            device,
            epochs=config["epochs"],
            learning_rate=config["learning_rate"],
            weight_decay=config["weight_decay"],
            patience=4,
        )
        record = {
            "candidate": index,
            "params": config,
            "best_val": result["best_val"],
            "best_epoch": result["best_epoch"],
            "last_train_mse": result["history"][-1]["train_mse"],
            "last_val_mse": result["history"][-1]["val_mse"],
        }
        records.append(record)
        print(f"{model_name} candidate {index}: val_mse={result['best_val']:.6f}, best_epoch={result['best_epoch']}", flush=True)
        if best is None or result["best_val"] < best["best_val"]:
            pred, true = collect_torch_predictions(result["model"], test_loader, device)
            best = {
                "model": model_name,
                "best_params": {**config, "best_epoch": result["best_epoch"]},
                "best_val": result["best_val"],
                "metrics": calculate_metrics(pred, true),
                "records": records,
            }

    best["records"] = records
    return best


def write_outputs(results):
    for result in results:
        filename = {
            "Linear": "results_linear.txt",
            "Exponential Smoothing": "results_exponential_smoothing.txt",
            "GRU": "results_gru.txt",
            "PatchTST": "results_patchtst.txt",
        }[result["model"]]
        (BASE_DIR / filename).write_text(format_metrics(result["metrics"]) + "\n")

    labels = ["Model", "MSE", "MAE", "RMSE", "RSE", "MAPE", "MSPE", "Corr mean"]
    rows = [labels]
    for result in results:
        metrics = result["metrics"]
        rows.append(
            [
                result["model"],
                f"{metrics['MSE']:.6f}",
                f"{metrics['MAE']:.6f}",
                f"{metrics['RMSE']:.6f}",
                f"{metrics['RSE']:.6f}",
                f"{metrics['MAPE']:.6f}",
                f"{metrics['MSPE']:.6f}",
                f"{metrics['Corr mean']:.6f}",
            ]
        )

    widths = [max(len(row[column]) for row in rows) for column in range(len(labels))]
    lines = []
    for row_index, row in enumerate(rows):
        lines.append(" | ".join(value.ljust(widths[index]) for index, value in enumerate(row)))
        if row_index == 0:
            lines.append("-+-".join("-" * width for width in widths))

    comparison = "\n".join(lines)
    print("\nFinal comparison:")
    print(comparison)
    (BASE_DIR / "results_final_comparison.txt").write_text(comparison + "\n")

    summary = [
        {
            "model": result["model"],
            "best_validation_mse": result["best_val"],
            "best_params": result["best_params"],
            "candidate_records": result.get("records", []),
        }
        for result in results
    ]
    (BASE_DIR / "results_best_hyperparameters.json").write_text(json.dumps(summary, indent=2) + "\n")


def main():
    set_seed()
    info = load_patchtst_weather()
    device = torch.device("cpu")

    results = [
        tune_linear(info, device),
        tune_exponential_smoothing(info),
        tune_gru(info, device),
        tune_patchtst(info, device),
    ]
    write_outputs(results)


if __name__ == "__main__":
    main()
