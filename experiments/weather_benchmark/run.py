import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from .data import inverse_target, load_weather_csv, make_loaders, targets, window_starts
from .metrics import calculate_metrics
from .modeling import exponential_smoothing_predict, make_torch_model
from .training import collect_predictions, set_seed, train_torch_model


DEFAULT_MODELS = ("linear", "exponential_smoothing", "gru", "patchtst", "timesnet", "itransformer")
METRIC_NAMES = ["MSE", "MAE", "RMSE", "RSE", "MAPE", "MSPE", "Corr mean"]


def _write_outputs(output_dir: Path, results: list[dict], log_lines: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "best_val",
        *METRIC_NAMES,
        "trainable_parameters",
        "training_time_sec",
        "inference_time_sec",
    ]
    with (output_dir / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = {"model": result["model"], "best_val": result["best_val"]}
            row.update(result["metrics"])
            row["trainable_parameters"] = result["trainable_parameters"]
            row["training_time_sec"] = result["training_time_sec"]
            row["inference_time_sec"] = result["inference_time_sec"]
            writer.writerow(row)
    with (output_dir / "six_model_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = {"model": result["model"], "best_val": result["best_val"]}
            row.update(result["metrics"])
            row["trainable_parameters"] = result["trainable_parameters"]
            row["training_time_sec"] = result["training_time_sec"]
            row["inference_time_sec"] = result["inference_time_sec"]
            writer.writerow(row)
    (output_dir / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    (output_dir / "run.log").write_text("\n".join(log_lines) + "\n")


def _target_metrics(info, pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    return calculate_metrics(inverse_target(info, pred), inverse_target(info, true))


def _trainable_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _evaluate_exponential_smoothing(info, seq_len: int, pred_len: int, batch_size: int):
    start_time = time.perf_counter()
    val_starts = window_starts(info, "val", seq_len, pred_len, batch_size, drop_last=True)
    test_starts = window_starts(info, "test", seq_len, pred_len, batch_size, drop_last=True)
    val_true = targets(info, "val", seq_len, pred_len, batch_size, drop_last=True)
    test_true = targets(info, "test", seq_len, pred_len, batch_size, drop_last=True)
    candidates = []
    for alpha in np.arange(0.1, 1.0, 0.1):
        val_pred = exponential_smoothing_predict(info.data, val_starts, seq_len, pred_len, float(alpha))
        val_mse = float(np.mean((inverse_target(info, val_pred) - inverse_target(info, val_true)) ** 2))
        candidates.append((float(alpha), val_mse))
    alpha, best_val = min(candidates, key=lambda item: item[1])
    training_time = time.perf_counter() - start_time
    inference_start = time.perf_counter()
    test_pred = exponential_smoothing_predict(info.data, test_starts, seq_len, pred_len, alpha)
    inference_time = time.perf_counter() - inference_start
    return {
        "best_val": best_val,
        "metrics": _target_metrics(info, test_pred, test_true),
        "params": {"alpha": alpha},
        "trainable_parameters": 0,
        "training_time_sec": training_time,
        "inference_time_sec": inference_time,
    }


def run_benchmark(
    data_path,
    output_dir,
    models=DEFAULT_MODELS,
    seq_len: int = 96,
    pred_len: int = 24,
    batch_size: int = 512,
    epochs: int = 5,
    target: str = "OT",
    device_name: str = "cpu",
    seed: int = 2021,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    patience: int | None = None,
    gru_hidden_size: int = 64,
    patchtst_d_model: int = 32,
    patchtst_heads: int = 4,
    patchtst_layers: int = 1,
):
    set_seed(seed)
    output_path = Path(output_dir)
    log_lines = [
        f"data_path={data_path}",
        f"models={','.join(models)}",
        f"target={target}",
        f"seq_len={seq_len}",
        f"pred_len={pred_len}",
        f"batch_size={batch_size}",
        f"epochs={epochs}",
        f"learning_rate={learning_rate}",
        f"weight_decay={weight_decay}",
        f"patience={patience}",
        f"seed={seed}",
    ]
    info = load_weather_csv(data_path, seq_len=seq_len, target=target)
    train_loader, val_loader, test_loader = make_loaders(info, seq_len=seq_len, pred_len=pred_len, batch_size=batch_size, seed=seed)
    device = torch.device(device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu")

    results = []
    for model_name in models:
        print(f"starting {model_name}", flush=True)
        log_lines.append(f"starting {model_name}")
        if model_name == "exponential_smoothing":
            evaluated = _evaluate_exponential_smoothing(info, seq_len, pred_len, batch_size)
            results.append({"model": model_name, **evaluated})
        else:
            model = make_torch_model(
                model_name,
                input_size=len(info.columns),
                seq_len=seq_len,
                pred_len=pred_len,
                gru_hidden_size=gru_hidden_size,
                patchtst_d_model=patchtst_d_model,
                patchtst_heads=patchtst_heads,
                patchtst_layers=patchtst_layers,
            )
            trainable_parameters = _trainable_parameters(model)
            trained = train_torch_model(
                model,
                train_loader,
                val_loader,
                device,
                epochs=epochs,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                patience=patience,
                val_score_fn=lambda pred, true: np.mean((inverse_target(info, pred) - inverse_target(info, true)) ** 2),
                log_prefix=model_name,
            )
            inference_start = time.perf_counter()
            pred, true = collect_predictions(trained["model"], test_loader, device)
            inference_time = time.perf_counter() - inference_start
            results.append(
                {
                    "model": model_name,
                    "best_val": trained["best_val"],
                    "metrics": _target_metrics(info, pred, true),
                    "trainable_parameters": trainable_parameters,
                    "training_time_sec": trained["training_time_sec"],
                    "inference_time_sec": inference_time,
                    "params": {
                        "epochs": epochs,
                        "best_epoch": trained["best_epoch"],
                        "learning_rate": learning_rate,
                        "weight_decay": weight_decay,
                        "patience": patience,
                        "gru_hidden_size": gru_hidden_size,
                        "patchtst_d_model": patchtst_d_model,
                        "patchtst_heads": patchtst_heads,
                        "patchtst_layers": patchtst_layers,
                    },
                    "history": trained["history"],
                }
            )
        print(f"finished {model_name}", flush=True)
        log_lines.append(f"finished {model_name}")

    _write_outputs(output_path, results, log_lines)
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Run fixed-epoch weather benchmark models.")
    parser.add_argument("--data-path", type=Path, default=Path("PatchTST_supervised/dataset/weather.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/weather_benchmark/results"))
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS), choices=list(DEFAULT_MODELS))
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--pred-len", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--gru-hidden-size", type=int, default=64)
    parser.add_argument("--patchtst-d-model", type=int, default=32)
    parser.add_argument("--patchtst-heads", type=int, default=4)
    parser.add_argument("--patchtst-layers", type=int, default=1)
    parser.add_argument("--target", default="OT")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    run_benchmark(
        data_path=args.data_path,
        output_dir=args.output_dir,
        models=tuple(args.models),
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        gru_hidden_size=args.gru_hidden_size,
        patchtst_d_model=args.patchtst_d_model,
        patchtst_heads=args.patchtst_heads,
        patchtst_layers=args.patchtst_layers,
        target=args.target,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
