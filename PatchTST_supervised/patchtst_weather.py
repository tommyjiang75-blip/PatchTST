from types import SimpleNamespace

import torch

from models import PatchTST
from weather_experiment_common import (
    BATCH_SIZE,
    PRED_LEN,
    SEQ_LEN,
    calculate_metrics,
    collect_torch_predictions,
    load_patchtst_weather,
    make_torch_loaders,
    save_and_print_metrics,
    set_seed,
    train_torch_model,
)


BEST_CONFIG = {
    "d_model": 32,
    "n_heads": 4,
    "e_layers": 1,
    "d_ff": 64,
    "dropout": 0.2,
    "fc_dropout": 0.2,
    "head_dropout": 0.0,
    "patch_len": 16,
    "stride": 8,
    "learning_rate": 0.0001,
    "weight_decay": 0.0,
    "epochs": 1,
    "batch_size": BATCH_SIZE,
}


def build_config(input_size, config=BEST_CONFIG):
    return SimpleNamespace(
        seq_len=SEQ_LEN,
        pred_len=PRED_LEN,
        enc_in=input_size,
        e_layers=config["e_layers"],
        n_heads=config["n_heads"],
        d_model=config["d_model"],
        d_ff=config["d_ff"],
        dropout=config["dropout"],
        fc_dropout=config["fc_dropout"],
        head_dropout=config["head_dropout"],
        patch_len=config["patch_len"],
        stride=config["stride"],
        padding_patch="end",
        revin=1,
        affine=0,
        subtract_last=0,
        decomposition=0,
        kernel_size=25,
        individual=0,
    )


def train_model(model, train_loader, val_loader, device, config=BEST_CONFIG):
    result = train_torch_model(
        model,
        train_loader,
        val_loader,
        device,
        epochs=config["epochs"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        patience=5,
    )
    return result["model"]


def main():
    set_seed()
    info = load_patchtst_weather()
    train_loader, val_loader, test_loader = make_torch_loaders(info, batch_size=BEST_CONFIG["batch_size"])
    device = torch.device("cpu")

    model = PatchTST.Model(build_config(len(info["columns"]))).float().to(device)
    model = train_model(model, train_loader, val_loader, device)

    pred, true = collect_torch_predictions(model, test_loader, device)
    metrics = calculate_metrics(pred, true)
    save_and_print_metrics(metrics, "results_patchtst.txt")


if __name__ == "__main__":
    main()
