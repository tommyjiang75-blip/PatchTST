import torch
import torch.nn as nn

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
    "epochs": 10,
    "learning_rate": 0.001,
    "weight_decay": 0.0,
    "batch_size": BATCH_SIZE,
}


class LinearForecast(nn.Module):
    def __init__(self, seq_len=SEQ_LEN, pred_len=PRED_LEN):
        super().__init__()
        self.linear = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        return self.linear(x.permute(0, 2, 1)).permute(0, 2, 1)


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

    model = LinearForecast().to(device)
    model = train_model(model, train_loader, val_loader, device)

    pred, true = collect_torch_predictions(model, test_loader, device)
    metrics = calculate_metrics(pred, true)
    save_and_print_metrics(metrics, "results_linear.txt")


if __name__ == "__main__":
    main()
