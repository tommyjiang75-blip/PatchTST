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
    "hidden_size": 64,
    "epochs": 3,
    "learning_rate": 0.001,
    "weight_decay": 0.0,
    "batch_size": BATCH_SIZE,
}


class GRUForecast(nn.Module):
    def __init__(self, input_size, hidden_size=64, seq_len=SEQ_LEN, pred_len=PRED_LEN):
        super().__init__()
        self.pred_len = pred_len
        self.input_size = input_size
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.projection = nn.Linear(hidden_size, pred_len * input_size)

    def forward(self, x):
        _, hidden = self.gru(x)
        output = self.projection(hidden[-1])
        return output.view(x.size(0), self.pred_len, self.input_size)


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

    model = GRUForecast(input_size=len(info["columns"]), hidden_size=BEST_CONFIG["hidden_size"]).to(device)
    model = train_model(model, train_loader, val_loader, device)

    pred, true = collect_torch_predictions(model, test_loader, device)
    metrics = calculate_metrics(pred, true)
    save_and_print_metrics(metrics, "results_gru.txt")


if __name__ == "__main__":
    main()
