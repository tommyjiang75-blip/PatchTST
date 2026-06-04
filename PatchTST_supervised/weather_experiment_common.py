from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset" / "weather.csv"

SEQ_LEN = 96
LABEL_LEN = 48
PRED_LEN = 24
BATCH_SIZE = 512
TARGET = "OT"
RANDOM_SEED = 2021


def set_seed(seed=RANDOM_SEED):
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.set_num_threads(min(4, max(1, torch.get_num_threads())))
    except ImportError:
        pass


def load_patchtst_weather(dataset_path=DATASET_PATH, seq_len=SEQ_LEN, target=TARGET):
    df_raw = pd.read_csv(dataset_path)

    cols = list(df_raw.columns)
    cols.remove(target)
    cols.remove("date")
    df_raw = df_raw[["date"] + cols + [target]]

    num_train = int(len(df_raw) * 0.7)
    num_test = int(len(df_raw) * 0.2)
    num_val = len(df_raw) - num_train - num_test

    border1s = {
        "train": 0,
        "val": num_train - seq_len,
        "test": len(df_raw) - num_test - seq_len,
    }
    border2s = {
        "train": num_train,
        "val": num_train + num_val,
        "test": len(df_raw),
    }

    values = df_raw[df_raw.columns[1:]].values.astype(np.float32)
    scaler = StandardScaler()
    scaler.fit(values[border1s["train"] : border2s["train"]])
    scaled = scaler.transform(values).astype(np.float32)

    return {
        "data": scaled,
        "columns": list(df_raw.columns[1:]),
        "borders": {key: (border1s[key], border2s[key]) for key in border1s},
        "counts": {"train": num_train, "val": num_val, "test": num_test},
        "scaler": scaler,
    }


def window_starts(info, flag, seq_len=SEQ_LEN, pred_len=PRED_LEN, batch_size=BATCH_SIZE, drop_last=True):
    border1, border2 = info["borders"][flag]
    starts = np.arange(border1, border2 - seq_len - pred_len + 1)
    if drop_last:
        usable = (len(starts) // batch_size) * batch_size
        starts = starts[:usable]
    return starts


def numpy_targets(info, flag, seq_len=SEQ_LEN, pred_len=PRED_LEN, batch_size=BATCH_SIZE, drop_last=True):
    data = info["data"]
    starts = window_starts(info, flag, seq_len, pred_len, batch_size, drop_last)
    return np.stack([data[start + seq_len : start + seq_len + pred_len] for start in starts])


class _WeatherWindowDataset:
    def __init__(self, data, starts, seq_len=SEQ_LEN, pred_len=PRED_LEN):
        self.data = data
        self.starts = starts
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        import torch

        start = self.starts[index]
        x = self.data[start : start + self.seq_len]
        y = self.data[start + self.seq_len : start + self.seq_len + self.pred_len]
        return torch.from_numpy(x), torch.from_numpy(y)


def make_torch_loaders(info, batch_size=BATCH_SIZE, seq_len=SEQ_LEN, pred_len=PRED_LEN):
    import torch
    from torch.utils.data import DataLoader

    generator = torch.Generator()
    generator.manual_seed(RANDOM_SEED)

    data = info["data"]
    train_starts = window_starts(info, "train", seq_len, pred_len, batch_size, drop_last=True)
    val_starts = window_starts(info, "val", seq_len, pred_len, batch_size, drop_last=True)
    test_starts = window_starts(info, "test", seq_len, pred_len, batch_size, drop_last=True)

    train_loader = DataLoader(
        _WeatherWindowDataset(data, train_starts, seq_len, pred_len),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        generator=generator,
    )
    val_loader = DataLoader(
        _WeatherWindowDataset(data, val_starts, seq_len, pred_len),
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
    )
    test_loader = DataLoader(
        _WeatherWindowDataset(data, test_starts, seq_len, pred_len),
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
    )
    return train_loader, val_loader, test_loader


def collect_torch_predictions(model, loader, device):
    import torch

    preds = []
    trues = []
    model.eval()
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.float().to(device)
            outputs = model(batch_x)
            preds.append(outputs.detach().cpu().numpy())
            trues.append(batch_y.numpy())
    return np.concatenate(preds, axis=0), np.concatenate(trues, axis=0)


def train_torch_model(
    model,
    train_loader,
    val_loader,
    device,
    epochs,
    learning_rate,
    weight_decay=0.0,
    patience=5,
):
    import copy
    import torch
    import torch.nn as nn

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    best_epoch = 0
    history = []
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)

            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        pred, true = collect_torch_predictions(model, val_loader, device)
        val_loss = float(np.mean((pred - true) ** 2))
        train_loss = float(np.mean(train_losses))
        history.append({"epoch": epoch, "train_mse": train_loss, "val_mse": val_loss})

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    model.load_state_dict(best_state)
    return {"model": model, "best_val": best_val, "best_epoch": best_epoch, "history": history}


def rse(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def corr(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    d += 1e-12
    return 0.01 * (u / d).mean(-1)


def calculate_metrics(pred, true):
    with np.errstate(divide="ignore", invalid="ignore"):
        mape = np.mean(np.abs((pred - true) / true))
        mspe = np.mean(np.square((pred - true) / true))

    mse = np.mean((pred - true) ** 2)
    mae = np.mean(np.abs(pred - true))
    rmse = np.sqrt(mse)
    return {
        "MSE": float(mse),
        "MAE": float(mae),
        "RMSE": float(rmse),
        "RSE": float(rse(pred, true)),
        "MAPE": float(mape),
        "MSPE": float(mspe),
        "Corr mean": float(np.mean(corr(pred, true))),
    }


def _format_value(value):
    if np.isfinite(value):
        return f"{value:.6f}"
    return str(value)


def format_metrics(metrics):
    labels = ["MSE", "MAE", "RMSE", "RSE", "MAPE", "MSPE", "Corr mean"]
    return "\n".join(f"{label}: {_format_value(metrics[label])}" for label in labels)


def save_and_print_metrics(metrics, filename):
    output = format_metrics(metrics)
    print(output)
    (BASE_DIR / filename).write_text(output + "\n")
