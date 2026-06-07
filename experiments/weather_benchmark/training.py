import copy
import time

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(4, max(1, torch.get_num_threads())))


def collect_predictions(model: nn.Module, loader, device: torch.device):
    preds = []
    trues = []
    model.eval()
    with torch.no_grad():
        for batch_x, batch_y in loader:
            output = model(batch_x.float().to(device))
            preds.append(output.detach().cpu().numpy())
            trues.append(batch_y.numpy())
    return np.concatenate(preds, axis=0), np.concatenate(trues, axis=0)


def train_torch_model(
    model: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float = 0.0,
    patience: int | None = None,
    val_score_fn=None,
    log_prefix: str | None = None,
):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    best_epoch = 0
    history = []
    epochs_without_improvement = 0

    model.to(device)
    start_time = time.perf_counter()
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
            train_losses.append(float(loss.item()))

        pred, true = collect_predictions(model, val_loader, device)
        val_mse = float(np.mean((pred - true) ** 2))
        val_score = float(val_score_fn(pred, true)) if val_score_fn is not None else val_mse
        train_mse = float(np.mean(train_losses))
        history.append({"epoch": epoch, "train_mse": train_mse, "val_mse": val_mse, "val_score": val_score})
        if log_prefix is not None:
            print(
                f"{log_prefix} epoch {epoch}/{epochs} train_mse={train_mse:.6f} val_score={val_score:.6f}",
                flush=True,
            )
        if val_score < best_val:
            best_val = val_score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if patience is not None and epochs_without_improvement >= patience:
                break

    model.load_state_dict(best_state)
    return {
        "model": model,
        "best_val": best_val,
        "best_epoch": best_epoch,
        "history": history,
        "training_time_sec": time.perf_counter() - start_time,
    }
