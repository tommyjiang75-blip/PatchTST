from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn


class LinearForecast(nn.Module):
    def __init__(self, seq_len: int, pred_len: int):
        super().__init__()
        self.linear = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        return self.linear(x.permute(0, 2, 1)).permute(0, 2, 1)


class GRUForecast(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, pred_len: int):
        super().__init__()
        self.input_size = input_size
        self.pred_len = pred_len
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.projection = nn.Linear(hidden_size, pred_len * input_size)

    def forward(self, x):
        _, hidden = self.gru(x)
        output = self.projection(hidden[-1])
        return output.view(x.size(0), self.pred_len, self.input_size)


def exponential_smooth(data: np.ndarray, alpha: float) -> np.ndarray:
    smoothed = np.empty_like(data)
    smoothed[0] = data[0]
    for index in range(1, len(data)):
        smoothed[index] = alpha * data[index] + (1.0 - alpha) * smoothed[index - 1]
    return smoothed


def exponential_smoothing_predict(data: np.ndarray, starts: np.ndarray, seq_len: int, pred_len: int, alpha: float) -> np.ndarray:
    smoothed = exponential_smooth(data, alpha)
    last_smoothed = smoothed[starts + seq_len - 1]
    return np.repeat(last_smoothed[:, None, :], pred_len, axis=1)


def make_patchtst(input_size: int, seq_len: int, pred_len: int, d_model: int, n_heads: int, e_layers: int):
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    supervised_root = repo_root / "PatchTST_supervised"
    if str(supervised_root) not in sys.path:
        sys.path.insert(0, str(supervised_root))
    from models import PatchTST

    config = SimpleNamespace(
        seq_len=seq_len,
        pred_len=pred_len,
        enc_in=input_size,
        e_layers=e_layers,
        n_heads=n_heads,
        d_model=d_model,
        d_ff=d_model * 2,
        dropout=0.1,
        fc_dropout=0.1,
        head_dropout=0.0,
        patch_len=min(16, seq_len),
        stride=max(1, min(8, seq_len // 2)),
        padding_patch="end",
        revin=1,
        affine=0,
        subtract_last=0,
        decomposition=0,
        kernel_size=25,
        individual=0,
    )
    return PatchTST.Model(config).float()


def make_torch_model(name: str, input_size: int, seq_len: int, pred_len: int, gru_hidden_size: int, patchtst_d_model: int, patchtst_heads: int, patchtst_layers: int):
    if name == "linear":
        return LinearForecast(seq_len=seq_len, pred_len=pred_len)
    if name == "gru":
        return GRUForecast(input_size=input_size, hidden_size=gru_hidden_size, pred_len=pred_len)
    if name == "patchtst":
        return make_patchtst(input_size=input_size, seq_len=seq_len, pred_len=pred_len, d_model=patchtst_d_model, n_heads=patchtst_heads, e_layers=patchtst_layers)
    raise ValueError(f"Unknown torch model: {name}")

