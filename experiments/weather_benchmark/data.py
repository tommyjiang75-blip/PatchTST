from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class WeatherData:
    data: np.ndarray
    columns: list[str]
    borders: dict[str, tuple[int, int]]
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    target: str

    @property
    def target_index(self) -> int:
        return self.columns.index(self.target)


class WeatherWindowDataset(Dataset):
    def __init__(self, data: np.ndarray, starts: np.ndarray, seq_len: int, pred_len: int):
        self.data = data
        self.starts = starts
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int):
        start = self.starts[index]
        x = self.data[start : start + self.seq_len]
        y = self.data[start + self.seq_len : start + self.seq_len + self.pred_len]
        return torch.from_numpy(x), torch.from_numpy(y)


def load_weather_csv(data_path: str | Path, seq_len: int, target: str = "OT") -> WeatherData:
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Weather CSV not found: {path}. Pass --data-path or place weather.csv under PatchTST_supervised/dataset/."
        )

    frame = pd.read_csv(path)
    if "date" not in frame.columns:
        raise ValueError(f"{path} must contain a 'date' column.")
    if target not in frame.columns:
        raise ValueError(f"{path} must contain target column '{target}'.")

    columns = [column for column in frame.columns if column not in {"date", target}] + [target]
    values = frame[columns].to_numpy(dtype=np.float32)

    num_train = int(len(values) * 0.7)
    num_test = int(len(values) * 0.2)
    num_val = len(values) - num_train - num_test
    borders = {
        "train": (0, num_train),
        "val": (num_train - seq_len, num_train + num_val),
        "test": (len(values) - num_test - seq_len, len(values)),
    }

    train_values = values[borders["train"][0] : borders["train"][1]]
    mean = train_values.mean(axis=0)
    scale = train_values.std(axis=0)
    scale[scale == 0.0] = 1.0
    scaled = ((values - mean) / scale).astype(np.float32)

    return WeatherData(data=scaled, columns=columns, borders=borders, scaler_mean=mean, scaler_scale=scale, target=target)


def inverse_transform(info: WeatherData, values: np.ndarray) -> np.ndarray:
    return values * info.scaler_scale + info.scaler_mean


def inverse_target(info: WeatherData, values: np.ndarray) -> np.ndarray:
    target_index = info.target_index
    return values[..., target_index] * info.scaler_scale[target_index] + info.scaler_mean[target_index]


def window_starts(info: WeatherData, flag: str, seq_len: int, pred_len: int, batch_size: int, drop_last: bool = True) -> np.ndarray:
    border1, border2 = info.borders[flag]
    starts = np.arange(border1, border2 - seq_len - pred_len + 1)
    if drop_last:
        usable = (len(starts) // batch_size) * batch_size
        starts = starts[:usable]
    return starts


def targets(info: WeatherData, flag: str, seq_len: int, pred_len: int, batch_size: int, drop_last: bool = True) -> np.ndarray:
    starts = window_starts(info, flag, seq_len, pred_len, batch_size, drop_last)
    return np.stack([info.data[start + seq_len : start + seq_len + pred_len] for start in starts])


def make_loaders(info: WeatherData, seq_len: int, pred_len: int, batch_size: int, seed: int):
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_starts = window_starts(info, "train", seq_len, pred_len, batch_size, drop_last=True)
    val_starts = window_starts(info, "val", seq_len, pred_len, batch_size, drop_last=True)
    test_starts = window_starts(info, "test", seq_len, pred_len, batch_size, drop_last=True)
    if min(len(train_starts), len(val_starts), len(test_starts)) == 0:
        raise ValueError("Dataset is too small for the requested seq_len, pred_len, and batch_size.")

    return (
        DataLoader(WeatherWindowDataset(info.data, train_starts, seq_len, pred_len), batch_size=batch_size, shuffle=True, drop_last=True, generator=generator),
        DataLoader(WeatherWindowDataset(info.data, val_starts, seq_len, pred_len), batch_size=batch_size, shuffle=False, drop_last=True),
        DataLoader(WeatherWindowDataset(info.data, test_starts, seq_len, pred_len), batch_size=batch_size, shuffle=False, drop_last=True),
    )
