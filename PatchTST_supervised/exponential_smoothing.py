import numpy as np

from weather_experiment_common import (
    BATCH_SIZE,
    PRED_LEN,
    SEQ_LEN,
    calculate_metrics,
    load_patchtst_weather,
    numpy_targets,
    save_and_print_metrics,
    set_seed,
    window_starts,
)


def exponential_smooth(data, alpha):
    smoothed = np.empty_like(data)
    smoothed[0] = data[0]
    for index in range(1, len(data)):
        smoothed[index] = alpha * data[index] + (1.0 - alpha) * smoothed[index - 1]
    return smoothed


def predict_from_smoothed(smoothed, starts, seq_len=SEQ_LEN, pred_len=PRED_LEN):
    last_smoothed = smoothed[starts + seq_len - 1]
    return np.repeat(last_smoothed[:, None, :], pred_len, axis=1)


def tune_alpha(info):
    data = info["data"]
    val_starts = window_starts(info, "val", batch_size=BATCH_SIZE, drop_last=True)
    val_true = numpy_targets(info, "val", batch_size=BATCH_SIZE, drop_last=True)

    best_alpha = 0.5
    best_mse = float("inf")
    for alpha in np.arange(0.1, 1.0, 0.1):
        smoothed = exponential_smooth(data, float(alpha))
        val_pred = predict_from_smoothed(smoothed, val_starts)
        mse = np.mean((val_pred - val_true) ** 2)
        if mse < best_mse:
            best_mse = mse
            best_alpha = float(alpha)
    return best_alpha


def main():
    set_seed()
    info = load_patchtst_weather()
    alpha = tune_alpha(info)

    smoothed = exponential_smooth(info["data"], alpha)
    test_starts = window_starts(info, "test", batch_size=BATCH_SIZE, drop_last=True)
    pred = predict_from_smoothed(smoothed, test_starts)
    true = numpy_targets(info, "test", batch_size=BATCH_SIZE, drop_last=True)

    metrics = calculate_metrics(pred, true)
    save_and_print_metrics(metrics, "results_exponential_smoothing.txt")


if __name__ == "__main__":
    main()
