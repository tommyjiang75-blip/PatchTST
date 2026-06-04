# Weather Benchmark Reorganization Design

Date: 2026-06-04

## Goal

Reorganize the student-added weather experiments into a reproducible benchmark without disturbing the original PatchTST implementation. The benchmark should train Linear, Exponential Smoothing, GRU, and PatchTST with fixed settings and report comparable test metrics.

## Scope

- Keep official PatchTST source code in place.
- Add a focused experiment package under `experiments/weather_benchmark/`.
- Provide one CLI for data preparation, training, evaluation, and result writing.
- Use shared split/window/scaling/metric code across all four models.
- Fixed default settings: `seq_len=96`, `pred_len=24`, `epochs=5`, `batch_size=512`, CPU by default unless `--device` is provided.

## Data

The CLI accepts `--data-path`. If no real data is available, tests use synthetic data only. The training script should fail with a clear message when real data is missing instead of a raw stack trace.

## Outputs

Write per-run artifacts under `experiments/weather_benchmark/results/`: metrics CSV, JSON summary, and log file. Include model name, validation MSE, and test metrics.

## Testing

Add smoke tests that generate synthetic weather-like data, run all model forward/training paths with tiny epochs, and verify metrics/results files are produced.
