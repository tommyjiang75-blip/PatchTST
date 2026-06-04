# Weather Benchmark

This folder contains the reorganized student weather experiments. It runs four comparable models on one shared data pipeline:

- `linear`
- `exponential_smoothing`
- `gru`
- `patchtst`

Default settings match the cleaned benchmark setup:

- `seq_len=96`
- `pred_len=24`
- `batch_size=512`
- `epochs=5`
- `target=OT`
- CPU execution unless `--device cuda` is passed

Run from the repository root:

```bash
python -m experiments.weather_benchmark.run \
  --data-path PatchTST_supervised/dataset/weather.csv \
  --epochs 5 \
  --models linear exponential_smoothing gru patchtst
```

Outputs are written to `experiments/weather_benchmark/results/`:

- `metrics.csv`
- `summary.json`
- `run.log`

