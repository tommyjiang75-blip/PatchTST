# Weather Benchmark Reorganization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible weather benchmark for Linear, Exponential Smoothing, GRU, and PatchTST with fixed epochs and comparable metrics.

**Architecture:** Keep the original PatchTST code untouched and add a small experiment package under `experiments/weather_benchmark`. The package owns data loading, model factories, training, metrics, and the CLI runner; tests use synthetic CSV data so they do not depend on downloaded datasets.

**Tech Stack:** Python, NumPy, pandas, PyTorch, stdlib `unittest`.

---

### Task 1: Add Smoke Test

**Files:**
- Create: `tests/test_weather_benchmark_smoke.py`

- [ ] **Step 1: Write failing smoke test**

Create a unittest that writes a synthetic `weather.csv`, runs all four benchmark models with tiny settings, and asserts result files are produced.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_weather_benchmark_smoke -v`
Expected: FAIL because `experiments.weather_benchmark.run` does not exist.

### Task 2: Add Benchmark Package

**Files:**
- Create: `experiments/weather_benchmark/__init__.py`
- Create: `experiments/weather_benchmark/data.py`
- Create: `experiments/weather_benchmark/metrics.py`
- Create: `experiments/weather_benchmark/modeling.py`
- Create: `experiments/weather_benchmark/training.py`
- Create: `experiments/weather_benchmark/run.py`

- [ ] **Step 1: Implement data loading**

Add CSV loading, train/val/test splits, simple train-fitted standardization, and torch dataloaders.

- [ ] **Step 2: Implement metrics**

Add MSE, MAE, RMSE, RSE, MAPE, MSPE, and correlation mean with NaN-safe division.

- [ ] **Step 3: Implement models**

Add Linear and GRU modules, exponential smoothing prediction, and a PatchTST factory that imports the original implementation.

- [ ] **Step 4: Implement runner**

Add CLI and callable `run_benchmark` that trains/evaluates requested models and writes `metrics.csv`, `summary.json`, and `run.log`.

### Task 3: Verify and Train

**Files:**
- Modify only if necessary: `README.md`

- [ ] **Step 1: Run smoke test**

Run: `python -m unittest tests.test_weather_benchmark_smoke -v`
Expected: PASS.

- [ ] **Step 2: Run compile check**

Run: `python -m compileall -q experiments tests PatchTST_supervised`
Expected: exit 0.

- [ ] **Step 3: Prepare real data**

Try to place Weather dataset at `PatchTST_supervised/dataset/weather.csv`. If unavailable, use synthetic data only and report that real training is blocked by missing data.

- [ ] **Step 4: Start training**

Run the benchmark with fixed epochs in the background:
`nohup python -m experiments.weather_benchmark.run --data-path PatchTST_supervised/dataset/weather.csv --epochs 5 --models linear exponential_smoothing gru patchtst > experiments/weather_benchmark/results/background.out 2>&1 &`

- [ ] **Step 5: Report process and outputs**

Report PID, log path, and where metrics will appear.
