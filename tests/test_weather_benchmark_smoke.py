import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


class WeatherBenchmarkSmokeTest(unittest.TestCase):
    def test_benchmark_writes_metrics_for_all_models(self):
        from experiments.weather_benchmark.run import run_benchmark

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            data_path = tmp / "weather.csv"
            output_dir = tmp / "results"

            rows = 900
            index = np.arange(rows, dtype=np.float32)
            frame = pd.DataFrame(
                {
                    "date": pd.date_range("2021-01-01", periods=rows, freq="h").astype(str),
                    "temp": np.sin(index / 24.0),
                    "humidity": np.cos(index / 48.0),
                    "OT": np.sin(index / 12.0) + 0.1 * np.cos(index / 6.0),
                }
            )
            frame.to_csv(data_path, index=False)

            results = run_benchmark(
                data_path=data_path,
                output_dir=output_dir,
                models=("linear", "exponential_smoothing", "gru", "patchtst"),
                seq_len=24,
                pred_len=6,
                batch_size=32,
                epochs=1,
                target="OT",
                device_name="cpu",
                patchtst_d_model=8,
                patchtst_heads=2,
                patchtst_layers=1,
                gru_hidden_size=8,
            )

            self.assertEqual({result["model"] for result in results}, {"linear", "exponential_smoothing", "gru", "patchtst"})
            metrics_path = output_dir / "metrics.csv"
            summary_path = output_dir / "summary.json"
            log_path = output_dir / "run.log"
            self.assertTrue(metrics_path.exists())
            self.assertTrue(summary_path.exists())
            self.assertTrue(log_path.exists())

            with metrics_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 4)
            self.assertIn("MSE", rows[0])


if __name__ == "__main__":
    unittest.main()
