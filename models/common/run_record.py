"""Run records and cost measurements shared by both neural networks.

Every training run writes, into <experiment>/<method>/<run_id>/:
  config.json        settings used (week-03 schema + seed, device, versions)
  history.csv        loss components every 10 steps
  eval_history.csv   test-A relative L2 and wall-clock time every 500 steps
  model.keras        trained network
  predictions.npz    full 51 x 64 x 64 field (x, y, t, u, v, p)
  performance.json   timing, throughput, inference latency, memory, storage
  gpu_monitor.csv    nvidia-smi log (GPU runs)
"""
import json
import os
import platform
import statistics
import time

import numpy as np

from models.common import config as C
from models.common import profiling
from models.common.dataset import scale_inputs

INFERENCE_BATCHES = [1, 64, 1024, 16384]   # plus the full field
INFERENCE_WARMUP = 3
INFERENCE_REPEATS = 20


class RunRecorder:
    def __init__(self, method, run_id, device, seed, reynolds_number,
                 exp_dir=C.EXPERIMENT_DIR):
        self.method = method
        self.run_id = run_id
        self.device = device
        self.seed = seed
        self.re = reynolds_number
        self.run_dir = os.path.join(exp_dir, method, run_id)
        os.makedirs(self.run_dir, exist_ok=True)
        self.history = []
        self.eval_history = []
        self.sampler = None
        self.gpu_monitor = None
        self.free_ram_at_start = None
        self.wall_start = None

    def path(self, name):
        return os.path.join(self.run_dir, name)

    def start(self):
        import tensorflow as tf
        self.free_ram_at_start = profiling.free_ram_bytes()
        self.sampler = profiling.MemorySampler().start()
        if self.device == "gpu":
            tf.config.experimental.reset_memory_stats("GPU:0")
            self.gpu_monitor = profiling.GpuMonitor(self.path("gpu_monitor.csv")).start()
        self.wall_start = time.perf_counter()

    def wall(self):
        return time.perf_counter() - self.wall_start

    def write_config(self, cfg):
        import tensorflow as tf
        base = {
            "run_id": self.run_id,
            "method": self.method,
            "reynolds_number": self.re,
            "seed": self.seed,
            "device": self.device,
            "tf_version": tf.__version__,
            "python_version": platform.python_version(),
            "deterministic_ops": True,
            "created": profiling.timestamp(),
        }
        base.update(cfg)
        profiling.write_json(self.path("config.json"), base)

    def stop_monitors(self):
        import tensorflow as tf
        peak_vram = None
        if self.device == "gpu":
            peak_vram = int(tf.config.experimental.get_memory_info("GPU:0")["peak"])
            self.gpu_monitor.stop()
        peak_rss = self.sampler.stop()
        return peak_rss, peak_vram

    def finish(self, model, timing, extra_perf=None, save_model=None):
        # model: callable on scaled inputs returning physical (u, v, p)
        # save_model: the Keras network to store (defaults to model)
        """Benchmarks inference, saves model / predictions / history, writes performance.json."""
        self.flush_history()
        inference = inference_benchmark(model)
        np.savez(self.path("predictions.npz"), **predict_full_field(model))
        (save_model or model).save(self.path("model.keras"))
        peak_rss, peak_vram = self.stop_monitors()

        perf = {
            "timing": {
                "total_runtime_seconds": self.wall(),
                **timing,
                "inference_time_seconds": inference["full_field_seconds"],
            },
            "inference": inference,
            "memory": {
                "peak_ram_bytes": peak_rss,
                "peak_vram_bytes": peak_vram,
                "free_ram_at_start_bytes": self.free_ram_at_start,
                "ram_sample_interval_seconds": self.sampler.interval,
            },
            "gpu_monitor": self.gpu_monitor.summary() if self.gpu_monitor else None,
            "storage": {
                "model_size_bytes": profiling.file_size(self.path("model.keras")),
                "output_size_bytes": profiling.file_size(self.path("predictions.npz")),
            },
        }
        if extra_perf:
            perf.update(extra_perf)
        profiling.write_json(self.path("performance.json"), perf)
        # Run-folder size includes performance.json itself, so write twice
        perf["storage"]["total_run_size_bytes"] = profiling.directory_size(self.run_dir)
        profiling.write_json(self.path("performance.json"), perf)
        return perf

    def flush_history(self):
        profiling.write_csv(self.path("history.csv"), self.history)
        profiling.write_csv(self.path("eval_history.csv"), self.eval_history)


def full_field_inputs(re=100.0):
    # The 51 x 64 x 64 FD space-time points in dataset row order
    d = np.load(C.dataset_path(re))
    return d["inputs"], d["x"], d["y"], d["t"]


def predict_full_field(model, re=100.0, batch=65536):
    inputs, x, y, t = full_field_inputs(re)
    scaled = scale_inputs(inputs).astype("float32")
    out = np.concatenate([model(scaled[i:i + batch], training=False).numpy()
                          for i in range(0, len(scaled), batch)])
    shape = (len(t), len(y), len(x))
    return {"x": x, "y": y, "t": t,
            "u": out[:, 0].reshape(shape).astype("float64"),
            "v": out[:, 1].reshape(shape).astype("float64"),
            "p": out[:, 2].reshape(shape).astype("float64")}


def inference_benchmark(model, re=100.0):
    """Median wall time of 20 calls after 3 warm-ups, per batch size.

    Each call runs the compiled forward pass and copies the result to host
    memory (.numpy()), so GPU timings include the device-to-host transfer.
    """
    import tensorflow as tf
    inputs, *_ = full_field_inputs(re)
    scaled = scale_inputs(inputs).astype("float32")
    forward = tf.function(lambda z: model(z, training=False), reduce_retracing=True)

    def time_batch(n):
        z = tf.constant(scaled[:n])
        for _ in range(INFERENCE_WARMUP):
            forward(z).numpy()
        times = []
        for _ in range(INFERENCE_REPEATS):
            t0 = time.perf_counter()
            forward(z).numpy()
            times.append(time.perf_counter() - t0)
        return statistics.median(times)

    sweep = {}
    for n in INFERENCE_BATCHES + [len(scaled)]:
        sweep[str(n)] = time_batch(n)
    full = sweep[str(len(scaled))]
    return {
        "full_field_points": int(len(scaled)),
        "full_field_seconds": full,
        "single_point_seconds": sweep["1"],
        "batch_sweep_seconds": sweep,
        "batch_sweep_points_per_second": {k: int(k) / v for k, v in sweep.items()},
        "method": f"median of {INFERENCE_REPEATS} calls after {INFERENCE_WARMUP} warm-ups",
    }


def relative_l2(pred, ref):
    return float(np.linalg.norm(pred - ref) / np.linalg.norm(ref))


def test_a_metrics(model, data, re=100.0):
    """Relative L2 of u, v, p on test A (p mean-removed per snapshot).

    The per-snapshot pressure means are taken over each full 64 x 64
    snapshot for both the prediction and the FD reference, as in the
    final evaluator.
    """
    pred = predict_full_field(model, re)
    idx = data["idx_test_a"]
    out = {}
    for k, name in enumerate(("u", "v", "p")):
        P = pred[name].reshape(len(data["t"]), -1)
        R = data[name].reshape(len(data["t"]), -1)
        if name == "p":
            P = P - P.mean(axis=1, keepdims=True)
            R = R - R.mean(axis=1, keepdims=True)
        pv = P.ravel()[idx]
        rv = R.ravel()[idx]
        out[f"rel_l2_{name}"] = relative_l2(pv, rv)
    return out
