"""Resource measurement helpers shared by the FD solver and both networks.

- MemorySampler: peak process RSS, sampled every 0.5 s in a background thread
- GpuMonitor: nvidia-smi utilization / memory / clock / temperature log to CSV
- directory and file sizes for storage cost
- small JSON / CSV writers used by every run record
"""
import csv
import json
import os
import platform
import subprocess
import threading
import time

import psutil


class MemorySampler:
    """Samples the resident set size (RSS) of this process in a thread.

    RSS is the physical RAM held by the process. Sampling every 0.5 s can
    miss short spikes, so the peak is a lower bound on the true peak.
    """

    def __init__(self, interval=0.5):
        self.interval = interval
        self.process = psutil.Process(os.getpid())
        self.peak_rss = 0
        self.samples = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.interval)

    def _sample(self):
        rss = self.process.memory_info().rss
        self.peak_rss = max(self.peak_rss, rss)
        self.samples += 1

    def start(self):
        self._sample()
        self._thread.start()
        return self

    def stop(self):
        self._sample()
        self._stop.set()
        self._thread.join()
        return self.peak_rss


class GpuMonitor:
    """Logs nvidia-smi readings once per second to a CSV file for one run."""

    QUERY = "timestamp,utilization.gpu,memory.used,clocks.sm,temperature.gpu"

    def __init__(self, csv_path):
        self.csv_path = csv_path
        self._proc = None
        self._file = None

    def start(self):
        try:
            self._file = open(self.csv_path, "w")
            self._proc = subprocess.Popen(
                ["nvidia-smi", f"--query-gpu={self.QUERY}", "--format=csv", "-l", "1"],
                stdout=self._file, stderr=subprocess.DEVNULL,
            )
        except OSError:
            self._proc = None
        return self

    def stop(self):
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._file is not None:
            self._file.close()

    def summary(self):
        # Mean / max utilization and max memory, clock, temperature over the run
        try:
            with open(self.csv_path) as f:
                rows = list(csv.reader(f))
        except OSError:
            return None
        if len(rows) < 2:
            return None
        values = []
        for row in rows[1:]:
            try:
                values.append([float(c.strip().split()[0]) for c in row[1:5]])
            except (ValueError, IndexError):
                continue
        if not values:
            return None
        util = [r[0] for r in values]
        return {
            "samples": len(values),
            "gpu_utilization_mean_percent": sum(util) / len(util),
            "gpu_utilization_max_percent": max(util),
            "gpu_memory_used_max_mib": max(r[1] for r in values),
            "sm_clock_max_mhz": max(r[2] for r in values),
            "sm_clock_mean_mhz": sum(r[2] for r in values) / len(values),
            "temperature_max_c": max(r[3] for r in values),
        }


def free_ram_bytes():
    return psutil.virtual_memory().available


def file_size(path):
    return os.path.getsize(path) if os.path.exists(path) else None


def directory_size(path):
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            total += os.path.getsize(os.path.join(root, name))
    return total


def write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def read_json(path):
    with open(path) as f:
        return json.load(f)


def write_csv(path, rows):
    # rows: list of dicts with identical keys
    if not rows:
        open(path, "w").close()
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def python_version():
    return platform.python_version()


def timestamp():
    return time.strftime("%Y-%m-%dT%H:%M:%S")
