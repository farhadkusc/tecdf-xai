"""NSL-KDD 20% loader.

Column order is the standard 41 KDD features, then the label, then the
difficulty score. The file in data/raw is the public 20% training split.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "data" / "raw" / "KDDTrain+_20Percent.txt"

# Indices used by the model. Names follow the NSL-KDD feature list.
NUMERIC = {
    "duration": 0,
    "src_bytes": 4,
    "dst_bytes": 5,
    "count": 23,
    "srv_count": 24,
    "serror_rate": 25,
    "same_srv_rate": 28,
    "dst_host_count": 31,
    "dst_host_srv_count": 32,
    "dst_host_serror_rate": 37,
}
SERVICE = 2
LABEL = 41


def load() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    services = []
    nums = []
    labels = []
    with PATH.open() as f:
        for line in f:
            parts = line.rstrip("\n").split(",")
            if len(parts) < 42:
                continue
            services.append(parts[SERVICE])
            nums.append([float(parts[i]) for i in NUMERIC.values()])
            labels.append(0 if parts[LABEL] == "normal" else 1)
    x = np.log1p(np.asarray(nums, dtype=float))
    y = np.asarray(labels, dtype=int)
    s = np.asarray(services)
    return x, y, s
