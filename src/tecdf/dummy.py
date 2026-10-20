"""Synthetic flows tagged with institution scenario labels. Not a packet capture.

See data/dummy/README.md. Scoring uses evaluate_arrays, the same functions as
the NSL-KDD benchmark. This module does not read or write evaluation.json.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import numpy as np

from . import __version__
from .framework import evaluate_arrays
from .nsl import NUMERIC

ROOT = Path(__file__).resolve().parents[2]
DUMMY = ROOT / "data" / "dummy"
FEATURES = list(NUMERIC)

STATEMENT = (
    "Synthetic fixture. Flows are not traffic from the named organization. "
    "Do not cite this object as an experimental result."
)


def load_organizations() -> list[dict]:
    with (DUMMY / "organizations.csv").open(newline="") as f:
        return list(csv.DictReader(f))


def load_flows() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    rows = []
    with (DUMMY / "flows.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    x = np.log1p(np.array([[float(r[name]) for name in FEATURES] for r in rows], dtype=float))
    y = np.array([int(r["label"]) for r in rows], dtype=int)
    services = np.array([r["service"] for r in rows])
    klasses = [r["klass"] for r in rows]
    return x, y, services, klasses


def run_dummy() -> dict:
    orgs = load_organizations()
    x, y, services, klasses = load_flows()
    overall = evaluate_arrays(x, y, services, seed=7)
    by_class = {}
    for klass in sorted(set(klasses)):
        idx = np.array([i for i, value in enumerate(klasses) if value == klass])
        if len(idx) < 40 or len(set(y[idx].tolist())) < 2:
            continue
        part = evaluate_arrays(x[idx], y[idx], services[idx], seed=7)
        by_class[klass] = {
            "n_rows": part["n_rows"],
            "majority_class_accuracy": part["majority_class_accuracy"],
            "graph_accuracy": part["graph_service_neighborhood"]["accuracy"],
            "logistic_accuracy": part["numeric_logistic"]["accuracy"],
            "graph_average_precision": part["graph_service_neighborhood"]["average_precision"],
        }
    return {
        "kind": "synthetic-fixture",
        "package": "tecdf-xai",
        "package_version": __version__,
        "statement": STATEMENT,
        "n_organizations": len(orgs),
        "counts": dict(Counter(o["klass"] for o in orgs)),
        "n_flows": int(len(y)),
        "overall": overall,
        "by_class": by_class,
        "klass_row_counts": dict(Counter(klasses)),
    }
