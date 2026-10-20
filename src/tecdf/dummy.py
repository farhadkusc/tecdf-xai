"""Synthetic flows tagged with institution scenario labels. Not a packet capture.

See data/dummy/README.md. Scoring uses evaluate_arrays, the same functions as
the NSL-KDD benchmark. This module does not read or write evaluation.json.
"""

from __future__ import annotations

import csv
import json
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


def _recall(rows: list[dict], rate: dict[str, float], spoof: str | None) -> float:
    tp = fn = 0
    for row in rows:
        service = spoof if spoof and int(row["label"]) == 1 else row["service"]
        pred = 1 if rate.get(service, 0.5) >= 0.5 else 0
        if int(row["label"]) != 1:
            continue
        if pred == 1:
            tp += 1
        else:
            fn += 1
    return tp / (tp + fn) if tp + fn else 0.0


def score_attack_scenario() -> dict:
    """Grey Harbor inject: malicious flows in the blue cell are relabeled to a quiet service."""
    scenario = json.loads((DUMMY / "attack_scenario.json").read_text())
    phase = next(item for item in scenario["phases"] if item["id"] == "PH1")
    spoof = phase["service_spoof"]
    with (DUMMY / "flows.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    blue_ids = set(scenario["blue_cell"])
    background = [row for row in rows if row["org_id"] not in blue_ids]
    participants = [row for row in rows if row["org_id"] in blue_ids]
    total: dict[str, int] = {}
    attacks: dict[str, int] = {}
    for row in background:
        service = row["service"]
        total[service] = total.get(service, 0) + 1
        attacks[service] = attacks.get(service, 0) + int(row["label"])
    rate = {service: attacks[service] / total[service] for service in total}
    n_spoofed = sum(1 for row in participants if int(row["label"]) == 1 and row["service"] != spoof)
    return {
        "exercise_id": scenario["exercise_id"],
        "title": scenario["title"],
        "kind": scenario["kind"],
        "statement": scenario["statement"],
        "spoofed_service": spoof,
        "blue_cell": scenario["blue_cell"],
        "out_of_scope": scenario["out_of_scope"],
        "sites": [site["id"] for site in scenario["sites"]],
        "n_participant_flows": len(participants),
        "n_spoofed": n_spoofed,
        "recall_before": round(_recall(participants, rate, None), 4),
        "recall_after": round(_recall(participants, rate, spoof), 4),
        "background_service_rates": {service: round(value, 4) for service, value in sorted(rate.items())},
    }


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
        "attack_scenario": score_attack_scenario(),
    }
