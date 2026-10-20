#!/usr/bin/env python3
"""Evaluate the service graph and the numeric logistic on NSL-KDD 20%."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tecdf.framework import evaluate  # noqa: E402


def main() -> None:
    report = evaluate(seed=7)
    out = ROOT / "benchmarks" / "results" / "evaluation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    g = report["graph_service_neighborhood"]
    a = report["graph_after_service_rewire"]
    n = report["numeric_logistic"]
    print(f"graph acc {g['accuracy']:.3f} F1 {g['f1']:.3f} AP {g['average_precision']:.3f}")
    print(f"logistic acc {n['accuracy']:.3f} F1 {n['f1']:.3f} AP {n['average_precision']:.3f}")
    print(f"graph after 20% service rewire acc {a['accuracy']:.3f}")
    print(f"explanation cosine {report['explanation_cosine_stability']:.3f}  paper R_e {report['explanation_paper_R_e']:.3f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
