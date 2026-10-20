#!/usr/bin/env python3
"""Score the synthetic flows. Does not touch evaluation.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tecdf.dummy import run_dummy  # noqa: E402


def main() -> None:
    report = run_dummy()
    out = ROOT / "benchmarks" / "results" / "dummy_evaluation.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    g = report["overall"]["graph_service_neighborhood"]
    print(f"synthetic graph acc {g['accuracy']:.3f} rows {report['n_flows']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
