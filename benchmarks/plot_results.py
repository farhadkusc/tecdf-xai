#!/usr/bin/env python3
"""Write docs/figures/results.svg from benchmarks/results/evaluation.json."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    report = json.loads((ROOT / "benchmarks" / "results" / "evaluation.json").read_text())
    rows = [
        ("Majority", report["majority_class_accuracy"], None, "#8d6e63"),
        ("Service graph", report["graph_service_neighborhood"]["accuracy"], report["graph_service_neighborhood"]["average_precision"], "#1f4e79"),
        ("Logistic", report["numeric_logistic"]["accuracy"], report["numeric_logistic"]["average_precision"], "#5b8c5a"),
        ("Rewired graph", report["graph_after_service_rewire"]["accuracy"], report["graph_after_service_rewire"]["average_precision"], "#c47b3a"),
    ]
    width, height = 640, 340
    left, bottom, plot_w, plot_h = 56, 56, 540, 220
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="56" y="28" font-family="Georgia, serif" font-size="16" fill="#1a1a1a">NSL-KDD 20% test split</text>',
    ]
    for tick in (0, 0.25, 0.5, 0.75, 1.0):
        y = bottom + plot_h - tick * plot_h
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e6e6e6"/>')
        parts.append(
            f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11" fill="#444">{tick:.2f}</text>'
        )
    group_w = plot_w / len(rows)
    for i, (name, acc, ap, color) in enumerate(rows):
        x = left + i * group_w + 24
        h = acc * plot_h
        y = bottom + plot_h - h
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="28" height="{h:.1f}" fill="{color}"/>')
        parts.append(
            f'<text x="{x + 14:.1f}" y="{y - 6:.1f}" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#222">{acc:.3f}</text>'
        )
        if ap is not None:
            h2 = ap * plot_h
            y2 = bottom + plot_h - h2
            parts.append(f'<rect x="{x + 36:.1f}" y="{y2:.1f}" width="28" height="{h2:.1f}" fill="{color}" opacity="0.45"/>')
        parts.append(
            f'<text x="{x + 32:.1f}" y="{bottom + plot_h + 20}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#222">{name}</text>'
        )
    parts.append('<text x="420" y="52" font-family="sans-serif" font-size="12" fill="#222">Solid: accuracy</text>')
    parts.append('<text x="420" y="70" font-family="sans-serif" font-size="12" fill="#222">Tint: average precision</text>')
    parts.append("</svg>")
    out = ROOT / "docs" / "figures" / "results.svg"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts) + "\n")
    print(out)


if __name__ == "__main__":
    main()
