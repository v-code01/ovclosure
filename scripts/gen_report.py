"""Regenerates results/summary.json, results/bench.json, and results/report.txt
from scratch. report.txt is the literal text claims.toml points its evidence
substrings at, so every number in README.md is grep-checkable against a file
this script produces deterministically (same seeds throughout).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench.bench_reduction import run_all as run_bench
from src.analyze import analyze_model
from src.null_baseline import null_summary

MODELS = ["gpt2", "qwen2.5-0.5b", "smollm2-135m-instruct", "qwen2.5-1.5b-instruct"]


def main():
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    all_records = []
    lines = []
    lines.append("ovclosure report (regenerate with scripts/gen_report.py)")
    lines.append("")
    lines.append("MODEL TABLE")
    for m in MODELS:
        recs = analyze_model(m, null_samples_per_shape=300, null_seed=0)
        all_records.extend(recs)
        rho = np.array([r["rho"] for r in recs])
        pct = np.array([r["null_percentile"] for r in recs])
        line = (
            f"{m:24s} n_heads={len(recs):4d}  median_rho={np.median(rho):.3f}  "
            f"frac_beat_p99={np.mean(pct > 99.0) * 100:.1f}%  "
            f"n_layers={recs[0]['n_layers']}  d_head={recs[0]['d_head']}"
        )
        lines.append(line)
    with open(results_dir / "summary.json", "w") as f:
        json.dump(all_records, f, indent=1)

    lines.append("")
    lines.append("NULL TABLE")
    for d in sorted({r["d_head"] for r in all_records}):
        s = null_summary(d, n_samples=2000, seed=1)
        lines.append(f"d_head={d}: median={s['p50']:.2f} p1={s['p01']:.2f} p99={s['p99']:.2f}")

    lines.append("")
    lines.append("OVERALL")
    pct_all = np.array([r["null_percentile"] for r in all_records])
    alpha_all = np.array([abs(r["alpha"]) for r in all_records])
    lines.append(f"total heads: {len(all_records)}")
    lines.append(f"overall frac_beat_p99: {np.mean(pct_all > 99.0) * 100:.2f}%")
    smallalpha = int(np.sum(alpha_all < 0.03))
    lines.append(
        f"heads with |alpha|<0.03 (near-nilpotent, rho ill-conditioned): "
        f"{smallalpha} / {len(all_records)}"
    )
    top = max(all_records, key=lambda r: r["rho"])
    lines.append(
        f"largest rho outlier: model={top['model']} layer={top['layer']} "
        f"head={top['head']} rho={top['rho']:.0f} alpha={top['alpha']:.3f}"
    )

    lines.append("")
    lines.append("BENCH TABLE")
    bench = run_bench()
    with open(results_dir / "bench.json", "w") as f:
        json.dump(bench, f, indent=1)
    for b in bench:
        lines.append(
            f"{b['label']:24s} naive_median={b['naive_median_ms']:.1f}ms "
            f"naive_min={b['naive_min_ms']:.1f}ms "
            f"reduced_median={b['reduced_median_ms']:.2f}ms "
            f"reduced_min={b['reduced_min_ms']:.2f}ms "
            f"speedup_median={b['measured_speedup_median']:.0f}x "
            f"speedup_min={b['measured_speedup_min']:.0f}x "
            f"theoretical={b['theoretical_ratio']:.0f}x"
        )

    report = "\n".join(lines) + "\n"
    (results_dir / "report.txt").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
