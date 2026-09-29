"""
run_benchmarks.py - one command for the whole experiment pipeline.

    python run_benchmarks.py           # full assignment sizes (nested loop up to 50,000; ~6 min)
    python run_benchmarks.py --quick   # nested loop capped at 10,000 (~20 s)

Steps
  1. run the automated suite           -> results/benchmark_results.csv, results/environment.json
  2. render every chart                -> graphs/*.png
  3. theoretical-vs-observed table     -> results/complexity_table.csv
  4. plain-language trend summary      -> results/trend_summary.txt
"""
from __future__ import annotations

import argparse
import time

import pandas as pd

from benchmark_engine import (RESULTS_CSV, RESULTS_DIR, BenchmarkEngine, format_kb,
                              format_seconds, run_automated_suite, save_environment)
from complexity_analysis import build_complexity_table, growth_factor_table
from visualization import generate_all_charts, summarize_trends


def progress(done: int, total: int, label: str) -> None:
    print(f"  [{done:>2}/{total}] {label}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="cap the nested loop at 10,000 iterations")
    parser.add_argument("--no-memory", action="store_true", help="skip the memory pass entirely")
    args = parser.parse_args()

    sizes = {"loops": [100, 1_000, 10_000]} if args.quick else None
    started = time.perf_counter()

    print("Step 1/4  running the automated benchmark suite")
    engine = BenchmarkEngine(measure_memory=not args.no_memory)
    results = run_automated_suite(sizes, engine=engine, on_progress=progress)
    csv_path = engine.save_csv(RESULTS_CSV)
    env_path = save_environment()
    print(f"          saved {csv_path.name} and {env_path.name} in {RESULTS_DIR}")

    print("Step 2/4  rendering charts")
    for name, path in generate_all_charts(results).items():
        print(f"          {path.name}")

    print("Step 3/4  building the complexity table")
    table = build_complexity_table(results)
    table_path = RESULTS_DIR / "complexity_table.csv"
    table.to_csv(table_path, index=False)
    growth = growth_factor_table(results)
    growth.to_csv(RESULTS_DIR / "growth_factors.csv")
    print(f"          saved {table_path.name} and growth_factors.csv")

    print("Step 4/4  summarising trends")
    lines = summarize_trends(results)
    summary_path = RESULTS_DIR / "trend_summary.txt"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for line in lines:
        print(f"          - {line}")

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    pretty = results[["algorithm", "input_size", "time_sec", "memory_kb", "operations", "runs"]].copy()
    pretty["time"] = pretty["time_sec"].map(format_seconds)
    pretty["memory"] = pretty["memory_kb"].map(format_kb)
    print()
    print(pretty[["algorithm", "input_size", "time", "memory", "operations", "runs"]].to_string(index=False))
    print()
    print(table[["Algorithm / Program", "Theoretical Time", "Observed exponent",
                 "Observed growth trend", "Verdict"]].to_string(index=False))
    print(f"\nFinished in {format_seconds(time.perf_counter() - started)}")


if __name__ == "__main__":
    main()
