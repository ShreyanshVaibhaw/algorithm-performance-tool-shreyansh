"""
complexity_analysis.py - Complexity Analysis Module (Task 6)

Puts the theoretical complexity of each algorithm next to what the benchmark
engine measured, and estimates the observed growth trend by fitting a power
law  time = c * n^k  to the (input_size, time) points on log-log axes.  The
fitted exponent k is the "observed growth exponent":  k ~ 1 means linear,
k ~ 2 quadratic, k close to 0 constant or logarithmic.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

# Theoretical complexities as given in the assignment (worst case), plus the
# best/average cases from the lab reference material.  `expected_exponent` is
# the log-log slope the worst-case bound predicts for large n.
THEORETICAL: Dict[str, Dict[str, object]] = {
    "Linear Search": dict(group="search", best="O(1)", average="O(n)", worst="O(n)",
                          space="O(1)", expected_exponent=1.0,
                          note="Examines every element when the key is absent."),
    "Binary Search": dict(group="search", best="O(1)", average="O(log n)", worst="O(log n)",
                          space="O(1)", expected_exponent=0.0,
                          note="Needs sorted input; halves the interval each probe."),
    "Recursive Factorial": dict(group="factorial", best="O(n)", average="O(n)", worst="O(n)",
                                space="O(n)", expected_exponent=1.0,
                                note="n stack frames; the default recursion limit (1000) has to be raised."),
    "Iterative Factorial": dict(group="factorial", best="O(n)", average="O(n)", worst="O(n)",
                                space="O(1)", expected_exponent=1.0,
                                note="One accumulator, no extra frames."),
    "Single Loop Traversal": dict(group="loops", best="O(n)", average="O(n)", worst="O(n)",
                                  space="O(1)", expected_exponent=1.0,
                                  note="One pass over n items."),
    "Nested Loop Traversal": dict(group="loops", best="O(n^2)", average="O(n^2)", worst="O(n^2)",
                                  space="O(1)", expected_exponent=2.0,
                                  note="n*n inner iterations."),
}

ALGORITHM_ORDER = list(THEORETICAL.keys())


def theoretical_table() -> pd.DataFrame:
    rows = []
    for name, info in THEORETICAL.items():
        rows.append({
            "Algorithm / Program": name,
            "Best Case": info["best"],
            "Average Case": info["average"],
            "Worst Case": info["worst"],
            "Time Complexity": info["worst"],
            "Space Complexity": info["space"],
            "Note": info["note"],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Growth-trend estimation
# ---------------------------------------------------------------------------
@dataclass
class GrowthFit:
    exponent: float          # slope of log(time) vs log(n)
    r_squared: float         # how well a single power law explains the points
    points: int

    @property
    def growth_per_10x(self) -> float:
        """Factor by which the metric grows when n grows tenfold."""
        return 10 ** self.exponent


def fit_growth(sizes: Sequence[float], values: Sequence[float]) -> Optional[GrowthFit]:
    """Least-squares fit of log(value) = k*log(n) + c.  Needs two or more valid points."""
    pairs = [(float(n), float(v)) for n, v in zip(sizes, values)
             if n and v and n > 0 and v > 0 and not math.isnan(v)]
    if len(pairs) < 2:
        return None
    x = np.log([p[0] for p in pairs])
    y = np.log([p[1] for p in pairs])
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    ss_res = float(np.sum((y - predicted) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return GrowthFit(float(slope), r2, len(pairs))


def classify_growth(exponent: Optional[float]) -> str:
    """Name the growth class a fitted exponent is closest to."""
    if exponent is None or math.isnan(exponent):
        return "insufficient data"
    if exponent < 0.35:
        return "constant / logarithmic"
    if exponent < 0.75:
        return "sub-linear"
    if exponent < 1.25:
        return "linear"
    if exponent < 1.75:
        return "between linear and quadratic (n log n-like)"
    if exponent < 2.25:
        return "quadratic"
    return "super-quadratic"


def compare_to_theory(expected: float, observed: Optional[float], tolerance: float = 0.3) -> str:
    if observed is None or math.isnan(observed):
        return "not enough data"
    if abs(observed - expected) <= tolerance:
        return "matches theory"
    if observed > expected:
        return "grows faster than the bound predicts"
    return "grows slower than the bound predicts"


# ---------------------------------------------------------------------------
# Tables combining theory and measurement
# ---------------------------------------------------------------------------
def build_complexity_table(results: pd.DataFrame) -> pd.DataFrame:
    """Theoretical vs observed performance, one row per algorithm."""
    rows = []
    for name in ALGORITHM_ORDER:
        info = THEORETICAL[name]
        sub = results[results["algorithm"] == name].sort_values("input_size")
        if sub.empty:
            continue
        fit = fit_growth(sub["input_size"], sub["time_sec"])
        exponent = fit.exponent if fit else math.nan
        smallest, largest = sub.iloc[0], sub.iloc[-1]
        mem = sub["memory_kb"].dropna()
        rows.append({
            "Algorithm / Program": name,
            "Theoretical Time": info["worst"],
            "Theoretical Space": info["space"],
            "n (min)": int(smallest["input_size"]),
            "Time at n (min)": smallest["time_sec"],
            "n (max)": int(largest["input_size"]),
            "Time at n (max)": largest["time_sec"],
            "Time growth (max/min)": (largest["time_sec"] / smallest["time_sec"]) if smallest["time_sec"] > 0 else math.nan,
            "Peak memory min (KB)": mem.min() if not mem.empty else math.nan,
            "Peak memory max (KB)": mem.max() if not mem.empty else math.nan,
            "Observed exponent": exponent,
            "Observed growth trend": classify_growth(exponent),
            "Verdict": compare_to_theory(float(info["expected_exponent"]), exponent),
        })
    return pd.DataFrame(rows)


def growth_factor_table(results: pd.DataFrame, metric: str = "time_sec") -> pd.DataFrame:
    """Each algorithm's metric at every size, relative to its smallest size."""
    table = results.pivot_table(index="algorithm", columns="input_size", values=metric, aggfunc="first")
    table = table.reindex([a for a in ALGORITHM_ORDER if a in table.index])
    relative = table.apply(lambda row: row / row.dropna().iloc[0] if row.dropna().size else row, axis=1)
    return relative


def pairwise_speedup(results: pd.DataFrame, faster: str, slower: str) -> pd.DataFrame:
    """Ratio slower/faster time at each shared input size."""
    a = results[results["algorithm"] == faster].set_index("input_size")["time_sec"]
    b = results[results["algorithm"] == slower].set_index("input_size")["time_sec"]
    shared = a.index.intersection(b.index)
    return pd.DataFrame({
        "input_size": shared,
        f"{faster} (s)": a.loc[shared].values,
        f"{slower} (s)": b.loc[shared].values,
        "speedup (x)": (b.loc[shared] / a.loc[shared]).values,
    })


if __name__ == "__main__":
    print(theoretical_table().to_string(index=False))
    for sizes, times, label in [
        ([100, 1000, 10000], [1e-5, 1e-4, 1e-3], "linear"),
        ([100, 1000, 10000], [1e-4, 1e-2, 1.0], "quadratic"),
        ([100, 1000, 10000], [7e-6, 1e-5, 1.4e-5], "logarithmic"),
    ]:
        fit = fit_growth(sizes, times)
        print(f"{label:12s} exponent={fit.exponent:.2f}  ->  {classify_growth(fit.exponent)}")
