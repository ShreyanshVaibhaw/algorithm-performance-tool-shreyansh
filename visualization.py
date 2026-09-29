"""
visualization.py - Performance Visualization Module (Task 5)

Turns the engine's results table into matplotlib charts:

    search    Dataset size vs execution time   Linear Search vs Binary Search
    factorial Input size vs execution time     Recursive vs Iterative Factorial
    loops     Iterations vs execution time     Single Loop vs Nested Loop

plus memory charts, a comparisons chart for the searches, an all-in-one
log-log overview, and short text summaries of the trends.

Every algorithm keeps one fixed colour across all charts so a reader can
follow it from figure to figure.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")  # render to files / Streamlit without a display
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
import pandas as pd  # noqa: E402

from benchmark_engine import format_seconds  # noqa: E402
from complexity_analysis import fit_growth  # noqa: E402

GRAPHS_DIR = Path(__file__).resolve().parent / "graphs"

# Fixed colour per algorithm (colour-blind-checked categorical palette).
COLORS: Dict[str, str] = {
    "Linear Search": "#2a78d6",
    "Binary Search": "#eb6834",
    "Recursive Factorial": "#1baf7a",
    "Iterative Factorial": "#eda100",
    "Single Loop Traversal": "#e87ba4",
    "Nested Loop Traversal": "#008300",
}
MARKERS: Dict[str, str] = {
    "Linear Search": "o", "Binary Search": "s",
    "Recursive Factorial": "o", "Iterative Factorial": "s",
    "Single Loop Traversal": "o", "Nested Loop Traversal": "s",
}
GROUPS: Dict[str, Dict[str, object]] = {
    "search": dict(title="Linear Search vs Binary Search", algorithms=["Linear Search", "Binary Search"],
                   xlabel="Dataset size (n elements)", log_x=True),
    "factorial": dict(title="Recursive vs Iterative Factorial",
                      algorithms=["Recursive Factorial", "Iterative Factorial"],
                      xlabel="Input value n", log_x=False),
    "loops": dict(title="Single Loop vs Nested Loop Traversal",
                  algorithms=["Single Loop Traversal", "Nested Loop Traversal"],
                  xlabel="Number of iterations (n)", log_x=True),
}

INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"


def _style_axes(ax) -> None:
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)


def _thousands(value, _pos=None) -> str:
    if value >= 1:
        return f"{value:,.0f}"
    return f"{value:g}"


def _series(df: pd.DataFrame, algorithm: str, metric: str) -> pd.DataFrame:
    sub = df[df["algorithm"] == algorithm].sort_values("input_size")
    return sub[["input_size", metric]].dropna()


def plot_metric(df: pd.DataFrame, group: str, metric: str = "time_ms", *,
                ylabel: str = "Execution time (ms)", log_x: Optional[bool] = None,
                log_y: bool = False, title: Optional[str] = None,
                annotate_last: bool = True) -> Figure:
    """Line chart of one metric against input size for the two algorithms in a group."""
    spec = GROUPS[group]
    use_log_x = spec["log_x"] if log_x is None else log_x
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=120)
    plotted = 0
    for algorithm in spec["algorithms"]:
        series = _series(df, algorithm, metric)
        if series.empty:
            continue
        ax.plot(series["input_size"], series[metric], marker=MARKERS[algorithm], markersize=7,
                linewidth=2, color=COLORS[algorithm], label=algorithm)
        plotted += 1
        if annotate_last:
            last = series.iloc[-1]
            value = last[metric]
            text = format_seconds(value / 1000) if metric == "time_ms" else f"{value:,.2f}"
            ax.annotate(text, (last["input_size"], value), textcoords="offset points",
                        xytext=(6, 4), fontsize=8.5, color=MUTED)
    if use_log_x:
        ax.set_xscale("log")
        sizes = sorted(df[df["algorithm"].isin(spec["algorithms"])]["input_size"].unique())
        ax.set_xticks(sizes)
        ax.xaxis.set_major_formatter(FuncFormatter(_thousands))
        ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, p: ""))
    else:
        ax.xaxis.set_major_formatter(FuncFormatter(_thousands))
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel(spec["xlabel"])
    ax.set_ylabel(ylabel + ("  [log scale]" if log_y else ""))
    ax.set_title(title or f"{spec['title']}: {ylabel.split(' (')[0].lower()}",
                 loc="left", fontsize=12, color=INK, pad=12)
    _style_axes(ax)
    if plotted >= 2:
        ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


def plot_time(df: pd.DataFrame, group: str, log_y: bool = False) -> Figure:
    return plot_metric(df, group, "time_ms", ylabel="Execution time (ms)", log_y=log_y)


def plot_memory(df: pd.DataFrame, group: str) -> Figure:
    fig = plot_metric(df, group, "memory_kb", ylabel="Peak heap memory (KB)")
    ax = fig.axes[0]
    ax.set_ylim(bottom=0)
    skipped = df[(df["algorithm"].isin(GROUPS[group]["algorithms"])) & (df["memory_kb"].isna())]
    if not skipped.empty:
        sizes = ", ".join(f"{int(n):,}" for n in sorted(skipped["input_size"].unique()))
        ax.text(0.01, -0.28, f"Memory not traced at n = {sizes}: tracemalloc would slow the run past the time budget.",
                transform=ax.transAxes, fontsize=8, color=MUTED)
        fig.tight_layout()
    return fig


def plot_operations(df: pd.DataFrame, group: str = "search") -> Figure:
    label = "Key comparisons" if group == "search" else "Basic operations"
    return plot_metric(df, group, "operations", ylabel=label, log_y=True,
                       title=f"{GROUPS[group]['title']}: {label.lower()} vs input size")


def plot_overview(df: pd.DataFrame) -> Figure:
    """Every algorithm on one log-log chart: slope shows the growth order."""
    fig, ax = plt.subplots(figsize=(9, 5.4), dpi=120)
    for algorithm, color in COLORS.items():
        series = _series(df, algorithm, "time_ms")
        if series.empty:
            continue
        fit = fit_growth(series["input_size"], series["time_ms"])
        label = f"{algorithm} (slope {fit.exponent:.2f})" if fit else algorithm
        ax.plot(series["input_size"], series["time_ms"], marker=MARKERS[algorithm], markersize=6,
                linewidth=2, color=color, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Input size n  [log scale]")
    ax.set_ylabel("Execution time (ms)  [log scale]")
    ax.set_title("All algorithms: execution time growth (steeper line = higher order)",
                 loc="left", fontsize=12, color=INK, pad=12)
    _style_axes(ax)
    ax.legend(frameon=False, fontsize=8.5, ncol=2)
    fig.tight_layout()
    return fig


def plot_growth_exponents(df: pd.DataFrame) -> Figure:
    """Observed log-log slope next to the exponent theory predicts."""
    from complexity_analysis import THEORETICAL
    names, observed, expected = [], [], []
    for name, info in THEORETICAL.items():
        series = _series(df, name, "time_sec")
        fit = fit_growth(series["input_size"], series["time_sec"])
        if fit is None:
            continue
        names.append(name)
        observed.append(fit.exponent)
        expected.append(float(info["expected_exponent"]))
    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=120)
    positions = range(len(names))
    width = 0.38
    ax.bar([p - width / 2 for p in positions], expected, width, color="#c3c2b7", label="Theoretical exponent")
    ax.bar([p + width / 2 for p in positions], observed, width,
           color=[COLORS[n] for n in names], label="Observed exponent")
    for p, value in zip(positions, observed):
        ax.annotate(f"{value:.2f}", (p + width / 2, value), textcoords="offset points",
                    xytext=(0, 3), ha="center", fontsize=8.5, color=MUTED)
    ax.set_xticks(list(positions))
    ax.set_xticklabels([n.replace(" Traversal", "").replace(" Factorial", "\nFactorial") for n in names], fontsize=9)
    ax.set_ylabel("Growth exponent k in time = c * n^k")
    ax.set_title("Observed growth exponent vs theory (1 = linear, 2 = quadratic, ~0 = logarithmic)",
                 loc="left", fontsize=12, color=INK, pad=12)
    _style_axes(ax)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


def plot_bar_comparison(labels: List[str], values: List[float], *, ylabel: str, title: str,
                        log_y: bool = False, formatter=None) -> Figure:
    """Bar chart for a one-off comparison (used by the Code Benchmarking page)."""
    fig, ax = plt.subplots(figsize=(7, 4), dpi=120)
    colors = [COLORS.get(label, "#2a78d6") for label in labels]
    bars = ax.bar(range(len(labels)), values, color=colors, width=0.6)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([l.replace(" Traversal", "").replace(" Factorial", "\nFactorial") for l in labels],
                       fontsize=9)
    if log_y and all(v > 0 for v in values):
        ax.set_yscale("log")
        ylabel += "  [log scale]"
    for bar, value in zip(bars, values):
        text = formatter(value) if formatter else f"{value:,.2f}"
        ax.annotate(text, (bar.get_x() + bar.get_width() / 2, value), textcoords="offset points",
                    xytext=(0, 3), ha="center", fontsize=8.5, color=MUTED)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=12)
    _style_axes(ax)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Batch generation and text summaries
# ---------------------------------------------------------------------------
def generate_all_charts(df: pd.DataFrame, out_dir: Path | str = GRAPHS_DIR) -> Dict[str, Path]:
    """Render every chart to PNG and return {chart name: path}."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "search_time": plot_time(df, "search"),
        "search_time_log": plot_time(df, "search", log_y=True),
        "search_comparisons": plot_operations(df, "search"),
        "search_memory": plot_memory(df, "search"),
        "factorial_time": plot_time(df, "factorial"),
        "factorial_memory": plot_memory(df, "factorial"),
        "loops_time": plot_time(df, "loops"),
        "loops_time_log": plot_time(df, "loops", log_y=True),
        "loops_memory": plot_memory(df, "loops"),
        "overview_time_loglog": plot_overview(df),
        "growth_exponents": plot_growth_exponents(df),
    }
    paths: Dict[str, Path] = {}
    for name, fig in charts.items():
        path = out_dir / f"{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths[name] = path
    return paths


def _ratio_sentence(df: pd.DataFrame, algorithm: str) -> Optional[str]:
    series = _series(df, algorithm, "time_sec")
    if len(series) < 2:
        return None
    first, last = series.iloc[0], series.iloc[-1]
    size_ratio = last["input_size"] / first["input_size"]
    time_ratio = last["time_sec"] / first["time_sec"] if first["time_sec"] > 0 else math.nan
    fit = fit_growth(series["input_size"], series["time_sec"])
    slope = f" (log-log slope {fit.exponent:.2f})" if fit else ""
    return (f"{algorithm}: input grew {size_ratio:,.0f}x from {int(first['input_size']):,} to "
            f"{int(last['input_size']):,}; time grew {time_ratio:,.1f}x from "
            f"{format_seconds(first['time_sec'])} to {format_seconds(last['time_sec'])}{slope}.")


def summarize_trends(df: pd.DataFrame) -> List[str]:
    """Plain-language statements about what the measurements show."""
    lines: List[str] = []
    for group, spec in GROUPS.items():
        algos = spec["algorithms"]
        for algorithm in algos:
            sentence = _ratio_sentence(df, algorithm)
            if sentence:
                lines.append(sentence)
        a, b = algos
        sa, sb = _series(df, a, "time_sec"), _series(df, b, "time_sec")
        if not sa.empty and not sb.empty:
            shared = sorted(set(sa["input_size"]) & set(sb["input_size"]))
            if shared:
                n = shared[-1]
                ta = float(sa[sa["input_size"] == n]["time_sec"].iloc[0])
                tb = float(sb[sb["input_size"] == n]["time_sec"].iloc[0])
                if ta > 0 and tb > 0:
                    faster, slower, ratio = (a, b, tb / ta) if ta < tb else (b, a, ta / tb)
                    lines.append(f"At n = {int(n):,}, {faster} is {ratio:,.1f}x faster than {slower}.")
    return lines


if __name__ == "__main__":
    from benchmark_engine import load_results

    results = load_results()
    if results is None:
        raise SystemExit("No results found. Run `python benchmark_engine.py` first.")
    for name, path in generate_all_charts(results).items():
        print(f"saved {path}")
    print()
    for line in summarize_trends(results):
        print("-", line)
