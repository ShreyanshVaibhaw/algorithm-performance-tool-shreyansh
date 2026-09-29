"""
benchmark_engine.py - Automated Benchmarking Engine (Task 4)

One reusable engine that runs any function across a list of input sizes and
records, for each size:

    time_sec      median wall-clock time of several runs (time.perf_counter)
    memory_kb     peak extra heap memory allocated during one call (tracemalloc)
    operations    comparisons / iterations / calls reported by the function

Measurement method
------------------
* Timing and memory are measured in SEPARATE runs.  tracemalloc hooks every
  allocation and slows allocation-heavy loops by roughly 50-60x on this
  machine, so timing a traced run would report the profiler, not the algorithm.
* Timing repeats the call until it has at least `repeats` samples and 50 ms of
  total time, stops early after a 3 s budget (so a 3-minute loop runs once), and
  reports the median.  Garbage collection is paused during timing, as timeit does.
* Argument construction (e.g. generating the dataset) happens before the clock
  starts, so only the algorithm itself is measured.
* If a traced run is predicted to exceed `memory_time_budget` seconds, the
  memory pass is skipped and the reason is recorded in `memory_method`.
  Otherwise the nested loop at n = 50,000 would take about three hours to trace.
* `memory_backend="memory_profiler"` samples the process's resident set size
  instead (whole-process, page granularity); it is offered for comparison.
"""
from __future__ import annotations

import gc
import math
import statistics
import time
import tracemalloc
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from code_benchmark import (factorial_iterative, factorial_recursive,
                            nested_loop, single_loop)
from search_analysis import (binary_search, choose_key, generate_dataset,
                             linear_search)

# Input sizes suggested by the assignment (Task 4).
DEFAULT_INPUT_SIZES: Dict[str, List[int]] = {
    "search": [100, 1_000, 10_000, 50_000],
    "factorial": [100, 500, 1_000],
    "loops": [100, 1_000, 10_000, 50_000],
}

GROUP_LABELS = {
    "search": "Search algorithms",
    "factorial": "Factorial programs",
    "loops": "Loop-based programs",
}

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_CSV = RESULTS_DIR / "benchmark_results.csv"

ProgressCallback = Callable[[int, int, str], None]


# ---------------------------------------------------------------------------
# Formatting helpers (shared by the app, the notebook and the report)
# ---------------------------------------------------------------------------
def format_seconds(seconds: float) -> str:
    """Pretty-print a duration with a sensible unit."""
    if seconds is None or (isinstance(seconds, float) and math.isnan(seconds)):
        return "n/a"
    if seconds >= 1:
        return f"{seconds:.2f} s"
    if seconds >= 1e-3:
        return f"{seconds * 1e3:.2f} ms"
    return f"{seconds * 1e6:.1f} us"


def format_kb(kb: float) -> str:
    if kb is None or (isinstance(kb, float) and math.isnan(kb)):
        return "n/a"
    if kb >= 1024:
        return f"{kb / 1024:.2f} MB"
    if kb >= 1:
        return f"{kb:.2f} KB"
    return f"{kb * 1024:.0f} bytes"


# ---------------------------------------------------------------------------
# Low-level measurement primitives
# ---------------------------------------------------------------------------
def time_function(func: Callable, args: Sequence[Any] = (), *, repeats: int = 5,
                  min_time: float = 0.05, time_budget: float = 3.0,
                  max_runs: int = 1000) -> Tuple[List[float], Any]:
    """Time func(*args) repeatedly and return (list of run times, last result).

    Runs until there are at least `repeats` samples covering `min_time`
    seconds in total, but stops as soon as `time_budget` seconds have been
    spent, so slow functions run once and fast ones run many times.
    """
    times: List[float] = []
    total = 0.0
    result = None
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        while True:
            start = time.perf_counter()
            result = func(*args)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            total += elapsed
            if total >= time_budget:
                break
            if len(times) >= repeats and total >= min_time:
                break
            if len(times) >= max_runs:
                break
    finally:
        if gc_was_enabled:
            gc.enable()
    return times, result


def measure_peak_memory(func: Callable, args: Sequence[Any] = ()) -> Tuple[float, Any]:
    """Peak extra heap memory (KB) allocated while func(*args) runs, via tracemalloc.

    Only Python heap allocations are traced.  Interpreter call frames live in a
    separate stack area that tracemalloc does not see, so recursion depth is
    reported through the `operations` count instead.
    """
    gc.collect()
    was_tracing = tracemalloc.is_tracing()
    if not was_tracing:
        tracemalloc.start()
    tracemalloc.reset_peak()
    before, _ = tracemalloc.get_traced_memory()
    try:
        result = func(*args)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        if not was_tracing:
            tracemalloc.stop()
    return max(peak - before, 0) / 1024.0, result


def measure_rss_memory(func: Callable, args: Sequence[Any] = (),
                       interval: float = 0.005) -> Tuple[float, Any]:
    """Peak growth of the process's resident set size (KB) while func runs.

    A sampling thread reads the whole process's RSS through
    memory_profiler.memory_usage every `interval` seconds while the function
    runs in the main thread.  This is what a task manager would show: page
    granularity, so small algorithms often report 0 KB here even when
    tracemalloc sees a few hundred bytes of heap.

    memory_profiler's own function mode (memory_usage((f, args))) is not used
    because it launches a monitoring subprocess, which hangs on Windows when
    the interpreter was started from Streamlit or a notebook kernel.
    """
    import threading

    from memory_profiler import memory_usage  # imported lazily: optional backend

    def sample() -> float:  # MiB
        value = memory_usage(-1, interval=0.0, max_usage=True)
        return float(value if isinstance(value, (int, float)) else max(value))

    baseline = sample()
    peak = baseline
    stop = threading.Event()

    def watch() -> None:
        nonlocal peak
        while not stop.is_set():
            peak = max(peak, sample())
            stop.wait(interval)

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    try:
        result = func(*args)
    finally:
        stop.set()
        watcher.join()
    peak = max(peak, sample())
    return max(peak - baseline, 0.0) * 1024.0, result


_TRACE_OVERHEAD: Optional[float] = None


def tracing_overhead_factor(force: bool = False) -> float:
    """How many times slower an allocation-heavy loop runs under tracemalloc.

    Measured once per process with a small nested loop so the engine can
    predict whether a traced run fits the memory time budget.
    """
    global _TRACE_OVERHEAD
    if _TRACE_OVERHEAD is None or force:
        probe = 300
        times, _ = time_function(nested_loop, (probe,), repeats=3, min_time=0.0, time_budget=1.0)
        untraced = statistics.median(times)
        start = time.perf_counter()
        measure_peak_memory(nested_loop, (probe,))
        traced = time.perf_counter() - start
        _TRACE_OVERHEAD = max(traced / untraced, 1.0) if untraced > 0 else 60.0
    return _TRACE_OVERHEAD


_NESTED_COST: Optional[float] = None


def estimate_nested_loop_seconds(n: int) -> float:
    """Rough prediction of how long nested_loop(n) takes here (n*n iterations).

    Calibrated once per process from a small probe so the app can warn before
    starting a multi-minute run.
    """
    global _NESTED_COST
    if _NESTED_COST is None:
        probe = 400
        times, _ = time_function(nested_loop, (probe,), repeats=3, min_time=0.0, time_budget=1.0)
        _NESTED_COST = statistics.median(times) / (probe * probe)
    return _NESTED_COST * n * n


# ---------------------------------------------------------------------------
# Result record
# ---------------------------------------------------------------------------
@dataclass
class Measurement:
    algorithm: str
    group: str
    input_size: int
    time_sec: float           # median of the timed runs
    time_min_sec: float
    time_max_sec: float
    runs: int
    memory_kb: float          # NaN when the memory pass was skipped
    memory_method: str
    operations: Optional[int]
    result_summary: str = ""

    @property
    def time_ms(self) -> float:
        return self.time_sec * 1000.0

    def to_row(self) -> dict:
        row = asdict(self)
        row["time_ms"] = self.time_ms
        return row


COLUMN_ORDER = ["algorithm", "group", "input_size", "time_sec", "time_ms",
                "time_min_sec", "time_max_sec", "runs", "memory_kb",
                "memory_method", "operations", "result_summary"]


def _summarize_result(result: Any) -> str:
    if hasattr(result, "describe"):
        return result.describe()
    if hasattr(result, "value_summary"):
        return result.value_summary()
    text = repr(result)
    return text if len(text) <= 60 else text[:57] + "..."


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------
class BenchmarkEngine:
    """Run functions across input sizes and collect Measurement records."""

    def __init__(self, *, repeats: int = 5, min_time: float = 0.05,
                 time_budget: float = 3.0, measure_memory: bool = True,
                 memory_backend: str = "tracemalloc",
                 memory_time_budget: float = 30.0):
        if memory_backend not in ("tracemalloc", "memory_profiler"):
            raise ValueError("memory_backend must be 'tracemalloc' or 'memory_profiler'")
        self.repeats = repeats
        self.min_time = min_time
        self.time_budget = time_budget
        self.measure_memory = measure_memory
        self.memory_backend = memory_backend
        self.memory_time_budget = memory_time_budget
        self.results: List[Measurement] = []

    # -- single measurement -------------------------------------------------
    def measure(self, func: Callable, args: Sequence[Any] = (), *, name: Optional[str] = None,
                group: str = "custom", input_size: Optional[int] = None) -> Measurement:
        """Time func(*args), then measure its memory in a separate run."""
        name = name or func.__name__
        times, result = time_function(func, args, repeats=self.repeats,
                                      min_time=self.min_time, time_budget=self.time_budget)
        median = statistics.median(times)
        memory_kb, method = self._measure_memory(func, args, median)
        record = Measurement(
            algorithm=name,
            group=group,
            input_size=input_size if input_size is not None else (args[0] if args else 0),
            time_sec=median,
            time_min_sec=min(times),
            time_max_sec=max(times),
            runs=len(times),
            memory_kb=memory_kb,
            memory_method=method,
            operations=getattr(result, "operations", None),
            result_summary=_summarize_result(result),
        )
        self.results.append(record)
        return record

    def _measure_memory(self, func: Callable, args: Sequence[Any],
                        untraced_time: float) -> Tuple[float, str]:
        if not self.measure_memory:
            return math.nan, "not measured (disabled)"
        if self.memory_backend == "memory_profiler":
            try:
                kb, _ = measure_rss_memory(func, args)
                return kb, "memory_profiler (process RSS)"
            except Exception as exc:  # backend unavailable in this environment
                return math.nan, f"not measured ({exc.__class__.__name__})"
        predicted = untraced_time * tracing_overhead_factor()
        if predicted > self.memory_time_budget:
            return math.nan, f"not measured (tracemalloc run predicted ~{predicted / 60:.0f} min)"
        kb, _ = measure_peak_memory(func, args)
        return kb, "tracemalloc (peak heap)"

    # -- one function across many sizes -----------------------------------
    def run(self, func: Callable, input_sizes: Sequence[int], *, name: Optional[str] = None,
            group: str = "custom", arg_builder: Callable[[int], Sequence[Any]] = lambda n: (n,),
            on_progress: Optional[ProgressCallback] = None) -> "BenchmarkEngine":
        """Benchmark `func` for every n in input_sizes; arg_builder(n) builds its arguments."""
        name = name or func.__name__
        for index, n in enumerate(input_sizes):
            if on_progress:
                on_progress(index, len(input_sizes), f"{name} (n = {n:,})")
            self.measure(func, arg_builder(n), name=name, group=group, input_size=n)
        if on_progress:
            on_progress(len(input_sizes), len(input_sizes), f"{name} done")
        return self

    # -- output -------------------------------------------------------------
    def as_dataframe(self) -> pd.DataFrame:
        if not self.results:
            return pd.DataFrame(columns=COLUMN_ORDER)
        df = pd.DataFrame([m.to_row() for m in self.results])
        return df[COLUMN_ORDER]

    def save_csv(self, path: Path | str = RESULTS_CSV) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.as_dataframe().to_csv(path, index=False)
        return path

    def clear(self) -> None:
        self.results.clear()


# ---------------------------------------------------------------------------
# The automated suite (Task 4)
# ---------------------------------------------------------------------------
def build_suite(sizes: Optional[Dict[str, Sequence[int]]] = None, seed: int = 42):
    """Return [(name, group, func, sizes, arg_builder), ...] for the whole study.

    Searches use a sorted dataset and a key that is absent, i.e. the worst case:
    linear search must examine every element and binary search must exhaust
    its interval.  The dataset is built before timing starts.
    """
    merged = {**DEFAULT_INPUT_SIZES, **(sizes or {})}

    def search_args(n: int):
        data = generate_dataset(n, seed=seed)
        return (data, choose_key(data, present=False))

    def single(n: int):
        return (n,)

    return [
        ("Linear Search", "search", linear_search, list(merged["search"]), search_args),
        ("Binary Search", "search", binary_search, list(merged["search"]), search_args),
        ("Recursive Factorial", "factorial", factorial_recursive, list(merged["factorial"]), single),
        ("Iterative Factorial", "factorial", factorial_iterative, list(merged["factorial"]), single),
        ("Single Loop Traversal", "loops", single_loop, list(merged["loops"]), single),
        ("Nested Loop Traversal", "loops", nested_loop, list(merged["loops"]), single),
    ]


def run_automated_suite(sizes: Optional[Dict[str, Sequence[int]]] = None, *,
                        engine: Optional[BenchmarkEngine] = None, seed: int = 42,
                        groups: Optional[Sequence[str]] = None,
                        on_progress: Optional[ProgressCallback] = None) -> pd.DataFrame:
    """Run every algorithm and code snippet across its input sizes."""
    engine = engine or BenchmarkEngine()
    plan = [p for p in build_suite(sizes, seed) if groups is None or p[1] in groups]
    total = sum(len(p[3]) for p in plan)
    done = 0
    for name, group, func, group_sizes, builder in plan:
        for n in group_sizes:
            if on_progress:
                on_progress(done, total, f"{name} (n = {n:,})")
            engine.measure(func, builder(n), name=name, group=group, input_size=n)
            done += 1
    if on_progress:
        on_progress(total, total, "Benchmark complete")
    return engine.as_dataframe()


def load_results(path: Path | str = RESULTS_CSV) -> Optional[pd.DataFrame]:
    """Load a previously saved results table, or None when it does not exist."""
    path = Path(path)
    if not path.exists():
        return None
    return pd.read_csv(path)


def environment_info() -> Dict[str, str]:
    """Machine and interpreter details worth recording next to the results."""
    import os
    import platform

    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "os": platform.platform(),
        "processor": platform.processor() or "unknown",
        "cpu_count": str(os.cpu_count()),
        "timer": "time.perf_counter",
        "memory": "tracemalloc peak heap (separate run)",
        "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def save_environment(path: Path | str = RESULTS_DIR / "environment.json") -> Path:
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(environment_info(), indent=2), encoding="utf-8")
    return path


def _print_progress(done: int, total: int, label: str) -> None:
    print(f"[{done:>2}/{total}] {label}", flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the automated benchmark suite.")
    parser.add_argument("--quick", action="store_true",
                        help="cap the nested loop at 10,000 iterations (skips the 3-minute case)")
    parser.add_argument("--output", default=str(RESULTS_CSV), help="CSV file to write")
    cli = parser.parse_args()

    sizes = None
    if cli.quick:
        sizes = {"loops": [100, 1_000, 10_000]}

    started = time.perf_counter()
    engine = BenchmarkEngine()
    frame = run_automated_suite(sizes, engine=engine, on_progress=_print_progress)
    engine.save_csv(cli.output)

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    show = frame[["algorithm", "input_size", "time_ms", "memory_kb", "operations", "runs", "memory_method"]]
    print()
    print(show.to_string(index=False))
    print(f"\nSaved {len(frame)} rows to {cli.output}  ({time.perf_counter() - started:.1f} s total)")
