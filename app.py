"""
app.py - Algorithm Performance Measurement and Benchmarking Tool (Streamlit UI)

Run with:
    streamlit run app.py

Pages map to the assignment tasks:
    Overview                 Task 1  what the tool does and how it measures
    Search Analysis          Task 2  linear vs binary search on generated datasets
    Code Benchmarking        Task 3  four predefined code snippets
    Automated Benchmarking   Task 4  every algorithm across the assignment's input sizes
    Visualization            Task 5  charts and trend summaries
    Complexity Analysis      Task 6  theoretical vs observed complexity
    Comparative Study        Task 7  discussion of the three head-to-head comparisons

Deep links: ?page=search-analysis, ?page=code-benchmarking&autorun=1, ...
"""
from __future__ import annotations

import time
from datetime import datetime
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from benchmark_engine import (DEFAULT_INPUT_SIZES, RESULTS_CSV, BenchmarkEngine,
                              environment_info, estimate_nested_loop_seconds,
                              format_kb, format_seconds, load_results,
                              run_automated_suite, save_environment)
from code_benchmark import CODE_SNIPPETS, SNIPPET_GROUP, get_source
from complexity_analysis import (THEORETICAL, build_complexity_table,
                                 growth_factor_table, pairwise_speedup,
                                 theoretical_table)
from search_analysis import (SEARCH_ALGORITHMS, choose_key, generate_dataset,
                             max_binary_comparisons)
from visualization import (GRAPHS_DIR, GROUPS, generate_all_charts,
                           plot_bar_comparison, plot_growth_exponents,
                           plot_memory, plot_operations, plot_overview,
                           plot_time, summarize_trends)

st.set_page_config(page_title="Algorithm Performance Tool", page_icon=":stopwatch:", layout="wide")

PAGES = [
    "Overview",
    "Search Analysis",
    "Code Benchmarking",
    "Automated Benchmarking",
    "Visualization",
    "Complexity Analysis",
    "Comparative Study",
]
SLUGS = {page: page.lower().replace(" ", "-") for page in PAGES}
PAGE_BY_SLUG = {slug: page for page, slug in SLUGS.items()}
MEMORY_BACKENDS = {
    "tracemalloc (peak heap, exact bytes)": "tracemalloc",
    "memory_profiler (process RSS, page granularity)": "memory_profiler",
}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def show_fig(fig) -> None:
    st.pyplot(fig)
    plt.close(fig)


def get_results() -> Optional[pd.DataFrame]:
    """Results from this session, or the CSV saved by an earlier run."""
    df = st.session_state.get("results")
    if df is None:
        df = load_results()
        if df is not None:
            saved = datetime.fromtimestamp(RESULTS_CSV.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            st.session_state["results"] = df
            st.session_state["results_source"] = f"Loaded from {RESULTS_CSV.name} (saved {saved})."
    return df


def results_notice() -> None:
    st.info("No benchmark results yet. Run the Automated Benchmarking module, or run "
            "`python run_benchmarks.py` in a terminal, then come back to this page.")


def pretty_results(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "Algorithm": df["algorithm"],
        "Input size (n)": df["input_size"].map(lambda v: f"{int(v):,}"),
        "Execution time": df["time_sec"].map(format_seconds),
        "Peak memory": df["memory_kb"].map(format_kb),
        "Operations": df["operations"].map(lambda v: f"{int(v):,}" if pd.notna(v) else "-"),
        "Timed runs": df["runs"],
        "Memory method": df["memory_method"],
    })


def parse_sizes(text: str) -> List[int]:
    sizes = []
    for token in text.replace(";", ",").split(","):
        token = token.strip().replace("_", "")
        if token:
            sizes.append(int(token))
    return sorted(set(sizes))


def size_selector(label: str, presets: List[int], default: int, *, key: str,
                  custom_max: int = 2_000_000) -> int:
    options = [f"{p:,}" for p in presets] + ["Custom"]
    choice = st.selectbox(label, options, index=presets.index(default), key=f"{key}_preset")
    if choice == "Custom":
        return int(st.number_input("Custom value", min_value=1, max_value=custom_max,
                                   value=default, step=100, key=f"{key}_custom"))
    return int(choice.replace(",", ""))


def trend_lines_for(df: pd.DataFrame, group: str) -> List[str]:
    names = GROUPS[group]["algorithms"]
    return [line for line in summarize_trends(df) if any(name in line for name in names)]


@st.cache_data(show_spinner=False)
def cached_dataset(size: int, seed: int) -> List[int]:
    return generate_dataset(size, seed=seed)


def sidebar_navigation() -> str:
    requested = st.query_params.get("page")
    default = PAGE_BY_SLUG.get(requested, PAGES[0])
    st.sidebar.title("Benchmarking Tool")
    page = st.sidebar.radio("Module", PAGES, index=PAGES.index(default))
    if SLUGS[page] != requested:
        st.query_params["page"] = SLUGS[page]
    env = environment_info()
    st.sidebar.caption(f"Python {env['python']} on {env['os'].split('-')[0]}  |  {env['cpu_count']} CPUs")
    st.sidebar.caption("ENCA351 Design and Analysis of Algorithms Lab, Lab Assignment 2")
    st.sidebar.caption("Shreyansh Vaibhaw, roll number 2401201094")
    return page


# ---------------------------------------------------------------------------
# Page: Overview (Task 1)
# ---------------------------------------------------------------------------
def page_overview() -> None:
    st.title("Algorithm Performance Measurement and Benchmarking Tool")
    st.write(
        "Pick an algorithm or a code snippet, choose an input size, run it, and read off the "
        "execution time, the memory it used and how many basic operations it performed. "
        "The automated module repeats this across a range of sizes so growth trends can be "
        "charted and compared with the theoretical complexity."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Search analysis")
        st.write("Linear and binary search on generated datasets of any size, with the number "
                 "of key comparisons, execution time and memory for each run.")
    with col2:
        st.subheader("Code benchmarking")
        st.write("Single loop, nested loop, recursive factorial and iterative factorial, "
                 "executed side by side with a comparison table and charts.")
    with col3:
        st.subheader("Automated study")
        st.write("Every algorithm across 100 to 50,000 elements, charts of size versus time, "
                 "a complexity table and a written comparison.")

    st.subheader("How the measurements are made")
    st.markdown(
        "- **Execution time** comes from `time.perf_counter()` around the call. The call is repeated "
        "until there are at least five samples covering 50 ms, or a 3 s budget is spent, and the "
        "median is reported. Garbage collection is paused while timing, as `timeit` does.\n"
        "- **Memory** is the peak extra heap allocated during one separate run traced with "
        "`tracemalloc`. Timing and tracing are never done in the same run: tracing every allocation "
        "slows a tight Python loop 25x to 60x, so a traced run measures the profiler, not the code.\n"
        "- **Operations** are key comparisons for the searches and loop iterations, multiplications "
        "or recursive calls for the snippets.\n"
        "- **Dataset generation is excluded** from the clock. Only the algorithm is timed.\n"
        "- If tracing a run is predicted to take longer than a budget (30 s by default) the memory "
        "pass is skipped and the reason is written into the results table."
    )

    left, right = st.columns([1, 1])
    with left:
        st.subheader("This machine")
        env = environment_info()
        st.table(pd.DataFrame({"Setting": list(env.keys()), "Value": list(env.values())}))
    with right:
        st.subheader("Where to start")
        st.markdown(
            "1. **Search Analysis**: generate a dataset and search it.\n"
            "2. **Code Benchmarking**: run the four snippets at one input size.\n"
            "3. **Automated Benchmarking**: run the whole study (about six minutes with the "
            "50,000-iteration nested loop).\n"
            "4. **Visualization**, **Complexity Analysis** and **Comparative Study** read those results."
        )
        results = get_results()
        if results is not None:
            st.success(f"Benchmark results are available ({len(results)} rows). "
                       f"{st.session_state.get('results_source', '')}")
        else:
            st.info("No saved results yet. The three analysis pages will fill in after a benchmark run.")


# ---------------------------------------------------------------------------
# Page: Search Analysis (Task 2)
# ---------------------------------------------------------------------------
def page_search() -> None:
    st.title("Search Algorithm Analysis")
    st.write("A digital library wants to know how its record lookups will behave as the catalogue "
             "grows. Generate a dataset, pick a key, and compare linear search with binary search.")

    c1, c2, c3, c4 = st.columns([1, 1, 1.4, 1.4])
    with c1:
        size = size_selector("Dataset size", [100, 1_000, 10_000, 50_000], 10_000, key="search_size")
    with c2:
        seed = int(st.number_input("Random seed", min_value=0, max_value=100_000, value=42))
    with c3:
        key_mode = st.radio("Search key", ["Random key from the dataset (average case)",
                                           "Key not in the dataset (worst case)",
                                           "Custom key"])
    with c4:
        algorithms = st.multiselect("Algorithms", list(SEARCH_ALGORITHMS), default=list(SEARCH_ALGORITHMS))
        backend = MEMORY_BACKENDS[st.selectbox("Memory measurement", list(MEMORY_BACKENDS), key="search_backend")]

    data = cached_dataset(size, seed)
    if key_mode.startswith("Random"):
        target = choose_key(data, present=True, seed=seed)
    elif key_mode.startswith("Key not"):
        target = choose_key(data, present=False)
    else:
        target = int(st.number_input("Key to search for", value=int(data[len(data) // 2])))

    with st.expander(f"Dataset preview: {size:,} distinct integers, sorted ascending (binary search needs sorted input)"):
        head = ", ".join(str(v) for v in data[:15])
        tail = ", ".join(str(v) for v in data[-5:])
        st.code(f"[{head}, ... , {tail}]")
        st.caption(f"Search key: {target}")

    if not algorithms:
        st.warning("Choose at least one algorithm.")
        return

    engine = BenchmarkEngine(memory_backend=backend)
    rows = []
    columns = st.columns(len(algorithms))
    for column, name in zip(columns, algorithms):
        func = SEARCH_ALGORITHMS[name]
        measurement = engine.measure(func, (data, target), name=name, group="search", input_size=size)
        result = func(data, target)
        bound = size if name == "Linear Search" else max_binary_comparisons(size)
        with column:
            st.subheader(name)
            st.metric("Search result", f"Found at index {result.index}" if result.found else "Not found")
            st.metric("Comparisons", f"{result.comparisons:,}", help=f"Worst case for n = {size:,}: {bound:,}")
            st.metric("Execution time", format_seconds(measurement.time_sec),
                      help=f"Median of {measurement.runs} timed runs")
            st.metric("Memory usage", format_kb(measurement.memory_kb), help=measurement.memory_method)
        rows.append({
            "Algorithm": name,
            "Result": f"index {result.index}" if result.found else "not found",
            "Comparisons": result.comparisons,
            "Worst-case comparisons": bound,
            "Theoretical time": THEORETICAL[name]["worst"],
            "Execution time": format_seconds(measurement.time_sec),
            "Peak memory": format_kb(measurement.memory_kb),
            "Timed runs": measurement.runs,
            "_time": measurement.time_sec,
        })

    table = pd.DataFrame(rows)
    st.subheader("Comparison")
    st.dataframe(table.drop(columns="_time"), hide_index=True)
    if len(rows) == 2:
        left, right = st.columns(2)
        with left:
            show_fig(plot_bar_comparison(table["Algorithm"].tolist(), table["Comparisons"].tolist(),
                                         ylabel="Key comparisons", title=f"Comparisons for n = {size:,}",
                                         log_y=True, formatter=lambda v: f"{v:,.0f}"))
        with right:
            show_fig(plot_bar_comparison(table["Algorithm"].tolist(), (table["_time"] * 1000).tolist(),
                                         ylabel="Execution time (ms)", title=f"Execution time for n = {size:,}",
                                         log_y=True, formatter=lambda v: format_seconds(v / 1000)))
    with st.expander("Reading these numbers"):
        st.markdown(
            f"- Linear search examines elements one by one, so an absent key costs exactly n = {size:,} "
            f"comparisons. A present key costs its position + 1.\n"
            f"- Binary search halves the interval each probe, so it never needs more than "
            f"floor(log2 n) + 1 = {max_binary_comparisons(size):,} probes for n = {size:,}.\n"
            "- Binary search only works because the dataset is sorted. Sorting costs O(n log n) once, "
            "which pays off when the same data is searched many times.\n"
            "- Both use O(1) extra memory; what tracemalloc reports is the loop counters and the result object."
        )


# ---------------------------------------------------------------------------
# Page: Code Benchmarking (Task 3)
# ---------------------------------------------------------------------------
def page_code_benchmark() -> None:
    st.title("Code Benchmarking")
    st.write("A development team wants to compare the cost of program structures it uses every day. "
             "Pick snippets and an input size, run them, and compare time and memory side by side.")

    c1, c2, c3, c4 = st.columns([1.6, 1, 1, 1.4])
    with c1:
        snippets = st.multiselect("Code snippets", list(CODE_SNIPPETS), default=list(CODE_SNIPPETS))
    with c2:
        loop_n = size_selector("n for loop snippets (iterations)", [100, 1_000, 10_000, 50_000], 1_000,
                               key="loop_n")
    with c3:
        fact_n = size_selector("n for factorial snippets", [100, 500, 1_000], 1_000, key="fact_n",
                               custom_max=20_000)
    with c4:
        backend = MEMORY_BACKENDS[st.selectbox("Memory measurement", list(MEMORY_BACKENDS), key="code_backend")]

    if "Nested Loop Traversal" in snippets:
        estimate = estimate_nested_loop_seconds(loop_n)
        if estimate > 5:
            st.warning(f"Nested loop at n = {loop_n:,} performs {loop_n * loop_n:,} inner iterations, "
                       f"about {format_seconds(estimate)} per run on this machine. The page stays busy until it finishes.")
    if "Recursive Factorial" in snippets and fact_n > 1000:
        st.info("Recursion deeper than Python's default limit of 1000 frames: the tool raises "
                "sys.setrecursionlimit for the run, which is itself a sign of the O(n) stack usage.")

    run = st.button("Run benchmark", type="primary")
    autorun = st.query_params.get("autorun") == "1" and not st.session_state.get("code_autorun_done")
    if (run or autorun) and snippets:
        st.session_state["code_autorun_done"] = True
        engine = BenchmarkEngine(memory_backend=backend)
        progress = st.progress(0.0, text="Starting")
        rows = []
        for index, name in enumerate(snippets):
            n = fact_n if SNIPPET_GROUP[name] == "factorial" else loop_n
            progress.progress(index / len(snippets), text=f"Running {name} (n = {n:,})")
            measurement = engine.measure(CODE_SNIPPETS[name], (n,), name=name,
                                         group=SNIPPET_GROUP[name], input_size=n)
            rows.append(measurement.to_row())
        progress.empty()
        st.session_state["code_results"] = pd.DataFrame(rows)
        st.session_state["code_results_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elif run and not snippets:
        st.warning("Choose at least one snippet.")

    df = st.session_state.get("code_results")
    if df is None:
        st.info("Choose the snippets and press Run benchmark.")
    else:
        st.caption(f"Last run: {st.session_state.get('code_results_time', '')}")
        st.subheader("Comparison table")
        table = pd.DataFrame({
            "Snippet": df["algorithm"],
            "n": df["input_size"].map(lambda v: f"{int(v):,}"),
            "Theoretical time": df["algorithm"].map(lambda a: THEORETICAL[a]["worst"]),
            "Theoretical space": df["algorithm"].map(lambda a: THEORETICAL[a]["space"]),
            "Operations": df["operations"].map(lambda v: f"{int(v):,}"),
            "Execution time": df["time_sec"].map(format_seconds),
            "Peak memory": df["memory_kb"].map(format_kb),
            "Timed runs": df["runs"],
            "Result": df["result_summary"],
        })
        st.dataframe(table, hide_index=True)
        left, right = st.columns(2)
        with left:
            st.subheader("Execution time report")
            show_fig(plot_bar_comparison(df["algorithm"].tolist(), (df["time_sec"] * 1000).tolist(),
                                         ylabel="Execution time (ms)", title="Median execution time",
                                         log_y=True, formatter=lambda v: format_seconds(v / 1000)))
        with right:
            st.subheader("Memory consumption report")
            mem = df["memory_kb"].fillna(0).tolist()
            show_fig(plot_bar_comparison(df["algorithm"].tolist(), mem, ylabel="Peak memory (KB)",
                                         title="Peak memory during the run", formatter=format_kb))
            if df["memory_kb"].isna().any():
                st.caption("Bars at 0 mark runs whose memory pass was skipped; see the Memory method column.")
            st.caption("Measured inside the web app, so a long traced run also counts what the server's "
                       "background threads allocate meanwhile. The command-line run in results/ is the reference.")

    with st.expander("Source code of the snippets"):
        for name in snippets or list(CODE_SNIPPETS):
            st.markdown(f"**{name}**  ({THEORETICAL[name]['worst']} time, {THEORETICAL[name]['space']} space)")
            st.code(get_source(name), language="python")


# ---------------------------------------------------------------------------
# Page: Automated Benchmarking (Task 4)
# ---------------------------------------------------------------------------
def page_automated() -> None:
    st.title("Automated Benchmarking")
    st.write("Runs every algorithm and snippet across the input sizes from the assignment and records "
             "execution time, memory consumption and the number of operations for each run.")

    if get_results() is not None and str(st.session_state.get("results_source", "")).startswith("Loaded"):
        st.info(f"Showing the saved results from results/{RESULTS_CSV.name}. "
                f"{st.session_state['results_source']} Run a new benchmark to replace them.")

    st.subheader("Input sizes")
    c1, c2, c3 = st.columns(3)
    with c1:
        use_search = st.checkbox("Search algorithms", True)
        search_text = st.text_input("Dataset sizes", ", ".join(str(s) for s in DEFAULT_INPUT_SIZES["search"]))
    with c2:
        use_factorial = st.checkbox("Factorial programs", True)
        factorial_text = st.text_input("Values of n", ", ".join(str(s) for s in DEFAULT_INPUT_SIZES["factorial"]))
    with c3:
        use_loops = st.checkbox("Loop-based programs", True)
        loops_text = st.text_input("Iteration counts", ", ".join(str(s) for s in DEFAULT_INPUT_SIZES["loops"]))

    st.subheader("Options")
    o1, o2, o3 = st.columns(3)
    measure_memory = o1.checkbox("Measure memory in a separate tracemalloc run", True)
    memory_budget = o2.slider("Skip the memory pass if tracing is predicted to exceed (seconds)", 5, 300, 30)
    save = o3.checkbox("Save results to results/benchmark_results.csv", True)

    try:
        sizes = {"search": parse_sizes(search_text), "factorial": parse_sizes(factorial_text),
                 "loops": parse_sizes(loops_text)}
    except ValueError:
        st.error("Input sizes must be comma-separated whole numbers.")
        return
    groups = [g for g, on in (("search", use_search), ("factorial", use_factorial), ("loops", use_loops)) if on]

    if use_loops and sizes["loops"]:
        biggest = max(sizes["loops"])
        estimate = estimate_nested_loop_seconds(biggest)
        if estimate > 10:
            st.warning(f"The nested loop at n = {biggest:,} alone takes about {format_seconds(estimate)} "
                       "on this machine. The page stays busy until the whole run finishes.")

    if st.button("Run automated benchmark", type="primary", disabled=not groups):
        progress = st.progress(0.0, text="Starting")

        def on_progress(done: int, total: int, label: str) -> None:
            progress.progress(done / total if total else 1.0, text=f"{label}  ({done}/{total})")

        engine = BenchmarkEngine(measure_memory=measure_memory, memory_time_budget=memory_budget)
        started = time.perf_counter()
        df = run_automated_suite(sizes, engine=engine, groups=groups, on_progress=on_progress)
        elapsed = time.perf_counter() - started
        progress.empty()
        st.session_state["results"] = df
        st.session_state["results_source"] = (f"Run in this session at "
                                              f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
                                              f"in {format_seconds(elapsed)}.")
        if save:
            engine.save_csv(RESULTS_CSV)
            save_environment()
            st.success(f"Saved {len(df)} rows to {RESULTS_CSV}")

    df = st.session_state.get("results")
    if df is None:
        return
    st.caption(st.session_state.get("results_source", ""))
    st.subheader("Recorded measurements")
    st.dataframe(pretty_results(df), hide_index=True)

    st.subheader("Pivot views (algorithm x input size)")
    t1, t2, t3 = st.tabs(["Execution time (ms)", "Peak memory (KB)", "Operations"])
    with t1:
        st.dataframe(df.pivot_table(index="algorithm", columns="input_size", values="time_ms", aggfunc="first").round(4))
    with t2:
        st.dataframe(df.pivot_table(index="algorithm", columns="input_size", values="memory_kb", aggfunc="first").round(2))
    with t3:
        st.dataframe(df.pivot_table(index="algorithm", columns="input_size", values="operations", aggfunc="first"))
    st.download_button("Download results as CSV", df.to_csv(index=False).encode("utf-8"),
                       "benchmark_results.csv", "text/csv")


# ---------------------------------------------------------------------------
# Page: Visualization (Task 5)
# ---------------------------------------------------------------------------
def page_visualization() -> None:
    st.title("Performance Visualization")
    df = get_results()
    if df is None:
        results_notice()
        return
    st.caption(st.session_state.get("results_source", ""))
    log_y = st.toggle("Logarithmic time axis (shows both lines when one is thousands of times faster)")

    tabs = st.tabs(["Search algorithms", "Factorial programs", "Loop-based programs", "Overview"])
    for tab, group in zip(tabs[:3], ["search", "factorial", "loops"]):
        with tab:
            left, right = st.columns(2)
            with left:
                show_fig(plot_time(df, group, log_y=log_y))
            with right:
                show_fig(plot_memory(df, group))
            if group == "search":
                show_fig(plot_operations(df, "search"))
            st.markdown("**What the measurements show**")
            for line in trend_lines_for(df, group):
                st.markdown(f"- {line}")
    with tabs[3]:
        show_fig(plot_overview(df))
        show_fig(plot_growth_exponents(df))
        st.markdown("On log-log axes a power law is a straight line whose slope is the exponent: "
                    "slope 1 is linear growth, slope 2 quadratic, slope near 0 constant or logarithmic.")

    if st.button("Save all charts to graphs/"):
        paths = generate_all_charts(df)
        st.success(f"Saved {len(paths)} charts to {GRAPHS_DIR}")
        st.code("\n".join(p.name for p in paths.values()))


# ---------------------------------------------------------------------------
# Page: Complexity Analysis (Task 6)
# ---------------------------------------------------------------------------
def page_complexity() -> None:
    st.title("Complexity Analysis")
    st.subheader("Theoretical complexity")
    st.dataframe(theoretical_table(), hide_index=True)

    df = get_results()
    if df is None:
        results_notice()
        return
    st.caption(st.session_state.get("results_source", ""))

    st.subheader("Theoretical vs observed performance")
    table = build_complexity_table(df)
    shown = pd.DataFrame({
        "Algorithm / Program": table["Algorithm / Program"],
        "Theoretical time": table["Theoretical Time"],
        "Theoretical space": table["Theoretical Space"],
        "Time at smallest n": [f"{format_seconds(t)} (n={n:,})" for t, n in zip(table["Time at n (min)"], table["n (min)"])],
        "Time at largest n": [f"{format_seconds(t)} (n={n:,})" for t, n in zip(table["Time at n (max)"], table["n (max)"])],
        "Time growth": table["Time growth (max/min)"].map(lambda v: f"{v:,.1f}x"),
        "Peak memory (min to max)": [f"{format_kb(a)} to {format_kb(b)}" for a, b in
                                     zip(table["Peak memory min (KB)"], table["Peak memory max (KB)"])],
        "Observed exponent": table["Observed exponent"].map(lambda v: f"{v:.2f}"),
        "Observed growth trend": table["Observed growth trend"],
        "Verdict": table["Verdict"],
    })
    st.dataframe(shown, hide_index=True)

    st.subheader("Time relative to the smallest input")
    st.dataframe(growth_factor_table(df).round(1))
    show_fig(plot_growth_exponents(df))

    st.subheader("Reading the table")
    st.markdown(
        "- The observed exponent is the slope of log(time) against log(n). It is fitted from the "
        "measured points, so with only three or four sizes it is an estimate, not a proof.\n"
        "- Binary search runs in a few microseconds at every size. At that scale the fixed cost of a "
        "function call and the timer itself is a large share of the measurement, which is why its "
        "line is nearly flat rather than a clean logarithm.\n"
        "- Both factorial versions grow faster than O(n) because Python integers have arbitrary "
        "precision: 1000! has 2,568 digits, and multiplying a number that long is not a constant-time "
        "step. The O(n) bound counts multiplications, not the digits they touch.\n"
        "- tracemalloc sees Python heap allocations. The recursive factorial shows O(n) growth because "
        "every open call keeps its own integer argument alive, one per stack frame.\n"
        "- The nested loop's memory pass is skipped at large n because tracing every allocation would "
        "take hours; its O(1) memory is verified at the smaller sizes where the trace is affordable."
    )


# ---------------------------------------------------------------------------
# Page: Comparative Study (Task 7)
# ---------------------------------------------------------------------------
def _at(df: pd.DataFrame, algorithm: str, column: str, size: Optional[int] = None):
    sub = df[df["algorithm"] == algorithm].sort_values("input_size")
    if sub.empty:
        return None
    row = sub.iloc[-1] if size is None else sub[sub["input_size"] == size].iloc[0]
    return row[column]


def pretty_speedup(df: pd.DataFrame, faster: str, slower: str) -> pd.DataFrame:
    table = pairwise_speedup(df, faster, slower)
    return pd.DataFrame({
        "Input size (n)": table["input_size"].map(lambda v: f"{int(v):,}"),
        faster: table[f"{faster} (s)"].map(format_seconds),
        slower: table[f"{slower} (s)"].map(format_seconds),
        "Speed-up": table["speedup (x)"].map(lambda v: f"{v:,.1f}x"),
    })


def page_comparative() -> None:
    st.title("Comparative Performance Study")
    df = get_results()
    if df is None:
        results_notice()
        return
    st.caption(st.session_state.get("results_source", ""))

    # Search algorithms
    st.subheader("Linear Search vs Binary Search")
    left, right = st.columns([1, 1])
    with left:
        st.dataframe(pretty_speedup(df, "Binary Search", "Linear Search"), hide_index=True)
    with right:
        for line in trend_lines_for(df, "search"):
            st.markdown(f"- {line}")
    n_max = _at(df, "Linear Search", "input_size")
    if n_max is not None:
        lin_ops = _at(df, "Linear Search", "operations")
        bin_ops = _at(df, "Binary Search", "operations")
        st.markdown(
            f"Binary search wins at every size, and the gap widens with n. At n = {int(n_max):,} a missing "
            f"key costs linear search {int(lin_ops):,} comparisons but binary search only {int(bin_ops):,}. "
            "Doubling the dataset doubles the linear search's work and adds a single probe to the binary "
            "search's. The price is the precondition: binary search needs sorted data, so for a catalogue "
            "that is searched once and thrown away the O(n log n) sort costs more than the linear scan "
            "it replaces. Both use constant extra memory."
        )

    # Factorial programs
    st.subheader("Recursive Factorial vs Iterative Factorial")
    left, right = st.columns([1, 1])
    with left:
        st.dataframe(pretty_speedup(df, "Iterative Factorial", "Recursive Factorial"), hide_index=True)
    with right:
        for line in trend_lines_for(df, "factorial"):
            st.markdown(f"- {line}")
    rec_mem = _at(df, "Recursive Factorial", "memory_kb")
    it_mem = _at(df, "Iterative Factorial", "memory_kb")
    fact_n = _at(df, "Recursive Factorial", "input_size")
    if fact_n is not None:
        st.markdown(
            f"Both versions do the same n - 1 multiplications, so their times stay within a small factor of "
            f"each other; the recursive version pays extra for a function call per level. Memory is where "
            f"they part: at n = {int(fact_n):,} the recursive version peaked at {format_kb(rec_mem)} against "
            f"{format_kb(it_mem)} for the loop, because each of the {int(fact_n):,} open frames holds its own "
            "argument. Python also refuses to recurse past 1,000 frames unless the limit is raised, and "
            "raising it does not remove the cost, it only defers the failure. Iteration is the safer choice "
            "for anything that may grow."
        )

    # Loop-based programs
    st.subheader("Single Loop vs Nested Loop Traversal")
    left, right = st.columns([1, 1])
    with left:
        st.dataframe(pretty_speedup(df, "Single Loop Traversal", "Nested Loop Traversal"), hide_index=True)
    with right:
        for line in trend_lines_for(df, "loops"):
            st.markdown(f"- {line}")
    loop_n = _at(df, "Nested Loop Traversal", "input_size")
    if loop_n is not None:
        nested_t = _at(df, "Nested Loop Traversal", "time_sec")
        single_t = _at(df, "Single Loop Traversal", "time_sec")
        st.markdown(
            f"The single loop does n steps, the nested loop n squared. At n = {int(loop_n):,} that is "
            f"{int(loop_n) ** 2:,} inner iterations: {format_seconds(nested_t)} against "
            f"{format_seconds(single_t)} for the single pass. Multiplying n by 10 multiplies the nested "
            "loop's time by about 100, which is what makes an O(n squared) routine unusable on inputs "
            "that an O(n) routine handles without effort. Neither uses memory beyond a few counters."
        )

    st.subheader("Time-space trade-offs")
    st.markdown(
        "- Binary search trades preparation time (sorting) for a much cheaper query. Once the data is "
        "sorted, every later search benefits.\n"
        "- The recursive factorial trades memory (one frame per level) for code that mirrors the "
        "mathematical definition. The iterative version uses one accumulator and cannot overflow the stack.\n"
        "- Nested loops are not a trade-off at all in this study: they cost quadratic time and still "
        "use constant memory, so they only make sense when every pair genuinely has to be visited.\n"
        "- Measurement itself has a cost. Tracing memory slowed the loops 25x to 60x, so the tool measures "
        "time and memory in separate runs and skips traces that would take hours."
    )

    st.subheader("Key findings")
    lines = summarize_trends(df)
    for line in [l for l in lines if l.startswith("At n =")]:
        st.markdown(f"- {line}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    page = sidebar_navigation()
    {
        "Overview": page_overview,
        "Search Analysis": page_search,
        "Code Benchmarking": page_code_benchmark,
        "Automated Benchmarking": page_automated,
        "Visualization": page_visualization,
        "Complexity Analysis": page_complexity,
        "Comparative Study": page_comparative,
    }[page]()


if __name__ == "__main__":
    main()
