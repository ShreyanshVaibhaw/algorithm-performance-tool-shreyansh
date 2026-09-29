"""
code_benchmark.py - Code Benchmarking Module (Task 3)

Four predefined code snippets that stand in for program structures found in
everyday application code:

    single_loop          one pass over n items                O(n)   time, O(1) space
    nested_loop          every (i, j) pair of n items         O(n^2) time, O(1) space
    factorial_recursive  n! via  n * (n-1)!                   O(n)   time, O(n) stack
    factorial_iterative  n! via a running product             O(n)   time, O(1) space

Each snippet returns a SnippetResult holding the computed value and the number
of basic operations performed, so the engine can relate work done to time
taken.  For these snippets the operation count follows exactly from the loop
structure, so it is derived from n rather than counted with an extra variable.
Counting inside the loop would slow the timed code and no longer be the
textbook version being studied.
"""
from __future__ import annotations

import inspect
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict


@dataclass
class SnippetResult:
    """Outcome of running one code snippet."""

    snippet: str
    n: int
    value: Any
    operations: int     # loop iterations / recursive calls / multiplications

    def value_summary(self, max_digits: int = 12) -> str:
        """Human-readable value; huge integers are shown by digit count."""
        text = str(self.value)
        if len(text) <= max_digits:
            return text
        return f"{len(text)}-digit integer ({text[:max_digits]}...)"


# ---------------------------------------------------------------------------
# Loop-based programs
# ---------------------------------------------------------------------------
def single_loop(n: int) -> SnippetResult:
    """Sum 0 .. n-1 with one loop: n iterations."""
    total = 0
    for i in range(n):
        total += i
    return SnippetResult("Single Loop Traversal", n, total, n)


def nested_loop(n: int) -> SnippetResult:
    """Visit every (i, j) pair with a loop inside a loop: n*n iterations."""
    total = 0
    for i in range(n):
        for j in range(n):
            total += 1
    return SnippetResult("Nested Loop Traversal", n, total, total)


# ---------------------------------------------------------------------------
# Factorial programs
# ---------------------------------------------------------------------------
def _ensure_recursion_limit(depth: int) -> None:
    """Raise the interpreter's recursion limit when `depth` frames are needed.

    Python's default limit is 1000, so the assignment's n = 1000 case would
    otherwise raise RecursionError.  Needing to do this at all is a practical
    sign of the recursive version's O(n) stack usage.
    """
    needed = depth + 100
    if sys.getrecursionlimit() < needed:
        sys.setrecursionlimit(needed)


def _factorial_recursive(k: int) -> int:
    if k <= 1:
        return 1
    return k * _factorial_recursive(k - 1)


def factorial_recursive(n: int) -> SnippetResult:
    """n! from the recurrence n! = n * (n-1)!  (n recursive calls, n stack frames)."""
    if n < 0:
        raise ValueError("factorial is undefined for negative numbers")
    _ensure_recursion_limit(n)
    value = _factorial_recursive(n)
    return SnippetResult("Recursive Factorial", n, value, max(n, 1))


def factorial_iterative(n: int) -> SnippetResult:
    """n! from a running product (n-1 multiplications, one accumulator)."""
    if n < 0:
        raise ValueError("factorial is undefined for negative numbers")
    result = 1
    for i in range(2, n + 1):
        result *= i
    return SnippetResult("Iterative Factorial", n, result, max(n - 1, 0))


# ---------------------------------------------------------------------------
# Registry used by the engine and the Streamlit app
# ---------------------------------------------------------------------------
CODE_SNIPPETS: Dict[str, Callable[[int], SnippetResult]] = {
    "Single Loop Traversal": single_loop,
    "Nested Loop Traversal": nested_loop,
    "Recursive Factorial": factorial_recursive,
    "Iterative Factorial": factorial_iterative,
}

SNIPPET_GROUP = {
    "Single Loop Traversal": "loops",
    "Nested Loop Traversal": "loops",
    "Recursive Factorial": "factorial",
    "Iterative Factorial": "factorial",
}


def expected_operations(name: str, n: int) -> int:
    """Operation count implied by the snippet's structure, without running it."""
    if name == "Single Loop Traversal":
        return n
    if name == "Nested Loop Traversal":
        return n * n
    if name == "Recursive Factorial":
        return max(n, 1)
    if name == "Iterative Factorial":
        return max(n - 1, 0)
    raise KeyError(name)


def get_source(name: str) -> str:
    """Source code of a snippet, for display in the app."""
    func = CODE_SNIPPETS[name]
    source = inspect.getsource(func)
    if name == "Recursive Factorial":
        source = inspect.getsource(_factorial_recursive) + "\n\n" + source
    return source


if __name__ == "__main__":
    for name, func in CODE_SNIPPETS.items():
        n = 1000 if "Factorial" in name else 2000
        res = func(n)
        print(f"{name:24s} n={n:5d}  operations={res.operations:>8}  value={res.value_summary()}")
