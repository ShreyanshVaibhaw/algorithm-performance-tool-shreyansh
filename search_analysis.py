"""
search_analysis.py - Search Algorithm Analysis Module (Task 2)

Instrumented Linear Search and Binary Search for the digital-library scenario.
Both functions return a SearchResult that records where the key was found and
how many key comparisons were needed.  Execution time and memory usage are
measured by benchmark_engine.py so the search code itself stays textbook-simple.

Counting convention
-------------------
* Linear search: one comparison per element examined.
* Binary search: one comparison per probe of the middle element.  A probe
  needs an equality test and an ordering test in Python, but algorithmically
  it is one three-way comparison, which is the usual textbook convention.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence


@dataclass
class SearchResult:
    """Outcome of one search call."""

    algorithm: str
    target: int
    index: int            # position of the key, or -1 when absent
    comparisons: int      # key comparisons performed
    dataset_size: int

    @property
    def found(self) -> bool:
        return self.index != -1

    @property
    def operations(self) -> int:
        """Generic name read by the benchmark engine."""
        return self.comparisons

    def describe(self) -> str:
        where = f"found at index {self.index}" if self.found else "not found"
        return f"{self.algorithm}: key {self.target} {where} after {self.comparisons} comparisons"


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------
def generate_dataset(size: int, *, seed: Optional[int] = None,
                     sorted_output: bool = True) -> List[int]:
    """Return `size` distinct non-negative integers, sorted unless told otherwise.

    Values are sampled from 0 .. 10*size so keys are spread out and it is easy
    to construct a key that is guaranteed to be absent.  Binary search needs the
    sorted form; linear search works on either.
    """
    if size < 0:
        raise ValueError("size must be non-negative")
    rng = random.Random(seed)
    data = rng.sample(range(size * 10 + 1), size)
    if sorted_output:
        data.sort()
    return data


def choose_key(data: Sequence[int], *, present: bool = True,
               seed: Optional[int] = None) -> int:
    """Pick a search key.

    present=True  -> a random element of `data` (average case).
    present=False -> -1, which can never occur in a generated dataset, so the
                     search runs to completion (worst case for linear search).
    """
    if present and len(data) > 0:
        return random.Random(seed).choice(list(data))
    return -1


def is_sorted(data: Sequence[int]) -> bool:
    return all(data[i] <= data[i + 1] for i in range(len(data) - 1))


# ---------------------------------------------------------------------------
# Search algorithms
# ---------------------------------------------------------------------------
def linear_search(data: Sequence[int], target: int) -> SearchResult:
    """Scan the sequence from left to right until the key appears.

    Works on unsorted data.  Best case O(1), average and worst case O(n) time;
    O(1) extra space.
    """
    comparisons = 0
    for index, value in enumerate(data):
        comparisons += 1
        if value == target:
            return SearchResult("Linear Search", target, index, comparisons, len(data))
    return SearchResult("Linear Search", target, -1, comparisons, len(data))


def binary_search(data: Sequence[int], target: int) -> SearchResult:
    """Halve a SORTED sequence around the middle element until the key appears.

    Best case O(1), average and worst case O(log n) time; O(1) extra space for
    this iterative version.  The caller must supply sorted data.
    """
    low, high = 0, len(data) - 1
    comparisons = 0
    while low <= high:
        mid = (low + high) // 2
        comparisons += 1
        if data[mid] == target:
            return SearchResult("Binary Search", target, mid, comparisons, len(data))
        if data[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return SearchResult("Binary Search", target, -1, comparisons, len(data))


SEARCH_ALGORITHMS: Dict[str, Callable[[Sequence[int], int], SearchResult]] = {
    "Linear Search": linear_search,
    "Binary Search": binary_search,
}


def max_binary_comparisons(n: int) -> int:
    """Upper bound on probes for a size-n array: floor(log2 n) + 1."""
    return n.bit_length() if n > 0 else 0


if __name__ == "__main__":
    data = generate_dataset(20, seed=1)
    print("Dataset:", data)
    key = choose_key(data, present=True, seed=1)
    print(linear_search(data, key).describe())
    print(binary_search(data, key).describe())
    print(linear_search(data, -1).describe())
    print(binary_search(data, -1).describe())
