#!/usr/bin/env python
"""
regex_mt_bench.py – does Python's re.search() scale across threads?

• Creates a single master regex (similar to the RegexValidator patch).
• Launches N threads; each thread calls .search() a fixed number of times.
• Reports wall-time vs process-CPU-time so you can see how many cores
  the regex engine actually used.
"""

from __future__ import annotations

import regex as re, threading, time, os, sys, random, math

# ---------------------------------------------------------------------
# 1.  Prepare synthetic workload
# ---------------------------------------------------------------------
NUM_PATTERNS   = 120             # similar to your real list
ITERATIONS     = 80_000           # 16× more work than before
TEXT_LEN_CHARS = 800_000          # force the engine to read a lot

random.seed(42)

# Make deterministic-ish patterns: a literal word or a short .* wildcard
_PATTERNS: list[str] = []
for i in range(NUM_PATTERNS):
    if i % 3 == 0:
        _PATTERNS.append(fr"\bword{i}\b")
    elif i % 3 == 1:
        _PATTERNS.append(fr"phrase{i}[^ ]+end")
    else:
        _PATTERNS.append(fr"token{i}.*?token{i+1}")

# Build a master alternation with named groups (as in the patch)
parts, _group2raw = [], {}
for i, p in enumerate(_PATTERNS):
    gname = f"P{i}"
    parts.append(f"(?P<{gname}>{p})")
    _group2raw[gname] = p
_BIG_RE = re.compile("|".join(parts), re.IGNORECASE | re.MULTILINE | re.DOTALL)

# Generate text that *sometimes* matches: sprinkle keywords every ~1000 chars
_chunks = []
for i in range(TEXT_LEN_CHARS // 50):
    if i % 20 == 0:                        # every 20th chunk drop a keyword
        k = random.randrange(NUM_PATTERNS)
        tok = f"word{k}" if k % 3 == 0 else f"phrase{k}xxend"
        _chunks.append(tok)
    else:
        _chunks.append("loremipsum")
_TEXT = " ".join(_chunks)

# ---------------------------------------------------------------------
# 2.  Benchmark helper
# ---------------------------------------------------------------------
def run_threads(n_threads: int) -> tuple[float, float]:
    """
    Launch n_threads that each call _BIG_RE.search(_TEXT) ITERATIONS times.

    Returns (wall_seconds, cpu_seconds) for the whole job.
    """
    def worker():
        s = _BIG_RE  # local var for speed
        t = _TEXT
        for _ in range(ITERATIONS):
            s.search(t)

    threads = [threading.Thread(target=worker, daemon=True)
               for _ in range(n_threads)]

    cpu_start = os.times()          # returns a 5-tuple
    t0 = time.perf_counter()

    for th in threads:
        th.start()
    for th in threads:
        th.join()

    wall = time.perf_counter() - t0
    cpu  = (os.times().user + os.times().system) - (cpu_start.user + cpu_start.system)
    return wall, cpu

# ---------------------------------------------------------------------
# 3.  Run for several thread counts
# ---------------------------------------------------------------------
def main():
    print(f"patterns            : {NUM_PATTERNS}")
    print(f"text length (chars) : {len(_TEXT):,}")
    print(f"regex searches/thread: {ITERATIONS}")
    print()

    for n in (1, 2, 4, 8):
        wall, cpu = run_threads(n)
        util = cpu / wall if wall else math.nan
        print(f"{n:>2} threads  →  wall {wall:6.2f} s   "
              f"CPU {cpu:6.2f} s   ratio {util:4.2f}")

    print("\nInterpretation:")
    print("  • ratio ≈ 1.0  → work is effectively single-core.")
    print("  • ratio → N    → regex scanning scales across N cores.")

if __name__ == "__main__":
    main()
