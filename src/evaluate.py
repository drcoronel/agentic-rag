"""Retrieval quality against a .jsonl of expected files."""
import json, sys
from pathlib import Path

from .agent import search_hits

def _found_rel(rel, want):
    return rel == want or rel.endswith("/" + want)


def _found(hit, want):
    return _found_rel(hit.payload["rel"], want)


def evaluate(ctx, path, k=5, modes=("dense", "sparse", "hybrid"), misses=False):
    """Rank of the expected file for each question in a .jsonl of {q, path, kind, src}.

    The point is the comparison between modes, not the absolute numbers. `src` splits
    hand-written questions from model-generated ones: a question drafted *from* the
    answer text reuses its vocabulary, so the synthetic column reads high and is only
    comparable with itself.
    """
    cases = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    if not cases:
        sys.exit(f"no cases in {path}")
    # (field, values) — one block of columns each, so a skew shows up as a split column
    cols = [(f, sorted({c.get(f, "all") for c in cases})) for f in ("kind", "src")]
    cols = [(f, vs) for f, vs in cols if len(vs) > 1]
    rate = lambda rs, n: sum(1 for r in rs if 0 < r <= n) / len(rs) if rs else 0.0

    print(f"{len(cases)} questions, k={k}  (r@n = expected file in top n, per-group is r@{k})\n")
    print(f"{'mode':<8}{'r@1':>7}{f'r@{k}':>7}{'mrr':>7}   "
          + "".join(f"{v:>12}" for _, vs in cols for v in vs))
    for mode in modes:
        scored = [(c, next((i for i, (h, _) in enumerate(
            search_hits(ctx, c["q"], k, c.get("collections"), mode), 1)
            if _found(h, c["path"])), 0)) for c in cases]
        rs = [r for _, r in scored]
        per = "".join(f"{rate([r for c, r in scored if c.get(f, 'all') == v], k):>12.2f}"
                      for f, vs in cols for v in vs)
        print(f"{mode:<8}{rate(rs, 1):>7.2f}{rate(rs, k):>7.2f}"
              f"{sum(1 / r for r in rs if r) / len(rs):>7.2f}   {per}")
        if misses:
            for c, r in scored:
                if not r:
                    print(f"    miss [{mode}] {c['q'][:70]!r} -> {c['path']}", file=sys.stderr)
