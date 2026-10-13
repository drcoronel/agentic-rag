"""Retrieval: hybrid search, the tool surface, and the local-model answer loop."""
import json, os, subprocess, sys
from pathlib import Path

from qdrant_client.http.exceptions import UnexpectedResponse

from .config import QUERY_PREFIX, SKIP_DIRS
from .llm import chat, embed
from .qdrant import client
from .state import scope
from .text import sparse

RRF_K = 60
# BM25 contributes at a fraction of dense's weight. Measured, not guessed: at equal weight
# it drags r@1 from 0.61 to 0.42 (`rag.py eval`), because a 20-word question matches a lot
# of mid-IDF prose. At 0.2 it adds recall on exact identifiers without that cost.
SPARSE_W = 0.2
# Hits past `k` come back as one-line digests. Measured on `eval.jsonl`: the expected file
# is in the top 8 for 0.85 of questions but in the top 50 for 0.96 — the answer is almost
# always retrieved, the caller just never sees it. Digests are ~90 chars each, so showing
# rank 9-50 costs ~4k characters and lets the calling agent do its own judging. A local
# model doing that judging instead measurably loses recall; see README.
WIDE = 42
FULL_CHARS = 12_000      # character budget for full chunk text; the rest digest


def _ranked(names, q, using, depth, errs=None):
    """One modality across every collection, merged into a single ranking.

    A collection that is not built yet 404s and is skipped silently. Anything else —
    Qdrant down, a transport error — is appended to `errs`, because swallowing it makes
    an outage read to the caller as a confident "no results".
    """
    hits = []
    for n in names:
        try:
            hits += [(h, n) for h in client().query_points(
                n, query=q, using=using, limit=depth, with_payload=True).points]
        except UnexpectedResponse as e:
            if e.status_code != 404 and errs is not None:
                errs.append((n, f"HTTP {e.status_code}"))
        except Exception as e:                          # transport, timeout, auth
            if errs is not None:
                errs.append((n, type(e).__name__))
    hits.sort(key=lambda x: -x[0].score)
    return hits


def search_hits(ctx, query, k=8, collections=None, mode="hybrid", project=None, errs=None):
    """[(hit, collection)] best first. mode: hybrid | dense | sparse (dense/sparse for eval).

    Fusion is deliberately done here and not by Qdrant: a per-collection FusionQuery hands
    back rank-derived scores, so every collection's best hit ties at 1/(RRF_K+1) and merging
    nine of those is a coin toss. Ranking globally per modality first keeps the comparison.
    """
    names = scope(ctx, collections, project)
    depth = max(int(k) * 4, 32)      # candidates per collection; 20 measurably loses recall
    lists = []
    if mode != "sparse":
        lists.append(_ranked(names, embed([query], QUERY_PREFIX)[0], "dense", depth, errs))
    if mode != "dense":
        lists.append(_ranked(names, sparse(query, query=True), "bm25", depth, errs))
    if len(lists) == 1:
        return lists[0][:int(k)]

    fused = {}
    for w, lst in zip((1.0, SPARSE_W), lists):
        for rank, (h, n) in enumerate(lst, 1):
            prev, _, _ = fused.get((n, h.id), (0.0, h, n))
            fused[(n, h.id)] = (prev + w / (RRF_K + rank), h, n)
    out = sorted(fused.values(), key=lambda x: -x[0])[:int(k)]
    for score, h, _ in out:
        h.score = score                                 # so callers can print one scale
    return [(h, n) for _, h, n in out]


def _cite(h, n):
    return f"[{n}] {h.payload['rel']}:{h.payload['s']}-{h.payload['e']}"


def t_search(ctx, query, k=8, collections=None, project=None, wide=WIDE, budget=FULL_CHARS):
    """Full chunk text for the best hits, then every remaining hit as a one-line digest.

    Nothing is ever dropped: the budget decides full-text-vs-digest, not inclusion. The
    caller is a better relevance judge than anything local, so it gets the whole pool.
    """
    names = scope(ctx, collections, project)
    if not names:
        return f"unknown collection. available: {', '.join(scope(ctx, project=project))}"
    errs = []
    hits = search_hits(ctx, query, int(k) + wide, collections, project=project, errs=errs)
    warn = ""
    if (down := sorted({n for n, _ in errs})):
        why = sorted({e for _, e in errs})[0]
        warn = (f"warning: the index is unreachable ({why}) — is Qdrant up?\n\n"
                if len(down) == len(names) else
                f"warning: {len(down)} of {len(names)} collections unreachable "
                f"({', '.join(down[:3])}{', …' if len(down) > 3 else ''}) — results are "
                "partial, the answer may be in one of them\n\n")
    if not hits:
        return (warn + "no results").strip()

    full, spent = [], 0
    for h, n in hits[:int(k)]:
        text = h.payload["text"]
        if full and spent + len(text) > budget:
            break                                       # the rest fall through to digests
        spent += len(text)
        full.append(f"--- {_cite(h, n)} (score {h.score:.4f})\n{text}")

    rest = hits[len(full):]
    tail = ""
    if rest:
        tail = (f"\n\n--- {len(rest)} more candidates, path only (open the file, or call "
                "search again narrowed):\n"
                + "\n".join(f"{_cite(h, n)}  {' '.join(h.payload['text'].split())[:90]}"
                            for h, n in rest))
    return warn + "\n\n".join(full) + tail


def t_grep(ctx, pattern, collection=None):
    """Exact search straight off disk — always fresh, and finds identifiers embeddings miss."""
    where = [ctx["repos"][collection]] if collection in ctx["repos"] else ctx["roots"]
    rg = subprocess.run(["which", "rg"], capture_output=True).returncode == 0
    cmd = (["rg", "-n", "-i", "--max-count", "5"] + [f"--glob=!{d}" for d in SKIP_DIRS] if rg
           else ["grep", "-rnIi", "--max-count=5"] + [f"--exclude-dir={d}" for d in SKIP_DIRS])
    r = subprocess.run(cmd + ["-e", pattern] + where, capture_output=True, text=True, timeout=60)
    out = []
    for line in r.stdout.splitlines()[:40]:
        f, _, rest = line.partition(":")
        n, _, body = rest.partition(":")
        out.append(f"{_rel(ctx, f)}:{n}: {body.strip()[:200]}")
    return "\n".join(out) or "no matches"


def t_read_file(ctx, path, start=1, end=400):
    cands = [Path(path)] + [Path(p).parent / path for p in ctx["repos"].values()] \
            + [Path(r) / path for r in ctx["roots"]]
    f = next((c for c in cands if c.is_file()), None)
    if f is None:
        return f"not found: {path}"
    lines = f.read_text(errors="replace").splitlines()
    start, end = max(1, int(start)), min(len(lines), int(end))
    body = "\n".join(f"{i:>5}  {lines[i - 1]}" for i in range(start, end + 1))
    return f"{_rel(ctx, str(f.resolve()))} (lines {start}-{end} of {len(lines)})\n{body}"


def _rel(ctx, path):
    for d in ctx["repos"].values():
        if path.startswith(str(Path(d).parent) + os.sep):
            return os.path.relpath(path, Path(d).parent)
    return path


TOOLS = [
    {"type": "function", "function": {
        "name": "search", "description":
        "Semantic search. Pass `collections` to search only the repositories that can plausibly "
        "answer the question (see the catalogue in the system prompt); omit it to search all.",
        "parameters": {"type": "object", "required": ["query"], "properties": {
            "query": {"type": "string"},
            "collections": {"type": "array", "items": {"type": "string"},
                            "description": "collection names from the catalogue"},
            "k": {"type": "integer", "description": "results, default 8"}}}}},
    {"type": "function", "function": {
        "name": "grep", "description":
        "Exact case-insensitive regex search over the files. Use for identifiers, YAML keys, "
        "function names, config values — anything where the literal string matters.",
        "parameters": {"type": "object", "required": ["pattern"], "properties": {
            "pattern": {"type": "string"},
            "collection": {"type": "string", "description": "restrict to one repository"}}}}},
    {"type": "function", "function": {
        "name": "read_file", "description":
        "Read a repo-relative file, optionally a line range. Use after search/grep to see full "
        "context — search results are chunks, not whole files.",
        "parameters": {"type": "object", "required": ["path"], "properties": {
            "path": {"type": "string"}, "start": {"type": "integer"}, "end": {"type": "integer"}}}}},
]

SYSTEM = """You are a code and documentation assistant for a set of local repositories,
grouped into projects. Each repository is a separate search collection:

{catalogue}

You cannot see the files directly — use the tools. Strategy:
- Pick the collections that plausibly hold the answer and pass them to `search`. Only search
  everything when the question spans repositories or you genuinely cannot tell.
- `related:` lists the repositories that cite a collection or cover the same ground. If one
  collection half-answers, search its related collections next instead of everything.
- `search` for concepts and "how do I" questions, `grep` for exact names and keys.
- Follow up promising hits with `read_file` before answering.
- Call tools more than once if the first result is thin or points somewhere better.
Answer only from what the tools returned, and cite sources as `path:line`. If the
repositories do not contain the answer, say so plainly."""


def catalogue(ctx, project=None):
    related = ctx.get("graph", {}).get("related", {})
    lines, names = [], scope(ctx, project=project)
    for p in sorted({ctx["proj"][n] for n in names}):
        if len(ctx["projects"]) > 1:
            lines.append(f"\nproject `{p}`:")
        for name in [n for n in names if ctx["proj"][n] == p]:
            try:
                n = client().count(name).count
            except Exception:
                n = 0
            rel = ", ".join(related.get(name, [])[:3])
            lines.append(f"- `{name}` ({n} chunks): {ctx['desc'].get(name, 'no description')}"
                         + (f" · related: {rel}" if rel else ""))
    return "\n".join(lines).strip()


def clip(s, limit=12000):
    """Truncate on a line boundary and say so — a silently chopped file makes a small model
    try to complete the code instead of answering the question."""
    if len(s) <= limit:
        return s
    cut = s.rfind("\n", 0, limit) or limit
    return s[:cut] + (f"\n\n[…truncated {len(s) - cut} more characters. This is NOT the end of the "
                      "file and nothing here is for you to complete — call read_file with "
                      "start/end if you need a specific range.]")


def ask(ctx, question, max_steps=10, verbose=True):
    # wide=0: the digest tail is for the calling harness, which judges well. The local
    # model measurably does not — handing it 42 extra paths just burns its context.
    dispatch = {"search": lambda c, **kw: t_search(c, wide=0, **kw),
                "grep": t_grep, "read_file": t_read_file}
    msgs = [{"role": "system", "content": SYSTEM.format(catalogue=catalogue(ctx))},
            {"role": "user", "content": question}]
    for _ in range(max_steps):
        m = chat(msgs, TOOLS)
        msgs.append(m)
        calls = m.get("tool_calls")
        if not calls:
            return m.get("content", "").strip()
        for tc in calls:
            name = tc["function"]["name"]
            args = tc["function"]["arguments"]
            if isinstance(args, str):
                args = json.loads(args)
            try:
                out = dispatch[name](ctx, **args)
            except Exception as e:                      # let the model recover from a bad call
                out = f"tool error: {e}"
            if verbose:
                print(f"  \033[2m⚙ {name}({json.dumps(args)[:110]}) → {len(out)} chars\033[0m",
                      file=sys.stderr)
            msgs.append({"role": "tool", "tool_name": name, "content": clip(out)})
    return "(gave up after tool-call limit)"
