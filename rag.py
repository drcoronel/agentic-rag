#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["qdrant-client", "watchdog", "mcp", "networkx", "numpy", "pyvis"]
# ///
"""Agentic RAG over local repos. Ollama embeds + answers, Qdrant stores.

One Qdrant collection per repository, each with a generated description so the
agent can pick which repos to search before it searches them. Repositories are
grouped into named *projects* (one root set each) so a session can scope to one.

  uv run rag.py index -p work ~/repos ~/notes
  uv run rag.py watch          # keeps the index in sync with the files
  uv run rag.py graph          # topology over the collections
  uv run rag.py ask "how is a failed job retried?"
"""
import argparse, contextlib, fcntl, fnmatch, hashlib, json, math, os, re, subprocess
import sys, threading, time, urllib.request, uuid
from collections import Counter
from pathlib import Path

from qdrant_client import QdrantClient, models as qm
from qdrant_client.http.exceptions import UnexpectedResponse

OLLAMA = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
QDRANT = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_KEY = os.environ.get("QDRANT_API_KEY") or None   # managed Qdrant wants one; local doesn't
EMBED_MODEL = os.environ.get("RAG_EMBED_MODEL", "embeddinggemma:300m")
CHAT_MODEL = os.environ.get("RAG_CHAT_MODEL", "qwen2.5:14b")
PREFIX = os.environ.get("RAG_PREFIX", "")        # collection namespace; selftest overrides it
DIM = int(os.environ.get("RAG_EMBED_DIM", "768"))
# Set to a Vertex model path to embed in GCP instead of Ollama, e.g.
# projects/<id>/locations/europe-north1/publishers/google/models/text-embedding-005
VERTEX_EMBED = os.environ.get("RAG_VERTEX_EMBED", "")

STATE = Path(__file__).parent / ".rag.json"

# embeddinggemma wants these task prefixes; skipping them measurably hurts recall.
DOC_PREFIX = "title: none | text: "
QUERY_PREFIX = "task: search result | query: "

TEXT_EXT = {".md", ".mdx", ".yaml", ".yml", ".py", ".ts", ".tsx", ".js", ".mjs", ".vue",
            ".json", ".tf", ".tfvars", ".sql", ".sh", ".html", ".css", ".toml", ".txt",
            ".cfg", ".ini", ".rs", ".go", ".java"}
SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", "dist", "build",
             ".next", ".nuxt", ".turbo", ".pytest_cache", ".mypy_cache", ".ruff_cache",
             "target", ".terraform"}
TEXT_NAMES = {"Makefile", "Dockerfile", "Justfile", "Procfile", "CODEOWNERS", "Caddyfile",
              ".gitignore", ".dockerignore", ".npmrc", ".nvmrc", ".env.example"}
DOC_FILES = ("AGENTS.md", "CLAUDE.md", "README.md")
MAX_BYTES = 200_000
CHUNK_CHARS = 1200
OVERLAP_LINES = 2


# ---------------------------------------------------------------- ollama

def _post(url, payload, timeout=600, headers=None):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json", **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


_token = (0.0, "")


def gcp_token():
    """Access token for Vertex: env override, then the metadata server (Cloud Run), then
    the local gcloud. Cached — tokens last an hour and an index pass is hundreds of calls."""
    global _token
    if _token[0] > time.time():
        return _token[1]
    t = os.environ.get("GOOGLE_ACCESS_TOKEN")
    if not t:
        try:
            t = _get("http://metadata.google.internal/computeMetadata/v1/instance/"
                     "service-accounts/default/token", {"Metadata-Flavor": "Google"})["access_token"]
        except Exception:
            t = subprocess.run(["gcloud", "auth", "print-access-token"],
                               capture_output=True, text=True).stdout.strip()
    if not t:
        sys.exit("RAG_VERTEX_EMBED is set but no GCP credentials — run `gcloud auth login`")
    _token = (time.time() + 2700, t)
    return t


def _get(url, headers, timeout=2):
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as r:
        return json.loads(r.read())


def embed(texts, prefix):
    """768-dim vectors. Ollama locally; Vertex when RAG_VERTEX_EMBED is set — the only
    part of the pipeline that is not already a URL in the environment."""
    out = []
    for i in range(0, len(texts), 64):
        batch = texts[i:i + 64]
        if VERTEX_EMBED:
            task = "RETRIEVAL_QUERY" if prefix == QUERY_PREFIX else "RETRIEVAL_DOCUMENT"
            region = VERTEX_EMBED.split("/locations/")[1].split("/")[0]
            r = _post(f"https://{region}-aiplatform.googleapis.com/v1/{VERTEX_EMBED}:predict",
                      {"instances": [{"task_type": task, "content": t} for t in batch],
                       "parameters": {"outputDimensionality": DIM}},
                      headers={"Authorization": f"Bearer {gcp_token()}"})
            out += [p["embeddings"]["values"] for p in r["predictions"]]
        else:
            out += _post(f"{OLLAMA}/api/embed",
                         {"model": EMBED_MODEL, "input": [prefix + t for t in batch]})["embeddings"]
    return out


def chat(messages, tools=None):
    body = {"model": CHAT_MODEL, "messages": messages, "stream": False, "options": {"temperature": 0}}
    if tools:
        body["tools"] = tools
    return _post(f"{OLLAMA}/api/chat", body)["message"]


# ---------------------------------------------------------------- repos & state

def slug(name):
    return PREFIX + re.sub(r"[^a-z0-9_-]+", "-", name.lower()).strip("-")


def discover(roots, include=None):
    """{collection: repo_dir} — every immediate subdirectory of every root is a repo.

    `include` is a list of globs matched against the directory name; when given, only
    matching directories count. That is what lets one root hold repos belonging to
    different projects, and lets a project pick up a later sibling automatically.
    """
    out = {}
    for r in roots:
        if not Path(r).is_dir():
            continue                     # a deleted root is doctor's problem, not a crash
        for d in sorted(Path(r).iterdir()):
            if not d.is_dir() or d.name in SKIP_DIRS:
                continue
            if include and not any(fnmatch.fnmatch(d.name, g) for g in include):
                continue
            out[slug(d.name)] = str(d)
    return out


def load_state():
    st = json.loads(STATE.read_text()) if STATE.exists() else {}
    old_roots, old_colls = st.pop("roots", None), st.get("collections")
    st.setdefault("projects", {})
    st.setdefault("desc", {})
    if old_roots and not st["projects"]:            # migrate pre-project state
        st["projects"]["default"] = old_roots
    if isinstance(old_colls, list):                 # collections were a flat list of names
        st["collections"] = {n: "default" for n in old_colls}
    st.setdefault("collections", {})
    return st


def save_state(st):
    STATE.write_text(json.dumps(st, indent=2, sort_keys=True))


def context(project=None, args_roots=None, include=None):
    """Resolve project -> roots (remembering it), then the repo -> collection map.

    No project selects every project at once; the collections stay in one namespace, so
    searching across projects is just not passing a filter.
    """
    st = load_state()
    inc = st.setdefault("include", {})
    if args_roots:
        me = project or "default"
        new = [str(Path(r).expanduser().resolve()) for r in args_roots]
        if include is not None:
            inc[me] = include
        # Two projects claiming the same *repo* silently break the filter rather than
        # erroring: both discover it, one overwrites the other in the collection map, and
        # the loser returns no results at all. Sharing a root is fine as long as the
        # include globs keep the repo sets disjoint, so compare the sets, not the roots.
        taken = set(discover(new, inc.get(me)))
        clash = {p for p, rs in st["projects"].items()
                 if p != me and taken & set(discover(rs, inc.get(p)))}
        if clash:
            sys.exit(f"those repositories already belong to project "
                     f"{', '.join(sorted(clash))}.\nProjects must not overlap — narrow "
                     f"with --include, or `rag.py forget {sorted(clash)[0]}` first.")
        st["projects"][me] = new
    projects = {p: rs for p, rs in st["projects"].items() if project in (None, p)}
    if not projects:
        sys.exit(f"no project {project!r} — known: {', '.join(st['projects']) or '(none)'}\n"
                 "index one with: rag.py index -p <name> <dir> [<dir>...]")
    save_state(st)

    ctx = {"projects": projects, "include": inc, "desc": st["desc"],
           "roots": [r for rs in projects.values() for r in rs], "graph": st.get("graph", {})}
    return rediscover(ctx)


def rediscover(ctx):
    """(Re)build the collection -> directory and collection -> project maps from disk."""
    repos, proj = {}, {}
    for p, roots in ctx["projects"].items():
        for name, d in discover(roots, ctx.get("include", {}).get(p)).items():
            # ponytail: same directory name in two projects collides on one collection;
            # namespace the slug per project if that ever actually happens.
            repos[name], proj[name] = d, p
    ctx["repos"], ctx["proj"] = repos, proj
    return ctx


def scope(ctx, collections=None, project=None):
    """The collection names a call may touch, after both filters."""
    allowed = [n for n in ctx["repos"] if project in (None, ctx["proj"].get(n))]
    sel = _as_list(collections)
    return [n for n in (sel or allowed) if n in allowed]


def best_doc(d):
    """Best description doc. Size floor matters: a 2-byte stub README beats nothing in an
    is_file() check, but gives the model nothing to work from and it invents a subject."""
    named = [p for f in DOC_FILES if (p := Path(d) / f).is_file() and p.stat().st_size > 200]
    if named:
        return named[0]
    mds = [p for p in Path(d).glob("*.md") if p.stat().st_size > 200]
    return max(mds, key=lambda p: p.stat().st_size, default=None)


def _listing(d):
    """Top-level entries, plus one level under `docs/`. A repo whose content lives in
    `docs/` otherwise gets described by its CONTRIBUTING-style AGENTS.md — platform-docs
    was catalogued as "guidelines for contributing" instead of the GCP platform docs."""
    top = sorted(x.name + ("/" if x.is_dir() else "")
                 for x in Path(d).iterdir() if not x.name.startswith("."))[:40]
    sub = sorted(f"docs/{x.name}" + ("/" if x.is_dir() else "")
                 for x in (Path(d) / "docs").iterdir()
                 if not x.name.startswith("."))[:40] if (Path(d) / "docs").is_dir() else []
    return ", ".join(top + sub)


def describe(ctx, force=False):
    """One-line summary per repo, from its own top-level doc. Cached in .rag.json."""
    st = load_state()
    desc = st.setdefault("desc", {})
    for name, d in ctx["repos"].items():
        if name in desc and not force:
            continue
        doc, listing = best_doc(d), _listing(d)
        if doc is None:
            desc[name] = f"{Path(d).name} — no description document. Top level: {listing}"
        else:
            # Entries go *after* the document, not before: a 4000-char contributing guide
            # otherwise buries the one signal that says what the repo is actually about.
            prompt = (f"Repository directory name: {Path(d).name}\n\n"
                      f"--- {doc.name}\n{doc.read_text(errors='replace')[:4000]}\n\n"
                      f"Entries: {listing}\n\n"
                      "In at most 40 words and one paragraph, state what this repository contains "
                      "and what kinds of questions it can answer. Describe only what is shown "
                      "above — never invent a subject area that is not mentioned. If the document "
                      "above is a contributing guide, style guide or similar meta-document about "
                      "working on the repository, describe the repository's actual subject matter "
                      "from the entries instead — a reader needs to know what is in here, not how "
                      "to submit a pull request. Name the concrete subject areas — this is a "
                      "routing signal, so a reader must be able to tell from it whether their "
                      "question belongs here. Never write 'and more', 'etc.' or 'among others'. "
                      "Reply in English. No preamble, no markdown.")
            try:
                desc[name] = chat([{"role": "user", "content": prompt}])["content"].strip().replace("\n", " ")
            except Exception as e:               # never let a model hiccup kill `watch`
                print(f"  {name}: describe failed ({e}) — rerun `rag.py describe`", file=sys.stderr)
                continue
        print(f"  {name}: {desc[name]}", file=sys.stderr)
    save_state(st)
    ctx["desc"] = desc
    return desc


# ---------------------------------------------------------------- store

_client = None


def client():
    global _client
    if _client is None:
        _client = QdrantClient(url=QDRANT, api_key=QDRANT_KEY, timeout=120)
    return _client


def is_hybrid(name):
    p = client().get_collection(name).config.params
    return (isinstance(p.vectors, dict) and "bm25" in (p.sparse_vectors or {})
            and getattr(p.vectors.get("dense"), "size", None) == DIM)


def ensure(name):
    c = client()
    if c.collection_exists(name):
        if is_hybrid(name):
            return c
        c.delete_collection(name)       # wrong schema or wrong dim; sync refills it
        print(f"  {name}: rebuilding (schema or embedding dim changed)", file=sys.stderr)
    c.create_collection(
        name,
        vectors_config={"dense": qm.VectorParams(size=DIM, distance=qm.Distance.COSINE)},
        sparse_vectors_config={"bm25": qm.SparseVectorParams(modifier=qm.Modifier.IDF)})
    c.create_payload_index(name, "path", qm.PayloadSchemaType.KEYWORD)
    return c


def by_path(path):
    return qm.Filter(must=[qm.FieldCondition(key="path", match=qm.MatchValue(value=path))])


def indexed_files(name):
    """path -> (hash, mtime, size) for everything currently in one collection."""
    seen, offset = {}, None
    while True:
        points, offset = client().scroll(name, limit=4096, offset=offset,
                                         with_payload=["path", "hash", "mtime", "size"],
                                         with_vectors=False)
        for p in points:
            seen[p.payload["path"]] = (p.payload["hash"], p.payload["mtime"], p.payload["size"])
        if offset is None:
            return seen


# ---------------------------------------------------------------- chunking

def chunk(text):
    """[(start_line, end_line, text)], 1-based inclusive, with a small overlap."""
    lines = text.splitlines()
    out, buf, start = [], [], 0
    for i, ln in enumerate(lines):
        buf.append(ln)
        if sum(map(len, buf)) >= CHUNK_CHARS:
            out.append((start + 1, i + 1, "\n".join(buf)))
            buf, start = buf[-OVERLAP_LINES:], max(0, i - OVERLAP_LINES + 1)
    if any(l.strip() for l in buf):
        out.append((start + 1, len(lines), "\n".join(buf)))
    return out


# ---------------------------------------------------------------- bm25

# Deliberately not fastembed's Bm25: its English stemmer and stopword list are tuned for
# prose, and this corpus is mostly YAML keys, project ids and identifiers. Splitting those
# the way a reader would is worth more here than stemming, and costs no dependency.
_WORD = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.\-/]*")
_PART = re.compile(r"[a-z0-9]+|[A-Z][a-z0-9]+|[A-Z]+(?![a-z])")
BM25_K1, BM25_B = 1.2, 0.75
BM25_AVGDL = 180            # ≈ tokens in a CHUNK_CHARS chunk; only the ratio matters


def tokens(text):
    """Whole identifiers, plus their camelCase / snake_case / dotted / hyphenated parts.

    `allowDelete` also yields allow and delete; `zoneclass` stays whole. Exact literals are
    precisely what dense vectors are worst at, so they have to survive intact.
    """
    for w in _WORD.findall(text):
        yield w.lower()
        parts = _PART.findall(w)
        if len(parts) > 1:
            for p in parts:
                if len(p) > 1:
                    yield p.lower()


def term_id(tok):
    """Stable u32 id for a token — Qdrant sparse indices are u32, and a hash means no
    vocabulary file to keep in sync with the index."""
    return int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=4).digest(), "big")


def sparse(text, query=False):
    """BM25 term weights. Qdrant's IDF modifier supplies the corpus half at query time,
    so a document stores only the saturated term frequency, and a query only its terms."""
    tf = Counter(term_id(t) for t in tokens(text))
    if query:
        return qm.SparseVector(indices=list(tf), values=[1.0] * len(tf))
    n = sum(tf.values()) or 1
    norm = BM25_K1 * (1 - BM25_B + BM25_B * n / BM25_AVGDL)
    return qm.SparseVector(indices=list(tf), values=[f / (f + norm) for f in tf.values()])


# ---------------------------------------------------------------- sync

def walk(repo_dir):
    for dirpath, dirnames, filenames in os.walk(repo_dir):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            if Path(fn).suffix.lower() not in TEXT_EXT and fn not in TEXT_NAMES:
                continue            # Makefile and friends carry no suffix to match on
            p = os.path.join(dirpath, fn)
            try:
                st = os.stat(p)
            except OSError:
                continue
            if 0 < st.st_size <= MAX_BYTES:
                yield p, st


def sync_repo(name, repo_dir, quiet=False):
    """Reconcile one collection with one repo directory. Idempotent."""
    ensure(name)
    known = indexed_files(name)
    c = client()
    on_disk, changed, touched = set(), [], []

    for p, st in walk(repo_dir):
        on_disk.add(p)
        prev = known.get(p)
        if prev and prev[1] == st.st_mtime and prev[2] == st.st_size:
            continue                                     # fast path: stat unchanged
        data = Path(p).read_bytes()
        h = hashlib.sha256(data).hexdigest()
        if prev and prev[0] == h:
            touched.append((p, st))                      # rewritten, same bytes
        else:
            changed.append((p, st, h, data.decode("utf-8", "replace")))

    gone = set(known) - on_disk
    for p in gone:
        c.delete(name, points_selector=by_path(p))
    for p, st in touched:
        c.set_payload(name, payload={"mtime": st.st_mtime, "size": st.st_size}, points=by_path(p))

    parent = str(Path(repo_dir).parent)
    for n, (p, st, h, text) in enumerate(changed, 1):
        pieces = chunk(text)
        r = os.path.relpath(p, parent)
        c.delete(name, points_selector=by_path(p))       # drop stale chunks first
        if pieces:
            vecs = embed([f"{r}\n{t}" for _, _, t in pieces], DOC_PREFIX)
            c.upsert(name, points=[
                qm.PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{p}:{i}")),
                    vector={"dense": v, "bm25": sparse(f"{r}\n{t}")},
                    payload={"path": p, "rel": r, "s": s, "e": e, "text": t,
                             "hash": h, "mtime": st.st_mtime, "size": st.st_size},
                ) for i, ((s, e, t), v) in enumerate(zip(pieces, vecs))])
        if not quiet:
            print(f"  [{name} {n}/{len(changed)}] {r} ({len(pieces)} chunks)", file=sys.stderr)
    return len(changed), len(gone)


@contextlib.contextmanager
def exclusive():
    """Advisory lock around anything that mutates collections.

    The `watch` daemon and a manual `index` reconcile the same collections against
    different snapshots of the project list. Run them at once and each prunes what the
    other just created — the loser's upserts then 404 against a collection that no longer
    exists. Blocking the daemon for the length of an index is the cheap half of the trade.
    """
    STATE.parent.mkdir(parents=True, exist_ok=True)
    with open(str(STATE) + ".lock", "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def sync(ctx, quiet=False):
    with exclusive():
        return _sync(ctx, quiet)


def _sync(ctx, quiet=False):
    st = load_state()
    known = st["collections"]
    # Only prune inside the projects being synced — `index -p a` must not evict project b.
    mine = {n for n, p in known.items() if p in ctx["projects"]}
    for stale in mine - set(ctx["repos"]):
        client().delete_collection(stale)                # repo directory disappeared
        known.pop(stale, None)
        st["desc"].pop(stale, None)
        if not quiet:
            print(f"dropped collection {stale}", file=sys.stderr)
    known.update(ctx["proj"])
    save_state(st)

    total = [0, 0]
    for name, d in ctx["repos"].items():
        a, b = sync_repo(name, d, quiet)
        total[0] += a
        total[1] += b
        if not quiet and (a or b):
            print(f"{name}: {a} indexed, {b} removed", file=sys.stderr)
    describe(ctx)          # a newly cloned repo needs its catalogue entry, not just chunks
    if any(total):
        # Derived views (graph, topics) key their caches off this. Chunk counts alone
        # would miss an edit that happens to re-chunk to the same number; sync is the only
        # thing that touches the index, so a counter it bumps is exact and free to read.
        st = load_state()
        st["revision"] = st.get("revision", 0) + 1
        save_state(st)
    if not quiet or any(total):
        print(f"sync: {total[0]} indexed, {total[1]} removed across "
              f"{len(ctx['repos'])} collections", file=sys.stderr)
    return tuple(total)


def forget(name, quiet=False):
    """Drop a whole project: its collections, their descriptions, and its root set.

    Keyed on the collection -> project map rather than on `projects`, so it also cleans
    up after a hand-edited state file — otherwise those collections are unreachable:
    `sync` only ever prunes inside the projects it was asked to sync.
    """
    with exclusive():
        return _forget(name, quiet)


def _forget(name, quiet=False):
    st = load_state()
    gone = [n for n, p in st["collections"].items() if p == name]
    if name not in st["projects"] and not gone:
        sys.exit(f"no project {name!r} — known: {', '.join(st['projects']) or '(none)'}")
    for n in gone:
        client().delete_collection(n)
        st["collections"].pop(n)
        st["desc"].pop(n, None)
        if not quiet:
            print(f"dropped collection {n}", file=sys.stderr)
    st["projects"].pop(name, None)
    st.get("include", {}).pop(name, None)
    drop_caches(st)                # every edge touching a removed node is now a lie
    save_state(st)
    if not quiet:
        print(f"forgot project {name!r}: {len(gone)} collections", file=sys.stderr)
    return gone


def doctor(ctx, fix=False, quiet=False):
    """Reconcile the three places the index lives: the state file, Qdrant, and disk.

    Nothing in the normal commands looks outside its own project — `sync` only prunes
    what it was asked to sync — so drift between the three is silent by construction.
    This is the one view that reads all of them at once. Returns the problem count.
    """
    st = load_state()
    tracked = st["collections"]
    # PREFIX-scoped: with RAG_PREFIX set, another namespace's collections are not ours
    # to call orphaned. PREFIX is "" by default, so normally this is every collection.
    live = {c.name for c in client().get_collections().collections if c.name.startswith(PREFIX)}

    untracked = live - set(tracked)
    vanished = set(tracked) - live
    orphaned = {n for n, p in tracked.items() if p not in st["projects"]}
    g = st.get("graph")

    found = []

    def issue(items, msg, hint):
        if items:
            found.append((sorted(items), msg, hint))

    issue([r for r in ctx["roots"] if not Path(r).is_dir()],
          "roots that no longer exist on disk",
          "rag.py index -p <project> <the roots you still have>")
    issue(untracked, "collections in Qdrant that no project tracks",
          "rag.py doctor --fix  (deletes them — all of it is reindexable from disk)")
    issue(vanished, "collections in the state file but missing from Qdrant",
          "rag.py doctor --fix  (drops the entries; the next sync rebuilds them)")
    issue(orphaned, "collections pointing at a project that no longer exists",
          "rag.py forget <project>, or rag.py doctor --fix")
    issue(set(ctx["repos"]) - set(tracked), "repositories on disk that were never indexed",
          "rag.py index")
    issue([n for n in sorted(set(tracked) & live) if not is_hybrid(n)],
          "collections with the wrong vector schema or dimension",
          "rag.py index  (ensure() rebuilds them)")
    issue(set(ctx["repos"]) - set(st["desc"]), "repositories with no catalogue description",
          "rag.py describe")
    inc = st.get("include", {})
    claims = Counter(n for p, roots in st["projects"].items() for n in discover(roots, inc.get(p)))
    issue([f"{n} (claimed {c}×)" for n, c in claims.items() if c > 1],
          "repositories claimed by more than one project — the filter silently drops one",
          "narrow one project with --include, or rag.py forget <the redundant one>")
    issue([p for p, roots in st["projects"].items() if not discover(roots, inc.get(p))],
          "projects that resolve to no repositories",
          "rag.py index -p <name> <dirs>, or rag.py forget <name>")
    issue([f"{p}: {', '.join(g)}" for p, g in inc.items() if p not in st["projects"]],
          "include globs left behind by a removed project", "rag.py doctor --fix")
    # Compared by cache key, not node set: an edit that leaves the collections unchanged
    # still invalidates the graph, and the node set alone cannot see that.
    key = cache_key(ctx)
    issue([] if g and g.get("key") == key
          else ["(never built)"] if not g else sorted(set(ctx["repos"]) ^ set(g["nodes"]))
          or [f"index moved on (revision {key['revision']})"],
          "topology graph missing or out of date", "rag.py graph")

    if not quiet:
        if not found:
            print(f"doctor: all clear — {len(tracked)} collections, "
                  f"{len(st['projects'])} projects, {len(ctx['roots'])} roots")
        for items, msg, hint in found:
            print(f"\n✗ {msg} ({len(items)})")
            print("".join(f"    {i}\n" for i in items[:8]), end="")
            if len(items) > 8:
                print(f"    … and {len(items) - 8} more")
            print(f"  fix: {hint}")

    stale_inc = [p for p in inc if p not in st["projects"]]
    if fix and (untracked or vanished or orphaned or stale_inc):
        with exclusive():
            for p in stale_inc:
                inc.pop(p)
            for n in sorted(untracked | orphaned):
                if n in live:
                    client().delete_collection(n)
                print(f"deleted collection {n}", file=sys.stderr)
            for n in sorted(vanished | orphaned):
                tracked.pop(n, None)
                st["desc"].pop(n, None)
            drop_caches(st)            # node set changed, so every edge is suspect
            save_state(st)
        print("\nrerun `rag.py doctor` to confirm, then `rag.py graph`", file=sys.stderr)
    return len(found)


RETRY_SECS = 30


def _try_sync(ctx, quiet=False):
    """sync(), but a failure is reported instead of raised. Returns whether it worked.

    Under `mcp --watch`, watch() is a daemon thread. An unhandled exception there kills
    the watcher while uvicorn keeps serving, so the index silently freezes at whatever it
    last held — and the startup race against Qdrant that SERVICE.md documents hits exactly
    this path. launchd cannot help: the process is still alive and still answering.
    """
    try:
        sync(ctx, quiet=quiet)
        return True
    except Exception as e:
        print(f"sync failed ({type(e).__name__}: {e}) — retrying in {RETRY_SECS}s",
              file=sys.stderr)
        return False


def watch(ctx):
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler

    dirty = threading.Event()

    class H(FileSystemEventHandler):
        def on_any_event(self, e):
            p = str(e.src_path)
            if set(p.split(os.sep)) & SKIP_DIRS:
                return
            if Path(p).suffix.lower() in TEXT_EXT or Path(p).name in TEXT_NAMES:
                dirty.set()

    ok = _try_sync(ctx)
    obs = Observer()
    for r in ctx["roots"]:
        obs.schedule(H(), r, recursive=True)
    obs.start()
    print(f"watching {len(ctx['roots'])} root(s) — ctrl-c to stop", file=sys.stderr)
    try:
        while True:
            if dirty.wait(timeout=1.0):
                time.sleep(1.0)      # let a burst (git checkout, save-all) settle
                dirty.clear()
            elif ok:
                continue
            else:
                time.sleep(RETRY_SECS)   # nothing changed, but the last sync failed
            rediscover(ctx)              # a repo may have been cloned or removed
            ok = _try_sync(ctx, quiet=True)
    except KeyboardInterrupt:
        pass
    finally:
        obs.stop()
        obs.join()


# ---------------------------------------------------------------- agent tools

def _as_list(v):
    if v is None:
        return None
    return [v] if isinstance(v, str) else list(v)


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


# ---------------------------------------------------------------- topology

def cache_key(ctx, project=None, **params):
    """What a derived view was computed from. Different key, stale cache.

    `revision` covers any change to the index itself, the scope covers a project being
    added or a repo cloned, and params cover the knobs — so a `-k 12` run cannot be
    served from a `-k 8` cache.
    """
    return {"revision": load_state().get("revision", 0),
            "scope": scope(ctx, project=project), **params}


def cached(name, key):
    """The stored view if it was computed from exactly this key, else None."""
    hit = load_state().get(name)
    return hit if hit and hit.get("key") == key else None


def store(name, key, payload):
    st = load_state()
    st[name] = {"key": key, **payload}
    save_state(st)
    return st[name]


def drop_caches(st):
    """Both derived views, whenever collections appear or disappear under them."""
    st.pop("graph", None)
    st.pop("topics", None)


def build_graph(ctx, quiet=False):
    """A graph over the collections, from two signals already present in the index:

    - **mentions** (directed, exact): chunks in A that name B. This is the sharp edge —
      a repo's docs pointing at another repo is a real dependency, not a guess.
    - **affinity** (undirected): cosine between the mean dense vector of each collection.
      Blurry by construction — a whole repo averaged to one point — so it is only ever
      read as a ranking among a node's own neighbours, never as an absolute number.

    No LLM and no second store: one scroll per collection over Qdrant.
    """
    names = sorted(ctx["repos"])
    alias = {n: {n.lower(), Path(ctx["repos"][n]).name.lower()} for n in names}
    cent = {n: [0.0] * DIM for n in names}
    ment = {n: Counter() for n in names}
    sizes = {}

    for n in names:
        offset, seen = None, 0
        while True:
            pts, offset = client().scroll(n, limit=512, offset=offset,
                                          with_payload=["text"], with_vectors=["dense"])
            for pt in pts:
                seen += 1
                c, v = cent[n], pt.vector["dense"]
                for i, x in enumerate(v):
                    c[i] += x
                t = pt.payload["text"].lower()
                for m in names:
                    if m != n and any(a in t for a in alias[m]):
                        ment[n][m] += 1
            if offset is None:
                break
        sizes[n] = seen
        if not quiet:
            print(f"  {n}: {seen} chunks, {len(ment[n])} outbound", file=sys.stderr)
        norm = math.sqrt(sum(x * x for x in cent[n])) or 1.0
        cent[n] = [x / norm for x in cent[n]]

    edges = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            edges.append({"a": a, "b": b,
                          "sim": round(sum(x * y for x, y in zip(cent[a], cent[b])), 4),
                          "mentions": {a: ment[a][b], b: ment[b][a]}})
    edges.sort(key=lambda e: -e["sim"])

    related = {}
    for n in names:
        nb = [(e, e["b"] if e["a"] == n else e["a"]) for e in edges if n in (e["a"], e["b"])]
        # mentions first (hard evidence), affinity only to fill the remaining slots
        ranked = (sorted((x for x in nb if sum(x[0]["mentions"].values())),
                         key=lambda x: -sum(x[0]["mentions"].values()))
                  + sorted(nb, key=lambda x: -x[0]["sim"]))
        related[n] = list(dict.fromkeys(o for _, o in ranked))[:4]

    g = store("graph", cache_key(ctx),
              {"nodes": {n: {"chunks": sizes[n], "project": ctx["proj"][n]} for n in names},
               "edges": edges, "related": related})
    ctx["graph"] = g
    return g


def graph_table(g, top=25):
    rows = sorted(g["edges"], key=lambda e: (-sum(e["mentions"].values()), -e["sim"]))[:top]
    w = max((len(n) for n in g["nodes"]), default=10) + 2
    out = [f"{len(g['nodes'])} collections, {len(g['edges'])} pairs "
           f"(top {len(rows)} by mentions, then affinity)\n",
           f"{'':<{w}}{'':<{w}}{'a→b':>6}{'b→a':>6}{'affinity':>10}"]
    for e in rows:
        out.append(f"{e['a']:<{w}}{e['b']:<{w}}{e['mentions'][e['a']]:>6}"
                   f"{e['mentions'][e['b']]:>6}{e['sim']:>10.3f}")
    return "\n".join(out)


def graph_mermaid(g, sim_top=10):
    """Mentions as labelled arrows; the strongest affinity pairs that have no mention
    edge as dotted lines — otherwise the affinity half is a complete graph and unreadable."""
    ids, out = {}, ["graph LR"]
    nid = lambda n: ids.setdefault(n, f"n{len(ids)}")
    for n, meta in g["nodes"].items():
        out.append(f'  {nid(n)}["{n}<br/><i>{meta["project"]}</i>"]')
    quiet = []
    for e in g["edges"]:
        if sum(e["mentions"].values()):
            for src, dst in ((e["a"], e["b"]), (e["b"], e["a"])):
                if e["mentions"][src]:
                    out.append(f'  {nid(src)} -->|{e["mentions"][src]}| {nid(dst)}')
        else:
            quiet.append(e)
    for e in sorted(quiet, key=lambda e: -e["sim"])[:sim_top]:
        out.append(f'  {nid(e["a"])} -. "{e["sim"]:.2f}" .- {nid(e["b"])}')
    return "\n".join(out)


# ---------------------------------------------------------------- topics

STOP = set("the a an and or of to in for is are be it this that with as on at by from "
           "if then else not no yes you we they can will should must may use used using "
           "see https http com www md yaml json true false null none name type value "
           "run all any new set get add https org io dev test".split())


def sample_chunks(ctx, budget=3000, project=None):
    """(vectors, texts, collections) spread evenly over the collections.

    Qdrant scrolls in point-id order, and ids are uuid5 of the file path, so taking the
    first slice of each collection is arbitrary with respect to content — good enough for
    a shape, and far cheaper than reading every chunk back out.
    """
    import numpy as np
    names = scope(ctx, project=project)
    per = max(1, budget // max(1, len(names)))
    V, T, C = [], [], []
    for n in names:
        offset = None
        while len(T) < per * (names.index(n) + 1):
            pts, offset = client().scroll(n, limit=256, offset=offset,
                                          with_payload=["text"], with_vectors=["dense"])
            for p in pts:
                V.append(p.vector["dense"])
                T.append(p.payload["text"])
                C.append(n)
            if offset is None or not pts:
                break
    return np.asarray(V, dtype="float32"), T, C


def label_topic(members, T, df, total, n_terms=3):
    """Top distinctive terms for one cluster: how much more often a term appears in this
    cluster than in the corpus. Plain tf-idf over the BM25 tokenizer already in use."""
    here = Counter()
    for i in members:
        here.update(set(tokens(T[i])) - STOP)      # presence per chunk, not repetition
    scored = [(c / len(members) * math.log(total / (1 + df[t])), t)
              for t, c in here.items() if c >= max(2, len(members) * .15) and len(t) > 2]
    scored.sort(reverse=True)
    return " / ".join(t for _, t in scored[:n_terms]) or "(no distinctive terms)"


def tg_to_json(TG):
    return {"nodes": [[n, d] for n, d in TG.nodes(data=True)],
            "edges": [[a, b, d["weight"]] for a, b, d in TG.edges(data=True)]}


def tg_from_json(j):
    import networkx as nx
    TG = nx.Graph()
    for n, d in j["nodes"]:
        TG.add_node(n, **d)
    for a, b, w in j["edges"]:
        TG.add_edge(a, b, weight=w)
    return TG


def topics(ctx, rebuild=False, quiet=False, **params):
    """Cached topic graph. Clustering is the expensive half — a scroll of every sampled
    chunk plus an n² similarity matrix — and it only changes when the index does."""
    key = cache_key(ctx, project=params.get("project"),
                    **{p: params.get(p) for p in ("k", "budget", "floor")})
    hit = None if rebuild else cached("topics", key)
    if hit:
        TG = tg_from_json(hit)
        if not quiet:
            print_topics(TG, len(hit["nodes"]), cached_from=key["revision"])
        return TG
    TG = topic_graph(ctx, quiet=quiet, **params)
    store("topics", key, tg_to_json(TG))
    return TG


def print_topics(TG, n_topics, cached_from=None):
    head = f"{TG.number_of_nodes()} topics, {TG.number_of_edges()} links"
    print(head + (f"  (cached at revision {cached_from} — --rebuild to recompute)\n"
                  if cached_from is not None else "\n"))
    print(f"{'#':>3}{'chunks':>8}{'repos':>7}  {'topic':<44}dominant repo")
    for t, d in sorted(TG.nodes(data=True), key=lambda x: -x[1]["size"]):
        print(f"{t:>3}{d['size']:>8}{d['spread']:>7}  {d['label'][:42]:<44}{d['top']}")


def topic_graph(ctx, k=8, budget=3000, floor=.55, project=None, quiet=False,
                min_size=3, keep_links=3):
    """Cluster the chunk vectors into topics, then graph how the topics connect.

    Nodes are topics rather than repositories: a k-nearest-neighbour graph over sampled
    chunk vectors, cut into communities by Louvain (NetworkX ships it, so no extra
    dependency and no LLM). The interesting ones are the topics that span repositories —
    those are the places the corpus is really joined, which a per-repo graph cannot show.
    """
    import numpy as np, networkx as nx

    V, T, C = sample_chunks(ctx, budget, project)
    if len(T) < k + 2:
        sys.exit(f"only {len(T)} chunks sampled — index something first")
    V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-9
    S = V @ V.T
    np.fill_diagonal(S, -1.0)

    kk = min(k, len(T) - 1)
    nn = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
    G = nx.Graph()
    G.add_nodes_from(range(len(T)))
    for i, row in enumerate(nn):
        for j in row:
            if S[i, j] >= floor:
                G.add_edge(int(i), int(j), weight=float(S[i, j]))

    comms = sorted(nx.community.louvain_communities(G, weight="weight", seed=0),
                   key=len, reverse=True)
    # Singletons are chunks nothing else resembles. They carry no topic, and because they
    # are disconnected the spring layout flings them to the corners and collapses the rest
    # into an unreadable blob — so they are dropped, not drawn.
    dropped = sum(1 for c in comms if len(c) < min_size)
    comms = [c for c in comms if len(c) >= min_size]
    where = {i: t for t, c in enumerate(comms) for i in c}

    df = Counter()
    for text in T:
        df.update(set(tokens(text)))

    TG = nx.Graph()
    for t, members in enumerate(comms):
        cols = Counter(C[i] for i in members)
        TG.add_node(t, size=len(members), label=label_topic(members, T, df, len(T)),
                    collections=dict(cols), top=cols.most_common(1)[0][0],
                    project=ctx["proj"][cols.most_common(1)[0][0]], spread=len(cols))
    for u, v in G.edges():
        a, b = where.get(u), where.get(v)
        if a is not None and b is not None and a != b:
            TG.add_edge(a, b, weight=TG.get_edge_data(a, b, {}).get("weight", 0) + 1)

    # Every topic links to nearly every other, so the full edge set is a hairball that no
    # layout can spread. Keep each topic's strongest few links — the backbone, not the mesh.
    keep = {tuple(sorted((n, m))) for n in TG
            for m, _ in sorted(TG[n].items(), key=lambda kv: -kv[1]["weight"])[:keep_links]}
    TG.remove_edges_from([e for e in list(TG.edges()) if tuple(sorted(e)) not in keep])

    if not quiet:
        print(f"{len(T)} chunks sampled from {len(set(C))} collections → ", end="")
        print_topics(TG, len(comms))
        if dropped:
            print(f"\n({dropped} singleton topics dropped)")
    return TG


PALETTE = ["#2d9c8f", "#4a7fd4", "#d98324", "#9b59b6", "#d1495b", "#5c8001"]

LEGEND = """<div style="position:fixed;top:12px;left:12px;padding:10px 14px;z-index:9;
 border-radius:8px;background:rgba(255,255,255,.93);box-shadow:0 1px 6px rgba(0,0,0,.18);
 font:13px/1.5 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif">
 <b>__TITLE__</b><br>__KEYS__
 <div style="opacity:.6;font-size:11px;margin-top:4px">drag to pin · hover for detail</div>
</div>"""


def draw_graph(G, path, title, label_key="label"):
    """Interactive page via pyvis — a Python API over vis.js, so the physics, scaling and
    hover behaviour are configured here in Python rather than in hand-written JS.

    `cdn_resources="in_line"` inlines the library, so the file is standalone and opens
    with no network. It is one HTML file with no server and nothing installed.
    """
    from pyvis.network import Network

    projects = sorted({d["project"] for _, d in G.nodes(data=True)})
    color = dict(zip(projects, PALETTE * (len(projects) // len(PALETTE) + 1)))
    path = Path(path).with_suffix(".html")

    net = Network(height="100vh", width="100%", directed=False, bgcolor="#ffffff",
                  font_color="#222", cdn_resources="in_line", notebook=False)
    for n, d in G.nodes(data=True):
        spread = d.get("spread", 1)
        detail = "<br>".join(f"{k}: {v}" for k, v in sorted(
            d.get("collections", {}).items(), key=lambda kv: -kv[1]))
        net.add_node(
            str(n), label=d[label_key], value=d["size"], color=color[d["project"]],
            borderWidth=3 if spread > 1 else 1,
            title=f"{d[label_key]}\n{d['size']} chunks · {d['project']}"
                  + (f"\nspans {spread} repositories" if spread > 1 else "")
                  + (f"\n\n{detail}" if detail else ""),
            font={"size": 16 if spread > 1 else 13,
                  "face": "-apple-system", "strokeWidth": 4, "strokeColor": "#fff"})
    for a, b, d in G.edges(data=True):
        net.add_edge(str(a), str(b), value=d["weight"], title=str(d["weight"]))

    net.set_options("""{
      "nodes": {"shape": "dot", "scaling": {"min": 8, "max": 52},
                "color": {"border": "#3338"}},
      "edges": {"color": {"color": "#999", "opacity": 0.4},
                "scaling": {"min": 1, "max": 10}, "smooth": {"type": "continuous"}},
      "physics": {"barnesHut": {"gravitationalConstant": -14000, "springLength": 220,
                                "avoidOverlap": 0.55},
                  "stabilization": {"iterations": 500}},
      "interaction": {"hover": true, "tooltipDelay": 120}
    }""")

    net.write_html(str(path), notebook=False, open_browser=False)
    keys = " ".join(
        f'<span style="display:inline-block;width:10px;height:10px;border-radius:2px;'
        f'background:{color[p]};margin:0 5px 0 10px"></span>{p}' for p in projects)
    html = path.read_text().replace(
        "</body>", LEGEND.replace("__TITLE__", title).replace("__KEYS__", keys) + "</body>")
    # pyvis's template still pulls Bootstrap off a CDN for filter/config panels we never
    # enable. Dropping those two tags is what makes the file genuinely offline.
    html = re.sub(r'\s*<(?:script|link)[^>]*(?:cdn\.jsdelivr|cdnjs|unpkg)[^>]*>'
                  r'(?:</script>)?', "", html)
    path.write_text(html)
    return path


def collection_graph(g, ctx):
    """The per-repository graph as a NetworkX object, so it draws through the same path."""
    import networkx as nx
    G = nx.Graph()
    for n, m in g["nodes"].items():
        G.add_node(n, size=m["chunks"], label=n, project=m["project"], spread=1)
    for e in g["edges"]:
        w = sum(e["mentions"].values())
        if w:
            G.add_edge(e["a"], e["b"], weight=w)
    return G


# ---------------------------------------------------------------- mcp server

def serve_mcp(ctx, transport="stdio", port=8080, also_watch=False):
    """Expose retrieval to an external agent harness (Antigravity, Claude Code, ...).

    Deliberately only two tools: the harness already has better grep/read_file than we
    would ship. Semantic search over the index is the thing it cannot do itself.
    """
    from mcp.server.mcpserver import MCPServer      # was FastMCP before mcp 2.x

    if also_watch:                                  # one process: serve + keep the index fresh
        threading.Thread(target=watch, args=(ctx,), daemon=True).start()

    m = MCPServer("repo-rag")

    @m.tool()
    def collections(project: str | None = None) -> str:
        """Map of every indexed repository: project, name, what it contains, size, and
        which other repositories it is related to.

        One cheap call. Use it instead of listing directories to work out which
        repository owns a topic — it covers repositories that may not be open in the
        current workspace folder. Repositories are grouped into projects; pass `project`
        to see only one group. `related:` names the repositories that cite this one or
        cover the same ground — follow those when one repository only half-answers.
        """
        return catalogue(ctx, project)

    @m.tool()
    def search(query: str, collections: list[str] | None = None, k: int = 8,
               project: str | None = None) -> str:
        """Search the indexed repositories by meaning. USE THIS FIRST for any question
        about these repositories, before grep or listing directories.

        It matches on meaning rather than exact strings, so it finds the right file when
        you do not know what term the docs use — and it searches the whole indexed
        corpus, including repositories outside the open workspace folder. One call
        typically replaces a directory crawl plus several greps.

        Prefer grep only when you already know the literal string (an identifier, a YAML
        key, a config value).

        Returns two parts. First the best hits in full, as
        `[collection] path:start-end (score)` then the chunk text. Then a list of further
        candidates as paths only — scan those too and open any that look right, because
        the ranking is imperfect and the answer is often just below the cut.

        Pass `collections` (names from the `collections` tool) to narrow to the
        repositories that could plausibly answer, or `project` to stay inside one group;
        omit both to search everything. Results are fragments — open the cited file for
        full context.
        """
        return t_search(ctx, query, k=k, collections=collections, project=project)

    if transport == "stdio":
        m.run()
    else:
        m.run(transport=transport, host="127.0.0.1", port=port)   # loopback only; no auth on these tools


# ---------------------------------------------------------------- eval

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


# ---------------------------------------------------------------- selftest

def selftest():
    global PREFIX
    text = "\n".join(f"line {i} " + "x" * 40 for i in range(100))
    cs = chunk(text)
    assert cs[0][0] == 1 and cs[-1][1] == 100, cs
    assert all(s <= e for s, e, _ in cs) and len(cs) > 1, cs
    assert all(cs[i + 1][0] <= cs[i][1] for i in range(len(cs) - 1)), "chunks must overlap, not gap"
    assert chunk("") == [] and chunk("   \n  \n") == []

    assert {"allowdelete", "allow", "delete"} <= set(tokens("  allowDelete: true")), "camelCase"
    assert {"non-exposed", "non", "exposed"} <= set(tokens("zoneclass: non-exposed")), "hyphen"
    assert "zoneclass" in set(tokens("zoneclass: non-exposed")), "key must survive whole"
    assert sparse("a a b").values[0] > sparse("a b").values[0], "tf must saturate upward"

    PREFIX = "selftest" + uuid.uuid4().hex[:6] + "-"
    try:
        _selftest_store()
    finally:
        # Every collection this run made is PREFIX-namespaced, so sweeping the prefix
        # cleans up whatever a failed assert skipped past. Keyed on the prefix rather
        # than on the state file: an early failure means there is no state file yet.
        for c in client().get_collections().collections:
            if c.name.startswith(PREFIX):
                client().delete_collection(c.name)
    print("selftest ok")


def _selftest_store():
    """The half that needs Qdrant and Ollama. Split out so selftest() can wrap it in the
    cleanup that deletes this run's collections even when an assertion fails."""
    import tempfile
    global STATE

    # a pre-hybrid collection must be rebuilt, not left to fail on every upsert
    legacy = slug("legacy-schema")
    client().create_collection(legacy, vectors_config=qm.VectorParams(size=DIM, distance=qm.Distance.COSINE))
    ensure(legacy)
    assert is_hybrid(legacy), "ensure must migrate an old single-vector collection"
    client().delete_collection(legacy)

    with tempfile.TemporaryDirectory() as tmp:
        STATE = Path(tmp) / "state.json"                 # never touch the real .rag.json
        root, root2 = Path(tmp) / "p1", Path(tmp) / "p2"
        alpha, beta, delta = root / "alpha-repo", root / "beta-repo", root2 / "delta-repo"
        for d in (alpha, beta, delta):
            d.mkdir(parents=True)
        (alpha / "a.md").write_text("# Alpha\nthe alpha document about penguins\n"
                                    "see also beta-repo for the other half\n")
        (alpha / "conf.yaml").write_text("zoneclass: non-exposed\nallowDelete: true\n")
        (beta / "b.md").write_text("# Beta\nthe beta document about turbines\n")
        (delta / "d.md").write_text("# Delta\nthe delta document about accordions\n")

        projects = {"one": [str(root)], "two": [str(root2)]}
        ctx = rediscover({"projects": projects, "roots": [str(root), str(root2)],
                          "desc": {}, "graph": {}})
        A, B, D = slug("alpha-repo"), slug("beta-repo"), slug("delta-repo")
        assert set(ctx["repos"]) == {A, B, D}, ctx["repos"]
        assert ctx["proj"] == {A: "one", B: "one", D: "two"}, ctx["proj"]
        st = load_state()                  # context() persists this in real use
        st["projects"] = projects
        save_state(st)

        assert sync(ctx, quiet=True) == (4, 0)
        assert sync(ctx, quiet=True) == (0, 0), "second sync must be a no-op"
        assert all(ctx["desc"].get(n) for n in (A, B, D)), "sync must describe every collection"

        # syncing one project must not evict another project's collections
        one = rediscover({"projects": {"one": projects["one"]}, "roots": [str(root)],
                          "desc": ctx["desc"], "graph": {}})
        assert sync(one, quiet=True) == (0, 0)
        assert client().collection_exists(D), "project-scoped sync must not prune other projects"

        # project is a search filter, not a separate index
        assert scope(ctx, project="two") == [D]
        assert "accordions" not in t_search(ctx, "accordions", project="one")
        assert "accordions" in t_search(ctx, "accordions", project="two")

        # topology: a repo naming another is a directed edge, and ranks first as related
        g = build_graph(ctx, quiet=True)
        ab = next(e for e in g["edges"] if {e["a"], e["b"]} == {A, B})
        assert ab["mentions"][A] == 1 and ab["mentions"][B] == 0, ab
        assert g["related"][A][0] == B, g["related"]
        assert f"{A} -->|1|" not in graph_mermaid(g), "mermaid must use node ids, not names"
        assert "graph LR" in graph_mermaid(g) and "-->" in graph_mermaid(g)
        CG = collection_graph(g, ctx)
        assert set(CG) == set(g["nodes"]), "every collection must become a node"
        assert CG.has_edge(A, B) and CG[A][B]["weight"] == 1, "a mention must become an edge"
        page = Path(draw_graph(CG, Path(tmp) / "c", "test"))
        assert page.suffix == ".html" and page.stat().st_size > 20000, "pyvis must inline vis.js"
        body = page.read_text()
        assert all(n in body for n in g["nodes"]), "every collection must reach the page"
        assert "__TITLE__" not in body and "one" in body, "legend must be filled in"
        assert "jsdelivr" not in body and "unpkg" not in body, "page must not call out"

        # topics: clustered chunk vectors, labelled from the corpus, drawn the same way
        TG = topic_graph(ctx, k=2, budget=50, floor=.0, quiet=True)
        assert TG.number_of_nodes() >= 1, "clustering must yield at least one topic"
        assert sum(d["size"] for _, d in TG.nodes(data=True)) == 4, "every chunk lands once"
        assert all(d["label"] for _, d in TG.nodes(data=True)), "every topic needs a label"
        assert all(d["project"] in ("one", "two") for _, d in TG.nodes(data=True))
        assert Path(draw_graph(TG, Path(tmp) / "t", "test")).stat().st_size > 20000

        # caching: a second call is served from state, and survives the JSON round trip
        P = dict(k=2, budget=50, floor=.0)
        first = topics(ctx, quiet=True, **P)
        assert cached("topics", cache_key(ctx, **P)), "topics must be cached after a run"
        again = topics(ctx, quiet=True, **P)
        assert tg_to_json(again) == tg_to_json(first), "cache must round-trip exactly"
        assert not cached("topics", cache_key(ctx, k=3, budget=50, floor=.0)), \
            "a different -k must not be served from the cache"
        assert cached("graph", cache_key(ctx)), "graph must be cached after build_graph"

        # a consistent index must be clean; PREFIX scoping keeps the real one out of view
        assert doctor(ctx, quiet=True) == 0, "healthy index must report no problems"

        # include globs: one root, two projects, disjoint repo sets
        assert set(discover([str(root)], ["alpha-*"])) == {A}, "glob must filter by dir name"
        assert set(discover([str(root)], ["beta-repo"])) == {B}, "a literal is a valid glob"
        assert set(discover([str(root)], ["nothing-*"])) == set()
        assert set(discover([str(root)])) == {A, B}, "no globs means every subdirectory"
        assert set(discover([str(root) + "-does-not-exist"])) == set(), "missing root, no crash"
        split = rediscover({"projects": {"x": [str(root)], "y": [str(root)]},
                            "include": {"x": ["alpha-*"], "y": ["beta-*"]}})
        assert split["proj"] == {A: "x", B: "y"}, split["proj"]

        # a repo cloned while watch is running must get chunks AND a catalogue entry
        gamma = root / "gamma-repo"
        gamma.mkdir()
        (gamma / "c.md").write_text("# Gamma\nthe gamma document about telescopes\n")
        rediscover(ctx)
        rev = load_state().get("revision", 0)
        assert sync(ctx, quiet=True) == (1, 0)
        assert ctx["desc"].get(slug("gamma-repo")), "new repo must be described, not just indexed"
        # indexing a repo must invalidate both derived views, not serve a stale one
        assert load_state()["revision"] > rev, "a sync that changed the index must bump it"
        assert not cached("topics", cache_key(ctx, k=2, budget=50, floor=.0))
        assert not cached("graph", cache_key(ctx)), "new collection must invalidate the graph"

        # collections are isolated: a beta term must not surface when searching alpha
        assert "penguins" in t_search(ctx, "penguins", collections=[A])
        assert "turbines" not in t_search(ctx, "turbines", collections=[A])
        assert "turbines" in t_search(ctx, "turbines")           # ...but does across all

        # the lexical half: an exact identifier must rank first, which is the whole
        # reason for the sparse vector — dense alone blurs rare literals into their topic
        top = search_hits(ctx, "allowDelete", k=1, collections=[A], mode="sparse")
        assert top and top[0][0].payload["rel"].endswith("conf.yaml"), top
        assert "allowDelete" in t_search(ctx, "allowDelete", k=2, collections=[A])

        (alpha / "a.md").write_text("# Alpha\nrewritten to talk about submarines\n")
        assert sync(ctx, quiet=True) == (1, 0), "edit must reindex exactly one file"
        assert "submarines" in t_search(ctx, "submarines", collections=[A])
        assert "penguins" not in t_search(ctx, "penguins", k=2, collections=[A])

        (beta / "b.md").unlink()
        assert sync(ctx, quiet=True) == (0, 1), "delete must evict from the index"
        assert "turbines" not in t_search(ctx, "turbines", k=5)

        # forget must reach collections sync can no longer see. Project "two" is absent
        # from the state file's `projects` here, which is exactly the hand-edited shape
        # that used to orphan collections in Qdrant forever.
        st = load_state()
        st["projects"].pop("two")          # the hand-edit that used to strand collections
        save_state(st)
        rediscover(one)                    # ...it has gained gamma-repo since it was built
        assert sync(one, quiet=True) == (0, 0) and client().collection_exists(D), \
            "precondition: a scoped sync cannot reach another project's collections"
        assert doctor(one, quiet=True), "doctor must notice the orphan sync cannot see"
        assert forget("two", quiet=True) == [D]
        assert not client().collection_exists(D), "forget must delete the collections"
        st = load_state()
        assert D not in st["collections"] and D not in st["desc"], st
        assert "graph" not in st, "a stale graph must not outlive a removed node"


# ---------------------------------------------------------------- cli

def _wrote(path, open_it=False):
    print(f"wrote {Path(path).resolve()}")
    if open_it:
        subprocess.run(["open", str(Path(path).resolve())])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # -p lives on each subcommand rather than on `ap`, so `index -p x dirs` reads naturally;
    # a parser-level option would have to come before the subcommand.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-p", "--project", default=None,
                        help="limit to one named project; default all")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("index", "watch"):
        p = sub.add_parser(name, parents=[common])
        p.add_argument("roots", nargs="*")
        p.add_argument("-i", "--include", action="append", metavar="GLOB",
                       help="only take repo directories matching this glob (repeatable); "
                            "lets one root hold repos belonging to different projects")
    p = sub.add_parser("search", parents=[common])   # also the tool surface for an external agent
    p.add_argument("query", nargs="+")
    p.add_argument("-c", "--collection", action="append", help="repeatable; default all")
    p.add_argument("-k", type=int, default=8)
    sub.add_parser("ask", parents=[common]).add_argument("question", nargs="+")
    sub.add_parser("describe", parents=[common]).add_argument("--force", action="store_true")
    sub.add_parser("collections", parents=[common])
    sub.add_parser("projects", help="named root sets and their repositories")
    sub.add_parser("forget", help="drop a project: its collections, descriptions and roots"
                   ).add_argument("project")
    sub.add_parser("doctor", help="check the state file, Qdrant and disk against each other"
                   ).add_argument("--fix", action="store_true", help="repair what is safe")
    p = sub.add_parser("graph", parents=[common],
                       help="rebuild the topology over the collections")
    p.add_argument("--mermaid", action="store_true", help="print a Mermaid diagram instead")
    p.add_argument("--draw", nargs="?", const="graph.html", metavar="FILE",
                   help="render the graph with pyvis (default graph.html)")
    p.add_argument("--open", action="store_true", help="open the drawn file")
    p.add_argument("--rebuild", action="store_true",
                   help="recompute even if the cache is current")
    p = sub.add_parser("topics", parents=[common],
                       help="cluster the chunks into topics and graph how they connect")
    p.add_argument("--draw", nargs="?", const="topics.html", metavar="FILE",
                   help="render the topic graph with pyvis (default topics.html)")
    p.add_argument("--open", action="store_true", help="open the drawn file")
    p.add_argument("-k", type=int, default=8, help="neighbours per chunk, default 8")
    p.add_argument("--budget", type=int, default=3000, help="chunks to sample, default 3000")
    p.add_argument("--floor", type=float, default=.55, help="min cosine for a link")
    p.add_argument("--rebuild", action="store_true",
                   help="recompute even if the cache is current")
    sub.add_parser("selftest")
    p = sub.add_parser("eval", parents=[common],
                       help="retrieval quality on a .jsonl of {q, path, kind}")
    p.add_argument("file", nargs="?", default=str(Path(__file__).parent / "eval.jsonl"))
    p.add_argument("-k", type=int, default=5)
    p.add_argument("-m", "--mode", action="append", choices=["dense", "sparse", "hybrid"])
    p.add_argument("--misses", action="store_true", help="list the questions that found nothing")
    p = sub.add_parser("mcp", parents=[common],
                       help="serve retrieval as MCP tools for an agent harness")
    p.add_argument("--http", action="store_true", help="streamable-http instead of stdio")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--watch", action="store_true", help="also keep the index synced")
    a = ap.parse_args()

    if a.cmd == "selftest":
        return selftest()
    if a.cmd == "forget":                 # must not go through context(): the project may
        return forget(a.project)          # already be half-gone from the state file
    ctx = context(getattr(a, "project", None), getattr(a, "roots", None),
                  getattr(a, "include", None))

    if a.cmd == "index":
        sync(ctx)                                   # sync() fills in missing descriptions
    elif a.cmd == "watch":
        watch(ctx)
    elif a.cmd == "describe":
        describe(ctx, force=a.force)
    elif a.cmd == "collections":
        print(catalogue(ctx))
    elif a.cmd == "doctor":
        sys.exit(1 if doctor(ctx, a.fix) and not a.fix else 0)
    elif a.cmd == "projects":
        for p, roots in sorted(ctx["projects"].items()):
            names = [n for n in ctx["repos"] if ctx["proj"][n] == p]
            globs = ctx["include"].get(p)
            print(f"{p}: {len(names)} repos in {', '.join(roots)}"
                  + (f" matching {', '.join(globs)}" if globs else "")
                  + "\n  " + "\n  ".join(names))
    elif a.cmd == "graph":
        g = None if a.rebuild else cached("graph", cache_key(ctx))
        if g:
            print(f"(cached at revision {g['key']['revision']} — --rebuild to recompute)",
                  file=sys.stderr)
        else:
            g = build_graph(ctx)
        if a.draw:
            out = draw_graph(collection_graph(g, ctx), Path(a.draw).expanduser(),
                             f"repo-rag — {len(g['nodes'])} collections, "
                             "edge width = cross-references")
            _wrote(out, a.open)
        else:
            print(graph_mermaid(g) if a.mermaid else graph_table(g))
    elif a.cmd == "topics":
        TG = topics(ctx, rebuild=a.rebuild, k=a.k, budget=a.budget, floor=a.floor,
                    project=a.project)
        if a.draw:
            cross = sum(1 for _, d in TG.nodes(data=True) if d["spread"] > 1)
            _wrote(draw_graph(TG, Path(a.draw).expanduser(),
                              f"repo-rag — {TG.number_of_nodes()} topics, {cross} spanning "
                              "more than one repository (bold)"), a.open)
    elif a.cmd == "eval":
        evaluate(ctx, a.file, a.k, tuple(a.mode or ("dense", "sparse", "hybrid")), a.misses)
    elif a.cmd == "mcp":
        serve_mcp(ctx, "streamable-http" if a.http else "stdio", a.port, a.watch)
    elif a.cmd == "search":
        print(t_search(ctx, " ".join(a.query), k=a.k, collections=a.collection,
                       project=a.project))
    elif a.cmd == "ask":
        print("\n" + ask(ctx, " ".join(a.question)) + "\n")


if __name__ == "__main__":
    main()
