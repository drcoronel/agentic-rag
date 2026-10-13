"""Reconciling the index with disk: one-shot sync, the watch daemon, and doctor."""
import contextlib, fcntl, hashlib, os, sys, threading, time, uuid
from collections import Counter
from pathlib import Path

from qdrant_client import models as qm

from . import config
from .config import DOC_PREFIX, MAX_BYTES, SKIP_DIRS, TEXT_EXT, TEXT_NAMES
from .llm import embed
from .qdrant import by_path, client, ensure, indexed_files, is_hybrid
from .state import (cache_key, describe, discover, drop_caches, load_state, rediscover,
                    save_state)
from .text import chunk, sparse

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
    config.STATE.parent.mkdir(parents=True, exist_ok=True)
    with open(str(config.STATE) + ".lock", "w") as f:
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
    live = {c.name for c in client().get_collections().collections if c.name.startswith(config.PREFIX)}

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
