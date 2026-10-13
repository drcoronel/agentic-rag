"""Repository discovery, the .rag.json state file, and the caches derived from it."""
import fnmatch, json, re, sys
from pathlib import Path

from . import config
from .config import DOC_FILES, SKIP_DIRS
from .llm import chat


def _as_list(v):
    if v is None:
        return None
    return [v] if isinstance(v, str) else list(v)


def slug(name):
    return config.PREFIX + re.sub(r"[^a-z0-9_-]+", "-", name.lower()).strip("-")


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
    st = json.loads(config.STATE.read_text()) if config.STATE.exists() else {}
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
    config.STATE.write_text(json.dumps(st, indent=2, sort_keys=True))


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
                     f"with --include, or `rag forget {sorted(clash)[0]}` first.")
        st["projects"][me] = new
    projects = {p: rs for p, rs in st["projects"].items() if project in (None, p)}
    if not projects:
        sys.exit(f"no project {project!r} — known: {', '.join(st['projects']) or '(none)'}\n"
                 "index one with: rag index -p <name> <dir> [<dir>...]")
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
                print(f"  {name}: describe failed ({e}) — rerun `rag describe`", file=sys.stderr)
                continue
        print(f"  {name}: {desc[name]}", file=sys.stderr)
    save_state(st)
    ctx["desc"] = desc
    return desc

# ---------------------------------------------------------------- derived caches

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

