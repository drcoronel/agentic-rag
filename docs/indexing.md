# Indexing notes

Implementation rationale for `config.py`, `llm.py`, `qdrant.py`, `state.py` and `sync.py`.
For what the commands *do*, see the README; this file is about why they do it that way.

## Configuration

`config.Settings` is a pydantic `BaseSettings`, read from the environment once at import
and exposed as the single `settings` instance that every module imports.

Being one shared object is load-bearing, not incidental. The selftest mutates
`settings.prefix` and `settings.state` at runtime so its collections are namespaced and
its state file lands in a temporary directory, and every module sees that immediately
because they all hold the same reference. The earlier module-constant version had to be
read as `config.PREFIX` at each use site for the same reason, which was easy to get
wrong — `from .config import PREFIX` silently captured the value and lost the override.

Field names are lowercase and env names come from `env_prefix="RAG_"`, with explicit
aliases where the variable predates that convention (`OLLAMA_HOST`, `QDRANT_URL`,
`QDRANT_API_KEY`, `RAG_EMBED_DIM`). `validate_assignment=True`, so a bad runtime
assignment raises at the assignment rather than somewhere downstream.

The corpus constants (extension and skip lists, chunk sizes, the embedding task
prefixes) are fields too. Nothing normally overrides them, but they are validated and
discoverable in the same place as everything else.

## Embedding

`embeddinggemma` wants the `doc_prefix` / `query_prefix` task strings. Skipping them
measurably hurts recall, so they are applied in `embed()` rather than left to callers.

Embedding runs against Ollama by default and against Vertex when `RAG_VERTEX_EMBED` is
set — the only part of the pipeline that is not already just a URL in the environment.

`gcp_token()` tries, in order: `GOOGLE_ACCESS_TOKEN`, the GCE metadata server (so it
works unchanged on Cloud Run), then the local `gcloud`. The result is cached for 45
minutes, because tokens last an hour and one index pass is hundreds of calls.

## Collection schema

Each repository is one Qdrant collection carrying a named dense vector and a `bm25`
sparse vector with the IDF modifier. `ensure()` checks the schema on every call and
deletes-and-recreates when the sparse vector is missing or the dense dimension changed;
sync refills it on the same pass. Everything in the index is rebuildable from disk, so
dropping a collection is cheap and always the right answer to a schema mismatch.

## Repository discovery

Every immediate subdirectory of every root is a repository. `include` globs match
against the directory name, which is what lets one root hold repos belonging to
different projects, and lets a project pick up a sibling cloned later.

Two projects claiming the same *repo* used to break silently rather than error: both
discover it, one overwrites the other in the collection map, and the loser's searches
return nothing at all. `context()` therefore compares the resolved repo *sets*, not the
root paths — sharing a root is fine as long as the globs keep the sets disjoint.

## Descriptions

`describe()` writes one line per repo into `.rag.json`, used as the routing catalogue in
the agent's system prompt.

Two things are deliberate in the prompt. The directory listing goes *after* the document,
because a 4000-character contributing guide otherwise buries the one signal that says
what the repo is about — a docs repo was being catalogued as "guidelines for
contributing". And `best_doc()` applies a 200-byte size floor: a two-byte stub README
passes an `is_file()` check but gives the model nothing, and it invents a subject.

A model hiccup during describe is logged and skipped, never raised, because `watch` calls
this on every sync and must not die of it.

## Reconciling with disk

`sync_repo()` is idempotent and takes the cheapest route that is still correct:

1. stat matches the recorded mtime and size → skip the file entirely.
2. stat differs but the SHA-256 matches → rewritten with the same bytes, just refresh
   the stored stat.
3. otherwise → re-chunk, re-embed, and replace the file's points.

Chunks carry a small line overlap so a statement split across a boundary survives in one
of the two pieces.

## The exclusive lock

Anything that mutates collections holds an advisory `flock` on `.rag.json.lock`.

The `watch` daemon and a manual `index` reconcile the same collections against different
snapshots of the project list. Run them at once and each prunes what the other just
created; the loser's upserts then 404 against a collection that no longer exists.
Blocking the daemon for the length of an index is the cheap half of that trade.

## The revision counter

Derived views (graph, topics) key their caches off `revision`, which `sync` bumps
whenever it actually changed something. Chunk counts alone would miss an edit that
happens to re-chunk to the same number. Sync is the only thing that touches the index,
so a counter it owns is exact and free to read.

`drop_caches()` discards both derived views whenever collections appear or disappear
underneath them — every edge touching a removed node is a lie.

## Scope rules, and why `doctor` exists

Nothing in the normal commands looks outside its own project. `sync` only prunes inside
the projects it was asked to sync, so `index -p a` cannot evict project b. The cost is
that drift between the three places the index lives — the state file, Qdrant, and disk —
is silent by construction.

`doctor` is the one view that reads all three at once. It is scoped to
`settings.prefix`, so with `RAG_PREFIX` set another namespace's collections are not
ours to call orphaned.

Its graph check compares the cache *key*, not the node set: an edit that leaves the
collection list unchanged still invalidates the graph, and a node-set comparison cannot
see that.

`forget` is keyed on the collection→project map rather than on `projects`, so it also
cleans up after a hand-edited state file. Those collections are otherwise unreachable
forever, since `sync` only prunes inside projects it can still see.

## Watch failure modes

Under `mcp --watch`, `watch()` runs as a daemon thread. An unhandled exception there
kills the watcher while the server keeps serving, so the index silently freezes at
whatever it last held — and the startup race against Qdrant that `SERVICE.md` documents
hits exactly this path. launchd cannot help, because the process is still alive and
still answering. Hence `_try_sync()`: failures are reported and retried every
`RETRY_SECS`, never raised.

After a burst of events (a git checkout, a save-all) the watcher waits a second for
things to settle, then re-runs discovery before syncing, since a repo may have been
cloned or removed rather than merely edited.
