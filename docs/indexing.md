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

## The secret guardrail

Indexing was a pure allowlist: a file was taken if its extension was in `text_ext` or
its name in `text_names`. That meant `.env` and `private.key` were skipped only
incidentally — nothing checked for secrets — while `terraform.tfvars`,
`secrets.yaml`, `service-account.json` and `.npmrc` were indexed like any other file,
with their full text stored in the chunk payload of a Qdrant that has no auth.

`sensitive.is_secret()` is now a denylist over the filename, checked **before** the
allowlist, at all three places a file can reach the model:

| path | guard |
|---|---|
| `sync.walk()` | the file is never read, so it is never chunked or embedded |
| `agent.t_grep()` | matching lines are dropped, with a count of what was withheld |
| `agent.t_read_file()` | refuses by name, since it takes an arbitrary path |

Grep and read_file need their own guard because they go to the filesystem directly and
never consult the index.

### Two layers

`is_secret()` matches the filename. `secret_in()` matches the *contents*: vendor-shaped
key material (PEM private keys, `AKIA…`, `ghp_…`, `AIza…`, JWTs) keeps the whole file
out whatever it is called, checked in `sync_repo` right after the file is read and
before it is chunked. A file rejected that way is also left out of `on_disk`, so a copy
indexed before the check existed gets pruned on the same pass.

The content rules are only vendor formats. A generic `password: "..."` rule was left
out on purpose: it fires on every placeholder in every how-to, and a skipped document
is a silent hole in the index that nobody notices.

With `settings.tracked_only`, `walk()` narrows again to what `git ls-files` reports.
That inherits the repository's own `.gitignore`, which is a better statement of "never
meant to be read" than any denylist written here. Non-git directories are unaffected.

The filename list is deliberately broad. A false positive costs one unsearchable file; a false
negative embeds a credential. `secret_allow` is checked first so committed placeholders
(`.env.example`, `*.template`) stay indexable, and so does prose — `*.md`, `*.mdx`,
`*.rst`, `*.txt` are exempt because the name hints (`*secret*`, `*credential*`) are
what catch documentation *about* credentials, like a `sealedsecrets-create.md` how-to.
Nothing is lost by that exemption: detection is filename-only, so a credential pasted
into a Markdown file was never going to be caught either way. The hard patterns
(`.env`, `*.key`, `.netrc`) cannot be prose, so they are unaffected.

`.secrets`, `.ssh` and `.gnupg` are in `skip_dirs`, so those directories are not walked
at all.

**What this does not do.** A hand-rolled secret with no recognisable shape, sitting in
a conventionally-named file, is still indexed — `DB_PASS: hunter2` in `config.yaml`
matches no pattern. This is a guardrail against the routine cases, not a scanner.

### Finding what is already indexed

The denylist only applies going forward. `rag secrets` scrolls the `path` payload of
every collection and reports what is already stored:

```bash
uv run rag secrets            # report; exits 1 if anything matches
uv run rag secrets --purge    # delete those points now
```

A plain `rag index` also evicts them, because a denied file stops appearing in `walk()`
and `sync` prunes whatever it can no longer see — `--purge` just does it immediately.

Purging removes the copy in Qdrant. It does not un-disclose anything that was already
embedded, so rotate whatever was found.

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

## Other hardening

- **Qdrant binds loopback.** `compose.yaml` maps `127.0.0.1:6333:6333`, not `6333:6333`.
  The default Docker form binds `0.0.0.0`, and Qdrant has no auth, so on any shared
  network every chunk of every indexed repo was readable by anyone who could route to
  the machine. Running "only locally" does not help when the listener is not local.

- **`read_file` is confined to the roots.** It accepts an arbitrary path, so it resolves
  the candidate and requires it under an indexed root. `resolve()` closes both the
  absolute-path case (`/etc/hosts`) and a symlink pointing out of a root. Only
  `collections` and `search` are exposed over MCP today, but that is one config change
  away from being remote.

- **`save_state` writes atomically**, via a temp file and `os.replace`. The flock keeps
  two writers apart but not a torn one, and that file holds every generated description
  plus both derived caches — a truncated write costs an LLM pass over the whole corpus.

- **`rag selftest --offline`** runs the half that needs no services: chunking, the
  tokenizer, BM25 saturation, and both secret guards. Suitable for CI or a pre-commit
  hook.
