# Agentic RAG over local repos — PoC

Ask questions about your repos in natural language. Fully local: Ollama for
embeddings and reasoning, Qdrant for vectors. The index stays in sync with the files.

## Three deployment shapes

The same index and the same tool surface, three ways to drive it:

| # | shape | agent | index | status |
|---|---|---|---|---|
| 1 | **Fully local** | `qwen2.5:14b` via Ollama, `rag ask` | Qdrant in Docker | working |
| 2 | **Hosted in GCP** | Antigravity | Vertex embeddings, managed Qdrant, MCP on Cloud Run | config path in place, not deployed |
| 3 | **Local index, Antigravity harness** | Antigravity | Qdrant in Docker, local MCP server over stdio | working |

Shape 1 is the "nothing leaves the machine" proof. Shape 3 is the daily driver — a strong
model on top of a local index. Shape 2 is the team version.

Shape 2 is four environment variables, not a rewrite — every backend the server talks to is
a URL it reads from the environment:

| var | default | shape 2 |
|---|---|---|
| `QDRANT_URL` | `http://localhost:6333` | the managed cluster |
| `QDRANT_API_KEY` | unset (local Qdrant has no auth) | the cluster key |
| `RAG_VERTEX_EMBED` | unset → Ollama | `projects/<id>/locations/<region>/publishers/google/models/text-embedding-005` |
| `RAG_EMBED_DIM` | `768` | whatever the model emits |

With `RAG_VERTEX_EMBED` set, `embed()` calls Vertex instead of Ollama and authenticates
from the Cloud Run metadata server (locally: `GOOGLE_ACCESS_TOKEN`, else `gcloud auth
print-access-token`). Changing `RAG_EMBED_DIM` rebuilds every collection on the next sync —
dense vectors of two different widths cannot share an index.

What is genuinely *not* built: the hosting project, the
Cloud Run service and IAP, and a shared index that isn't one person's laptop. The `ask`
path still needs Ollama for the chat model; the MCP path does not.

Shapes 2 and 3 share one surface: `rag mcp`. It exposes exactly two tools —
`collections` and `search` — because the harness already has better `grep` and file
reading than we would ship. Semantic search over the index is the only thing it can't do
itself. `search` deliberately returns more than it ranks confidently: full text for the
best hits, then every remaining candidate as a path-only digest, because the harness is a
better relevance judge than anything that runs locally (measured, see *Retrieval quality*).

### Shape 3 setup (local index + Antigravity)

Start the server — one process serves the tools *and* keeps the index synced:

```bash
uv run rag mcp --http --watch
```

Then register it in `~/.gemini/config/mcp_config.json`:

```json
{
  "mcpServers": {
    "repo-rag": {
      "serverUrl": "http://127.0.0.1:8077/mcp"
    }
  }
}
```

Reload Antigravity and `repo-rag` shows up with two tools. Ask it something about the
repos and it calls `collections` to see what exists, then `search` to retrieve.

**Each Antigravity install reads its own config.** The language server is launched with
`--app_data_dir <name>` and reads `~/.gemini/<name>/mcp_config.json` — `antigravity` for
`Antigravity.app`, `antigravity-ide` for `Antigravity IDE.app`. Keep a real copy in each,
not a symlink; edit `~/.gemini/config/mcp_config.json` and fan it out:

```bash
for d in antigravity antigravity-ide; do
  cp ~/.gemini/config/mcp_config.json ~/.gemini/$d/mcp_config.json
done
```

If a new install appears and doesn't see the server, find which dir it reads:

```bash
ps -eo command= | grep language_server | tr ' ' '\n' | grep -A1 app_data_dir
```

**Config changes need the language server killed, not the app restarted.** It's a daemon
that outlives the window — closing or reopening Antigravity leaves the old process running
with the old config, so a new server silently never appears. Kill it and the IDE respawns
it on next interaction:

```bash
pkill -f language_server
```

Verify it actually attached — a `language_server` line here is the only real proof:

```bash
lsof -nP -iTCP:8077 | grep ESTABLISHED
```

**Connected is not the same as used.** Antigravity has its own workspace grep and file
reader, and will happily crawl directories instead of calling `search` — it answers the
question, just slowly and without the index. Two things fix that, and you want both:

- Tool descriptions that argue for themselves. The docstrings in `serve_mcp` say when to
  prefer `search` over grep, not just what it returns. They ship in the handshake, so
  **restart the server** after editing them.
- A rules file. Antigravity reads `AGENTS.md` / `GEMINI.md` hierarchically from the
  workspace root — it does *not* read `CLAUDE.md`. `~/code/AGENTS.md` tells it to reach
  for `repo-rag` first.

**Stop the server with Ctrl-C twice.** Antigravity holds a long-lived SSE stream, and
uvicorn's graceful shutdown closes the listener and then waits forever for that stream to
drain. The process stays alive holding the port, so the next start dies with
`[Errno 48] address already in use` while `lsof` shows no LISTEN socket. Second Ctrl-C
forces the exit; if it's already wedged, `kill -9` it.

HTTP rather than stdio on purpose: Antigravity connects to a URL instead of spawning the
process, so it never has to find `uv` on a GUI app's `PATH` — the usual way stdio MCP
servers fail to start. The cost is that you manage the process yourself; if the server
isn't running, the tools just don't appear. To hand that job to launchd instead of a
terminal window, see [SERVICE.md](SERVICE.md).

## Projects

A **project** is a named set of roots. One index, one server, one collection namespace —
the project is a filter, not a separate deployment, so cross-project search still works
when you want it:

```bash
uv run rag index -p work ~/repos ~/notes
uv run rag index -p side-projects ~/personal
uv run rag projects                        # who is in what
uv run rag index                           # no -p: sync every project
uv run rag search -p work "retry policy"
```

Re-running `index -p <name>` **replaces** that project's roots. Pruning is scoped: syncing
one project never drops another's collections.

### Picking repos out of a shared directory

A root takes *every* immediate subdirectory. When a directory holds repos belonging to
different projects — or repos you don't want indexed at all — narrow it with `--include`
globs, matched against the directory name:

```bash
uv run rag index -p platform ~/repos -i 'svc-*' -i 'handbook'
```

The globs are stored with the project, so a `svc-something-new` cloned later is picked
up on the next sync without touching the config. Quote them — the shell must not expand
them first. `rag projects` prints the globs in effect.

`--include` replaces the stored globs when given and leaves them alone when omitted, so a
plain `rag index` never silently widens a project.

**Two projects must not claim the same repository.** Both would discover it, one would
overwrite the other in the collection map, and the loser would return no results at all —
no error, just silence. Sharing a *root* is fine as long as the globs keep the repo sets
disjoint; `index` compares the resulting sets and refuses a real overlap, and `doctor`
flags any that predate the check.

Moving a root from one project to another costs a reindex: the old project's sync sees the
directories gone and prunes their collections before the new project can claim them. Split
first, then `rag index`.

### doctor

The state file, Qdrant and the disk can disagree, and no normal command looks at all three
— `sync` only prunes inside the projects it was asked to sync. `doctor` is that view:

```bash
uv run rag doctor          # exits 1 if anything is wrong
uv run rag doctor --fix    # repairs what is unambiguously safe
```

It checks for roots that no longer exist, collections in Qdrant that no project tracks,
state entries whose collection is gone, collections pointing at a deleted project, repos
on disk never indexed, wrong vector schema or dimension, missing descriptions, overlapping
projects, and a stale topology graph. Each finding prints the command that fixes it.

`--fix` only touches the three mechanical cases — deleting untracked collections, dropping
dead state entries, and clearing orphans. Everything it deletes is reindexable from disk.
It will not reindex, redescribe or rebuild the graph for you, because those cost real time.

### Removing things from the index

Three units, and only the last one needs a command:

| remove | how |
|---|---|
| a file | delete it on disk — `sync`/`watch` evicts its chunks by `path` filter |
| a repository | delete or move the directory out of the root — its whole collection is dropped on the next sync |
| a root | `index -p <name> <the roots you want to keep>` — roots are replaced, so the omitted one's repos are pruned |
| a whole project | `uv run rag forget <name>` |

```bash
uv run rag forget side-projects     # deletes its collections, descriptions and roots
```

Everything except `forget` is a side effect of reconciling against disk, so there is no
way to leave the index disagreeing with the files. `forget` exists because a project is
the one thing with no on-disk representation to reconcile against: `sync` only prunes
inside the projects it was asked to sync, so a project removed from `.rag.json` by hand
would strand its collections in Qdrant with nothing able to reach them. `forget` keys off
the collection→project map rather than the project list, so it cleans those up too.

There is deliberately no "remove one collection": the directory is still on disk, so the
next sync would just rebuild it. Move the directory, or split it into its own project.

`forget` also drops the stored graph — every edge touching a removed node is now wrong.
Re-run `rag graph` after.

The catalogue groups by project, and both MCP tools take an optional `project`, so an agent
working in one context doesn't have to read past ten irrelevant repos. Existing state
migrates on first run — the old flat root list becomes a project called `default`; rename
it by re-indexing under the name you want.

One sharp edge: collection names come from the directory name, so two projects with a repo
directory of the same name collide on one collection. There is a `ponytail:` note at the
collision point; namespace the slug per project if it ever bites.

## Topology

`rag graph` builds a graph over the collections from two signals already in the index —
no LLM, no second store, one scroll per collection (~30s over 33k chunks):

- **mentions** (directed, exact) — chunks in A that name B. A repo's docs pointing at
  another repo is a real dependency, and this is the signal worth trusting.
- **affinity** (undirected) — cosine between each collection's mean dense vector. A whole
  repo averaged to one point is blurry by construction: on this corpus everything scores
  0.58–0.93, so it is only ever read as a *ranking among one node's neighbours*, never as
  an absolute number. It exists to break ties where no mention edge exists.

```bash
uv run rag graph                  # print the edge table (cached if current)
uv run rag graph --rebuild        # recompute even if the cache is current
uv run rag graph --mermaid        # paste into any Markdown renderer
uv run rag graph --draw --open    # interactive page, default graph.html
```

`--draw` builds a NetworkX graph and renders it with [pyvis](https://pyvis.readthedocs.io)
— a Python API over vis.js, so physics, scaling and hover behaviour are configured in
Python, not hand-written JS. Node area is chunk count, colour is project, edge width is
cross-reference count. The page inlines its own copy of vis.js and makes no network
calls, so it opens offline.

## Topics

The per-repository graph can only say *which repos* reference each other. `rag topics`
asks the sharper question — **what the corpus is actually about, and which subjects cut
across repository boundaries**:

```bash
uv run rag topics                 # the topic table
uv run rag topics --draw --open   # render it, default topics.html
uv run rag topics -p platform # topics within one project
```

The pipeline is four steps and needs no LLM:

1. **Sample** chunk vectors evenly across collections (`--budget`, default 3000). Qdrant
   scrolls in point-id order and ids are a uuid5 of the file path, so the slice is
   arbitrary with respect to content.
2. **k-nearest-neighbour graph** over cosine similarity (`-k`, default 8; `--floor`,
   default 0.55) — one NumPy matmul, then NetworkX holds the graph.
3. **Louvain communities** (`nx.community.louvain_communities`, seeded for determinism).
   NetworkX ships it, so no extra dependency and no clustering library.
4. **Label** each community by tf-idf over the BM25 tokenizer already used for retrieval:
   the terms that are far more common inside the cluster than in the corpus.

Topic nodes are sized by how many sampled chunks they hold, coloured by the project that
dominates them, and **ringed and larger-labelled when they span more than one
repository** — those are where the corpus is genuinely joined. Hover one for its
per-repository breakdown.

> Not `regraph`. The PyPI package of that name is a graph-*rewriting* framework, last
> released in 2020; it needs `greenery<4`, `neo4j-driver==1.7`, and `networkx<2.4`, and
> that last one only imports on Python ≤3.8. It also ships no plotting module. The other
> ReGraph is Cambridge Intelligence's React SDK, which is commercially licensed.

Two things keep the picture legible, both measured rather than guessed. Communities
below `min_size` (3) are dropped: they are chunks nothing else resembles, they carry no
topic, and being disconnected they drag the spring layout apart and collapse everything
else into a blob. And each topic keeps only its 3 strongest links — at this `--floor`
nearly every topic touches every other, so the full edge set is a mesh no layout can
spread.

### Caching

Both derived views are cached in `.rag.json` and recomputed only when their inputs
change. Nothing to remember and no flag to pass — a stale answer is never served:

| | cold | cached |
|---|---|---|
| `graph` | 47s | 0.6s |
| `topics` | 4.7s | 0.7s |

A cache entry records the key it was computed from:

- **`revision`** — a counter `sync` bumps whenever it actually indexed or removed
  something. Chunk counts alone would miss an edit that re-chunks to the same number;
  `sync` is the only thing that touches the index, so a counter it bumps is both exact
  and free to read.
- **scope** — the collections in play, so cloning a repo or adding a project invalidates.
- **parameters** — `-k`, `--budget`, `--floor`, so a `-k 12` run is never served from a
  `-k 8` cache. Each parameter set occupies the one slot; alternating between two
  recomputes each time.

`forget` and `doctor --fix` drop both caches outright, since collections disappearing
underneath them makes every edge suspect. `--rebuild` forces recomputation; `doctor`
reports a stale graph by comparing the same key, so it catches content changes and not
just added or removed collections.

Expect some topics to be build artefacts rather than subjects — `sha512 / integrity /
tgz` is a lockfile, `c22 / c27 / c25` is minified CSS. They are honestly in the index,
so they are honestly in the graph; narrow `TEXT_EXT` if they bother you.

The payoff isn't the picture. The top four neighbours per collection land in the catalogue
as `· related: …`, which the agent reads in its system prompt — so when one repo
half-answers, the next search goes to the repos that cite it rather than to all eleven.

Measured on the current index, the mention graph recovers the real structure: the
service repo cites the shared-schema repo it depends on 53 times, a library and its
example app cite each other 44 times, and the docs repo cites the main service 27
times. The affinity half mostly says "these are all repos from the same org".

## Shape

**One Qdrant collection per repository.** Each collection carries a generated one-line
description of what the repo holds, and the agent sees that catalogue in its system prompt:

```
- `platform-config` (2431 chunks): YAML declarations of every environment,
  service and access group…
- `handbook` (1876 chunks): onboarding guides, runbooks and team
  process documentation…
```

So the agent routes *before* it retrieves — a question about deployment searches the
platform-config collection, not all five. That keeps recall high on a corpus where five
repos use overlapping vocabulary ("project", "dataset", "access") to mean different things.

Descriptions are written once at index time by the chat model from each repo's
`AGENTS.md` / `CLAUDE.md` / `README.md`, cached in `.rag.json`. Edit that file to
hand-tune a description, or `rag describe --force` to regenerate.

## Why "agentic"

Classic RAG does one vector search and stuffs the top-k into the prompt. Here the model
drives its own retrieval in a loop, with three tools:

| tool | what it's for |
|---|---|
| `search` | semantic search, scoped to chosen collections — concepts, "how do I…" |
| `grep` | exact match straight off disk — identifiers, YAML keys, config values |
| `read_file` | pull full context once a hit looks promising (chunks are fragments) |

So it can grep for a key, read the file it lives in, then search the docs repo for the
page that explains it — refining until it can answer. Tool calls are printed as it goes,
so the demo shows the reasoning path, not just the answer.

## Setup

```bash
docker compose up -d                     # Qdrant on :6333
ollama pull embeddinggemma:300m          # embeddings, 768-dim
ollama pull qwen2.5:14b                  # reasoning + tool calling
```

`uv` handles Python deps from `pyproject.toml` — no venv to manage.

## Use

```bash
uv run rag index -p work ~/repos ~/notes              # roots are remembered
uv run rag watch                                      # keep index synced with the files
uv run rag ask "how is a failed job retried?"
uv run rag collections                                # the catalogue the agent sees
uv run rag projects                                   # named root sets and their repos
uv run rag forget side-projects                       # drop a project and its collections
uv run rag doctor                                     # state vs Qdrant vs disk
uv run rag graph                                      # topology over the collections
uv run rag graph --draw --open                        # interactive topology page
uv run rag topics --draw --open                       # topic clusters across repos
uv run rag search -c handbook "on-call rotation"      # raw retrieval, no LLM
uv run rag mcp --http --watch                         # serve tools to Antigravity + stay synced
uv run rag mcp                                        # same tools over stdio
uv run rag selftest                                   # sync correctness check
uv run rag eval -k 8 --misses                         # retrieval quality on eval.jsonl
```

Each immediate subdirectory of a root is treated as one repository, unless the project
narrows that with `--include` globs.

## Keeping the index in sync

`sync()` reconciles each collection against its repo directory on every run, and is
idempotent:

- **stat fast path** — matching mtime+size, file is skipped without reading it
- **content hash** — file touched but bytes unchanged, only the payload is refreshed
- **changed** — old chunks deleted by `path` filter, then re-embedded and upserted
- **deleted** — present in Qdrant but gone from disk, evicted
- **repo removed** — its whole collection is dropped

`watch` runs the same reconcile, triggered by filesystem events with a 1s debounce so a
`git checkout` or save-all collapses into one pass. Because it re-reconciles rather than
replaying individual events, nothing drifts if events are missed while it wasn't running.
It re-discovers repos each pass, so cloning a new one creates its collection.

File hash and vectors live in the **same Qdrant point payload**, so there's no second
store to fall out of step.

## Demo script

1. `uv run rag watch` in one terminal.
2. `uv run rag collections` — show the catalogue the agent routes on.
3. Ask something one repo answers: `uv run rag ask "how is authentication configured?"`
   — note it searches only that collection.
4. Ask something spanning repos: `uv run rag ask "which repo do I change to add a new
   environment, and what do I add?"`
5. Edit a doc in one of the repos — watch the terminal reindex just that file.
6. Ask about the text you just added; the new answer comes back.

## Retrieval quality

`eval.jsonl` is 79 questions, each with the one file that should answer it, tagged by kind
(conceptual / lexical / literal / mixed). `rag eval` reports how often that file lands in
the top k, for dense, BM25 and hybrid, so a retrieval change can be judged instead of argued.

At k=8 on the current index:

| mode | r@1 | r@8 | mrr |
|---|---|---|---|
| dense | 0.54 | 0.84 | 0.65 |
| sparse (BM25) | 0.29 | 0.47 | 0.35 |
| hybrid | **0.57** | **0.86** | **0.67** |

Honest reading: hybrid is ahead of dense on every column, by two or three questions out of
79. It earns its place because BM25 is dramatically better on bare identifiers
(`connection_secret_id`, `owner_group_slug`), where dense alone returns topically-related
files from the wrong repo — the per-kind split shows it, `lexical` 0.80 → 0.90. Fusing at
equal weight is actively harmful (r@1 0.61 → 0.42 on the original 33): a 20-word question
matches too much mid-IDF prose.

Caveats worth knowing before trusting these numbers: 79 questions is still a smoke test
rather than a benchmark; one question is worth 0.013. Each question has exactly one accepted
file, so a genuinely good answer from a different file scores as a miss.

Questions carry a `src` field and `eval` breaks the score out by it. `human` is the original
hand-written 33; `synthetic` is 46 drafted by the chat model *from* the answer chunk. Those
were filtered — no question may name its own file, leak a path segment or quote the chunk —
and the split reads 0.85 human / 0.87 synthetic at k=8, so the residual easiness is about
one question. Still compare synthetic against synthetic when it matters.

Targets were also checked for byte-identical copies elsewhere in the corpus and dropped:
a shared set of config-as-code templates is synced into every content repo, so a question about one is
unscoreable when retrieval can legitimately return any of three copies.

### Depth, and why `search` returns a digest tail

Recall keeps climbing well past where the results used to stop:

| | r@1 | r@5 | r@8 | r@24 | r@50 | r@100 |
|---|---|---|---|---|---|---|
| hybrid | 0.54 | 0.81 | 0.85 | 0.92 | **0.96** | 0.96 |

The answer is nearly always retrieved — it just sat below the cut where nobody could see it.
So `search` now returns full text for the top `k` and every remaining hit out to rank 50 as
a one-line digest (`WIDE`, `FULL_CHARS`). Ranking is untouched; the caller gets the whole
0.96 pool for about 4k extra characters and does its own judging.

The case that prompted it: *"I need temporary elevated access for an hour to debug a
production issue"* returned ten near-identical config YAMLs, all matching on the bare
token `access:`, while the one how-to page that explained the process sat at rank 37 — and the
old `clip()` threw away everything past rank 10 before the agent ever saw it.

### Judging with the local model: measured, and rejected

The obvious next step was to have Ollama judge the hits before returning them. It was tried
and it loses:

Measured on the original 33, before the set was grown:

| | r@1 | r@5 | added latency |
|---|---|---|---|
| hybrid (baseline) | 0.55 | 0.82 | — |
| `llama3.2:3b` reranks 24 → 5 *(n=12)* | 0.50 → 0.17 | 0.92 → 0.25 | +3.1s |
| `qwen2.5:14b` reranks 24 → 5 *(n=12)* | 0.50 → 0.50 | 0.92 → 0.75 | +11s |
| `qwen2.5:14b` picks collections first *(n=20)* | 0.60 → 0.45 | 0.90 → 0.55 | +2.3s |
| per-collection/dir diversity cap, no LLM | 0.55 | 0.82 → 0.79 | 0s |

Every variant throws away recall, and the router throws away most of it: it excludes the
right repository outright, which no later stage can undo. A 14B model asked to rank 24
passages is simply a worse judge than the frontier model already sitting on the other end
of the MCP connection — so the digest tail hands that model the pool instead. `ask` keeps
the narrow result (`wide=0`); 42 extra paths only crowd a small model's context.

A purpose-built cross-encoder over the depth-50 pool is the one untried thing that could
still claim the +0.09 between r@8 and r@50. Ollama has no rerank endpoint, so it would mean
a new dependency or a second serving process.

## Shortcuts taken (it's a PoC)

- Fixed ~1200-char line-window chunks with 2-line overlap. Fine for Markdown and YAML;
  code would retrieve better split on function boundaries.
- Retrieval is hybrid: dense vectors fused with a BM25 sparse vector by reciprocal rank,
  BM25 weighted at `SPARSE_W` (0.2). Measured on `eval.jsonl`, not assumed — see below.
  No reranker: an LLM one was measured and made things worse, see above.
- Loose files sitting directly in a root (e.g. `notes/Decision.md`) belong to no repo and
  are not indexed. Move them into a repo, or give them their own directory.
- `watch` re-scans all roots on any event. ~4.5k files, so it's fast enough; scope the
  reconcile to changed paths if the corpus grows an order of magnitude.
- Descriptions are generated from one doc per repo, plus the top-level and `docs/` listings.
  The listings go *after* the document in the prompt, because a repo whose content lives in
  `docs/` otherwise gets described by its CONTRIBUTING-style `AGENTS.md` — `platform-docs`
  was catalogued as "guidelines for contributing" rather than the GCP platform docs, and the
  `collections` tool is read straight by the calling agent. If a description still lies,
  hand-edit `.rag.json`; `describe` only overwrites with `--force`.
- The graph is rebuilt only when you run `rag graph`; `watch` does not refresh it. It
  shifts slowly, so a weekly rebuild is enough.
- `watch` reads the project list once at startup. Adding a project needs a restart
  (`launchctl kickstart -k …`) — cloning a repo into an *existing* root does not.
- No auth on Qdrant. Local-only.
