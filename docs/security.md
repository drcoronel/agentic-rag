# Security

This indexes whatever you point it at and makes it searchable by a language model. The
two things that go wrong are **indexing something that should never have been read**,
and **exposing the index to something that should not reach it**. Everything below is
one of those two.

## Where your data goes

| stage | component | leaves the machine? |
|---|---|---|
| file read | `sync.walk()` → `sync_repo()` | no |
| embedding | Ollama on `OLLAMA_HOST` | no, unless `RAG_VERTEX_EMBED` is set |
| embedding | Vertex, when `RAG_VERTEX_EMBED` is set | **yes** — raw chunk text is POSTed to Google |
| storage | Qdrant on `QDRANT_URL` | no, with the default loopback binding |
| `rag ask` | local chat model via Ollama | no |
| `rag mcp` | the calling harness's model | **yes** — `search` returns chunk text to whatever is driving the tools |

Two of those are easy to forget. Setting `RAG_VERTEX_EMBED` turns every indexed byte
into an outbound request. And the MCP server exists to hand chunk text to an external
agent — if that agent is a hosted frontier model, your repository contents are going to
it, one search result at a time. "Fully local" is shape 1 only.

`grep` and `read_file` are **not** exposed over MCP; only `collections` and `search`
are. Those two tools are reachable solely by the local `ask` loop.

## The secret guardrail

Indexing is an allowlist on file extension. That alone meant `.env` and `private.key`
were skipped only incidentally — nothing was looking for secrets — while
`terraform.tfvars`, `secrets.yaml`, `service-account.json` and `.npmrc` were indexed
like any other file and stored in full.

Two layers now sit in front of it, in `sensitive.py`:

- **`is_secret(name)`** — a denylist over the filename, checked *before* the extension
  allowlist. `settings.secret_globs` covers env files, private keys and keystores, the
  `*secret*` / `*credential*` / `*password*` name hints, token-bearing tool config
  (`.netrc`, `.pgpass`, `.npmrc`, `kubeconfig`), `*.tfvars`, and service-account keys.
- **`secret_in(text)`** — a denylist over the *contents*. Vendor-shaped key material
  (PEM private keys, `AKIA…`, `ghp_…`, `glpat-…`, `xox[baprs]-…`, `AIza…`, `sk-…`,
  JWTs) keeps the whole file out whatever it is called.

Both are enforced at every point a file can reach the model, because they are different
code paths with different inputs:

| path | guard | why it needs its own |
|---|---|---|
| `sync.walk()` | name, before the file is opened | the index |
| `sync_repo()` | contents, after read, before chunking | catches a key in a normal-looking file |
| `agent.t_grep()` | name; matching lines dropped, count reported | shells out to ripgrep, never consults the index |
| `agent.t_read_file()` | name, **and** containment in a root | takes an arbitrary path from the model |

A file rejected on contents is also left out of `on_disk`, so a copy indexed before the
check existed is pruned on the same pass.

### Tuning

`secret_allow` is checked first, so committed placeholders (`.env.example`,
`*.template`) stay indexable — and so does prose. `*.md`, `*.mdx`, `*.rst` and `*.txt`
are exempt from the *name* hints, because those hints are what catch documentation
about credentials: a `sealedsecrets-create.md` how-to is not a key. Nothing is lost by
that exemption, since name matching never looked inside the file anyway. The hard
patterns cannot be prose, so they are unaffected. Content matching still applies to
prose in full.

`RAG_TRACKED_ONLY=1` narrows `walk()` to what `git ls-files` reports. That inherits
each repository's own `.gitignore`, which is a better statement of "never meant to be
read" than any list written here. It is off by default because it also stops indexing
untracked scratch notes. Non-git directories are unaffected.

### What it does not catch

A hand-rolled secret with no recognisable shape, in a conventionally-named file.
`DB_PASS: hunter2` in `config.yaml` matches nothing and is indexed. A generic
`password:\s*"..."` rule was considered and rejected: it fires on every placeholder in
every how-to, and a skipped document is a silent hole in the index that nobody notices,
which is a worse failure than the one it prevents.

Treat this as a guardrail against the routine cases, not as a scanner. If you need real
coverage, run a dedicated secret scanner over the repositories themselves.

## Finding and removing what is already indexed

The guards apply going forward. `rag secrets` checks what is already stored, against
both layers — the payload path *and* the payload text, so it also finds key material in
normally-named files indexed before any of this existed:

```bash
uv run rag secrets            # report; exits 1 if anything matches
uv run rag secrets --purge    # delete those points now
```

A plain `rag index` also evicts them, since a denied file stops appearing in `walk()`
and `sync` prunes what it can no longer see. `--purge` just does it immediately.

**Purging removes the stored copy. It does not undo a disclosure.** Anything found was
embedded and stored in plaintext, and if `RAG_VERTEX_EMBED` or an MCP harness was in
play it also left the machine. Rotate it.

## Network exposure

**Qdrant has no authentication.** `compose.yaml` therefore maps
`127.0.0.1:6333:6333`. The ordinary Docker form, `6333:6333`, binds `0.0.0.0` — on a
café, hotel or office network that makes every chunk of every indexed repository
readable by anyone who can route to your machine. Running the tool "only locally" is no
protection when the listener is not local. Check it:

```bash
lsof -nP -iTCP:6333 -sTCP:LISTEN     # want 127.0.0.1:6333, not *:6333
```

`QDRANT_API_KEY` exists for a managed cluster and is sent on every request when set. It
does nothing for the local container, which has no auth to key against.

`rag mcp --http` binds `127.0.0.1` and has **no auth** — any local process can call the
tools. That is acceptable for a single-user laptop and not acceptable anywhere else.
Do not put it on `0.0.0.0` without putting something in front of it.

## Data at rest

Qdrant stores the **full text** of every chunk in the point payload, unencrypted — the
index is a readable copy of your corpus, not just vectors. Anything that can read the
Qdrant volume or reach its port has your repositories.

Three files in the working directory also carry corpus content and are gitignored for
that reason:

| file | holds |
|---|---|
| `.rag.json` | absolute repo paths, and a generated description of each repository |
| `graph.html` | repository names and their cross-reference counts |
| `topics.html` | topic labels mined from chunk text, plus per-repository breakdowns |

The two HTML pages are standalone and open with no network, which also makes them easy
to send to someone by accident. They are derived from your corpus; treat them as such.

## Operator checklist

```bash
uv run rag selftest --offline    # guards behave; no services needed, good for CI
uv run rag secrets               # what the index holds that it should not
lsof -nP -iTCP:6333 -sTCP:LISTEN # loopback only
```

- Run `rag secrets` after the first index of a new root, and periodically after.
- Rotate anything it finds. Purging is not rotation.
- Consider `RAG_TRACKED_ONLY=1` if every root is a git repository.
- Keep `compose.yaml` on `127.0.0.1`.
- Remember that `rag mcp` sends chunk text to whatever model is driving it.
