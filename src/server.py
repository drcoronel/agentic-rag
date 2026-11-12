"""MCP server: retrieval exposed to an external agent harness.

Deliberately only two tools. The harness already has better grep and read_file than we
would ship; semantic search over the index is the thing it cannot do itself.
"""
import functools, inspect, json, os, re, sys, threading, time
from contextlib import nullcontext
from datetime import datetime, timezone

from .agent import catalogue, t_search
from .config import settings
from .sync import watch


_CITE = re.compile(r"^(?:--- )?\[([^\]]+)\] (\S+):(\d+)-\d+", re.M)
_log_lock = threading.Lock()


_mlflow = None
"""The mlflow module once `start_mlflow` has connected, else None (tracing off)."""


def start_mlflow():
    """Connect tracing to `settings.mlflow_uri`, if set.

    A tracking server that is down only costs the traces: this runs under an agent
    harness, and observability must never take the tools down with it.
    """
    global _mlflow
    if not settings.mlflow_uri:
        return
    # mlflow's default is ~7 retries with backoff, minutes of a stalled harness startup
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "5")
    try:
        import mlflow
        mlflow.set_tracking_uri(settings.mlflow_uri)
        mlflow.set_experiment(settings.mlflow_experiment)
        _mlflow = mlflow
    except Exception as e:
        print(f"mlflow tracing off: {settings.mlflow_uri} unreachable ({e})", file=sys.stderr)


def _span(name):
    if not _mlflow:
        return nullcontext()
    from mlflow.entities import SpanType
    return _mlflow.start_span(name=name, span_type=SpanType.TOOL)


def logged(fn):
    """Record every tool call: when, which tool, the arguments, how long it took, the
    cited hits in rank order, and the full result.

    Always to `settings.calls_log` as one JSON line; also as an MLflow trace when
    `start_mlflow` connected. `functools.wraps` keeps the signature and docstring the
    MCP schema is built from.
    """
    sig = inspect.signature(fn)

    @functools.wraps(fn)
    def wrapper(*a, **kw):
        bound = sig.bind(*a, **kw)
        bound.apply_defaults()
        with _span(fn.__name__) as span:
            t0, out, err = time.perf_counter(), None, None
            try:
                out = fn(*a, **kw)
                return out
            except Exception as e:
                err = repr(e)
                raise                       # the span records the error on the way out
            finally:
                hits = [{"collection": c, "path": p, "line": int(l)}
                        for c, p, l in _CITE.findall(out or "")]
                rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                       "tool": fn.__name__, "args": bound.arguments,
                       "ms": round((time.perf_counter() - t0) * 1000, 1),
                       "error": err, "hits": hits, "result": out}
                # 0600: the result carries chunk text, same as the index itself.
                with _log_lock:
                    fd = os.open(settings.calls_log,
                                 os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
                    with os.fdopen(fd, "a") as f:
                        f.write(json.dumps(rec, default=str) + "\n")
                if span:
                    span.set_inputs(bound.arguments)
                    span.set_outputs(out)
                    span.set_attributes({"hits": hits, "n_hits": len(hits)})
                    # trace tags are what the MLflow UI filters on: `tags.rank1 = 'x'`
                    _mlflow.update_current_trace(tags={
                        "tool": fn.__name__,
                        "rank1": hits[0]["collection"] if hits else "",
                        "collections_hit": ",".join(dict.fromkeys(h["collection"] for h in hits))})
    return wrapper


def serve_mcp(ctx, transport="stdio", port=8080, also_watch=False):
    """Serve the `collections` and `search` tools over stdio or streamable HTTP.

    With `also_watch`, one process both serves and keeps the index fresh. HTTP binds
    loopback only — these tools have no auth.
    """
    from mcp.server.mcpserver import MCPServer      # was FastMCP before mcp 2.x

    if also_watch:
        threading.Thread(target=watch, args=(ctx,), daemon=True).start()
    start_mlflow()

    m = MCPServer("repo-rag")

    @m.tool()
    @logged
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
    @logged
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
        m.run(transport=transport, host="127.0.0.1", port=port)
