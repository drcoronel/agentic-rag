"""MCP server — retrieval exposed to an external agent harness."""
import threading

from .agent import catalogue, t_search
from .sync import watch

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
