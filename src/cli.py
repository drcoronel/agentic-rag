"""Command line surface.

Agentic RAG over local repos: Ollama embeds and answers, Qdrant stores. One collection
per repository, each with a generated description, so the agent picks which repos to
search before it searches them. Repositories are grouped into named *projects*.

    uv run rag index -p work ~/repos ~/notes
    uv run rag watch
    uv run rag ask "how is a failed job retried?"

`-p/--project` sits on each command rather than on the app callback, so `index -p work
<dirs>` reads in the order you think it. See docs/indexing.md for what the commands do
to the index, and docs/retrieval.md for the search knobs.
"""
import subprocess
import sys
from enum import Enum
from pathlib import Path
from typing import Annotated, Optional

import typer

from .agent import ask as run_ask
from .agent import catalogue, t_search
from .evaluate import evaluate
from .graph import (build_graph, collection_graph, draw_graph, graph_mermaid, graph_table,
                    topics)
from .selftest import selftest as run_selftest
from .sensitive import purge, scan
from .server import serve_mcp
from .state import cache_key, cached, context, describe as run_describe
from .sync import doctor as run_doctor
from .sync import forget as run_forget
from .sync import sync, watch as run_watch


app = typer.Typer(add_completion=False, no_args_is_help=True, rich_markup_mode="rich",
                  help=__doc__)

EVAL_FILE = Path(__file__).parent.parent / "eval.jsonl"


class Mode(str, Enum):
    dense = "dense"
    sparse = "sparse"
    hybrid = "hybrid"


Project = Annotated[Optional[str], typer.Option(
    "-p", "--project", help="Limit to one named project. Default: all.")]
Include = Annotated[Optional[list[str]], typer.Option(
    "-i", "--include", metavar="GLOB",
    help="Only take repo directories matching this glob (repeatable). Lets one root hold "
         "repos belonging to different projects.")]
Roots = Annotated[Optional[list[Path]], typer.Argument(
    help="Directories whose immediate subdirectories are the repositories.")]
Rebuild = Annotated[bool, typer.Option("--rebuild", help="Recompute even if the cache is current.")]
Open = Annotated[bool, typer.Option("--open", help="Open the drawn file.")]


def _wrote(path: Path, open_it: bool = False) -> None:
    print(f"wrote {Path(path).resolve()}")
    if open_it:
        subprocess.run(["open", str(Path(path).resolve())])


Draw = Annotated[bool, typer.Option("--draw", help="Render an interactive page with pyvis.")]


def _out_option(default_name: str):
    """`--out` names the file `--draw` writes. Typer has no optional-value option, so the
    filename is a separate flag rather than `--draw [FILE]`."""
    return typer.Option("--out", metavar="FILE", help="Where --draw writes.")


@app.command()
def index(roots: Roots = None, project: Project = None, include: Include = None):
    """Index repositories, or re-sync the ones already known."""
    sync(context(project, roots or None, include or None))


@app.command()
def watch(roots: Roots = None, project: Project = None, include: Include = None):
    """Keep the index in sync with the files until interrupted."""
    run_watch(context(project, roots or None, include or None))


@app.command()
def search(
    query: Annotated[list[str], typer.Argument(help="Words to search for.")],
    project: Project = None,
    collection: Annotated[Optional[list[str]], typer.Option(
        "-c", "--collection", help="Restrict to these collections (repeatable).")] = None,
    k: Annotated[int, typer.Option("-k", "--top-k", help="Hits returned in full.")] = 8,
):
    """Search the index and print the matching chunks."""
    print(t_search(context(project), " ".join(query), k=k, collections=collection,
                   project=project))


@app.command()
def ask(question: Annotated[list[str], typer.Argument(help="The question to answer.")],
        project: Project = None):
    """Answer a question with the local model, using the index as its tools."""
    print("\n" + run_ask(context(project), " ".join(question)) + "\n")


@app.command()
def describe(project: Project = None,
             force: Annotated[bool, typer.Option("--force", help="Re-describe repos that already have one.")] = False):
    """Generate the one-line catalogue description for each repository."""
    run_describe(context(project), force=force)


@app.command()
def collections(project: Project = None):
    """Print the catalogue the agent sees: every repo, its description and size."""
    print(catalogue(context(project)))


@app.command()
def projects():
    """List the named root sets and the repositories they resolve to."""
    ctx = context()
    for p, roots in sorted(ctx["projects"].items()):
        names = [n for n in ctx["repos"] if ctx["proj"][n] == p]
        globs = ctx["include"].get(p)
        print(f"{p}: {len(names)} repos in {', '.join(roots)}"
              + (f" matching {', '.join(globs)}" if globs else "")
              + "\n  " + "\n  ".join(names))


@app.command()
def forget(project: Annotated[str, typer.Argument(help="The project to drop.")]):
    """Drop a project: its collections, their descriptions, and its root set."""
    # Deliberately not routed through context(): the project may already be half-gone
    # from the state file, which is the case forget() exists to clean up.
    run_forget(project)


@app.command()
def doctor(fix: Annotated[bool, typer.Option("--fix", help="Repair what is safe to repair.")] = False,
           project: Project = None):
    """Check the state file, Qdrant and disk against each other."""
    problems = run_doctor(context(project), fix)
    raise typer.Exit(1 if problems and not fix else 0)


@app.command()
def graph(
    project: Project = None,
    mermaid: Annotated[bool, typer.Option("--mermaid", help="Print a Mermaid diagram instead.")] = False,
    draw: Draw = False,
    out: Annotated[Path, _out_option("graph.html")] = Path("graph.html"),
    open_it: Open = False,
    rebuild: Rebuild = False,
):
    """Topology over the collections, from cross-references and vector affinity."""
    ctx = context(project)
    g = None if rebuild else cached("graph", cache_key(ctx))
    if g:
        print(f"(cached at revision {g['key']['revision']} — --rebuild to recompute)",
              file=sys.stderr)
    else:
        g = build_graph(ctx)
    if draw:
        _wrote(draw_graph(collection_graph(g, ctx), out.expanduser(),
                          f"repo-rag — {len(g['nodes'])} collections, "
                          "edge width = cross-references"), open_it)
    else:
        print(graph_mermaid(g) if mermaid else graph_table(g))


@app.command("topics")
def topics_(
    project: Project = None,
    draw: Draw = False,
    out: Annotated[Path, _out_option("topics.html")] = Path("topics.html"),
    open_it: Open = False,
    k: Annotated[int, typer.Option("-k", "--neighbours", help="Neighbours per chunk.")] = 8,
    budget: Annotated[int, typer.Option("--budget", help="Chunks to sample.")] = 3000,
    floor: Annotated[float, typer.Option("--floor", help="Minimum cosine for a link.")] = .55,
    rebuild: Rebuild = False,
):
    """Cluster the chunks into topics and graph how the topics connect."""
    ctx = context(project)
    TG = topics(ctx, rebuild=rebuild, k=k, budget=budget, floor=floor, project=project)
    if draw:
        cross = sum(1 for _, d in TG.nodes(data=True) if d["spread"] > 1)
        _wrote(draw_graph(TG, out.expanduser(),
                          f"repo-rag — {TG.number_of_nodes()} topics, {cross} spanning "
                          "more than one repository (bold)"), open_it)


@app.command("eval")
def eval_(
    file: Annotated[Path, typer.Argument(help="A .jsonl of {q, path, kind}.")] = EVAL_FILE,
    project: Project = None,
    k: Annotated[int, typer.Option("-k", "--top-k", help="Cut-off for r@k.")] = 5,
    mode: Annotated[Optional[list[Mode]], typer.Option("-m", "--mode", help="Repeatable.")] = None,
    misses: Annotated[bool, typer.Option("--misses", help="List the questions that found nothing.")] = False,
):
    """Retrieval quality: rank of the expected file for each question."""
    modes = tuple(m.value for m in mode) if mode else ("dense", "sparse", "hybrid")
    evaluate(context(project), file, k, modes, misses)


@app.command()
def mcp(
    project: Project = None,
    http: Annotated[bool, typer.Option("--http", help="Streamable HTTP instead of stdio.")] = False,
    port: Annotated[int, typer.Option("--port")] = 8080,
    watch_too: Annotated[bool, typer.Option("--watch", help="Also keep the index synced.")] = False,
):
    """Serve retrieval as MCP tools for an agent harness."""
    serve_mcp(context(project), "streamable-http" if http else "stdio", port, watch_too)


@app.command()
def secrets(
    project: Project = None,
    purge_them: Annotated[bool, typer.Option(
        "--purge", help="Delete the matching points from Qdrant now.")] = False,
):
    """Report indexed files that look like they hold credentials.

    The denylist is applied at index time, so a normal `rag index` already evicts these
    (they stop appearing on disk from the indexer's point of view, and sync prunes what
    it cannot see). `--purge` deletes them immediately instead of waiting for a sync.
    """
    ctx = context(project)
    found = scan(ctx, project)
    if not found:
        print("no indexed file matches the secret denylist")
        raise typer.Exit(0)

    chunks = sum(n for hits in found.values() for n, _ in hits.values())
    files = sum(len(h) for h in found.values())
    print(f"{files} file(s), {chunks} chunk(s) in the index look sensitive:\n")
    for name, hits in sorted(found.items()):
        print(f"{name}:")
        for path, (n, why) in sorted(hits.items()):
            print(f"    {path}  ({n} chunks, {why})")
    if purge_them:
        print()
        gone = purge(found)
        print(f"\npurged {gone} chunks. Re-run `rag index` to confirm nothing returns.")
    else:
        print("\nThe chunk text is stored in Qdrant, so treat these as disclosed and "
              "rotate them.\nRun `rag secrets --purge`, or `rag index`, to remove them.")
    raise typer.Exit(1)


@app.command()
def selftest(offline: Annotated[bool, typer.Option(
        "--offline", help="Only the checks that need no running service.")] = False):
    """Run the self check. Needs Qdrant and Ollama up unless --offline."""
    run_selftest(offline)


def main():
    app()
