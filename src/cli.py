"""Agentic RAG over local repos. Ollama embeds + answers, Qdrant stores.

One Qdrant collection per repository, each with a generated description so the
agent can pick which repos to search before it searches them. Repositories are
grouped into named *projects* (one root set each) so a session can scope to one.

  uv run rag index -p work ~/repos ~/notes
  uv run rag watch          # keeps the index in sync with the files
  uv run rag graph          # topology over the collections
  uv run rag ask "how is a failed job retried?"
"""
import argparse, subprocess, sys
from pathlib import Path

from .agent import ask, catalogue, t_search
from .evaluate import evaluate
from .graph import (build_graph, collection_graph, draw_graph,
                    graph_mermaid, graph_table, topics)
from .selftest import selftest
from .server import serve_mcp
from .state import cache_key, cached, context, describe
from .sync import doctor, forget, sync, watch

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
    p.add_argument("file", nargs="?", default=str(Path(__file__).parent.parent / "eval.jsonl"))
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
