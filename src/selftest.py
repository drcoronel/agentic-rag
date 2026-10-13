"""End-to-end self check. Needs Qdrant and Ollama up; namespaces and cleans up after itself."""
import uuid
from pathlib import Path

from qdrant_client import models as qm

from . import config
from .agent import search_hits, t_search
from .config import DIM
from .graph import (build_graph, collection_graph, draw_graph, graph_mermaid, tg_to_json,
                    topic_graph, topics)
from .qdrant import client, ensure, is_hybrid
from .state import (cache_key, cached, discover, load_state, rediscover, save_state, scope,
                    slug)
from .sync import doctor, forget, sync
from .text import chunk, sparse, tokens

def selftest():
    # PREFIX is namespaced for the run so a failure cannot touch the real index
    text = "\n".join(f"line {i} " + "x" * 40 for i in range(100))
    cs = chunk(text)
    assert cs[0][0] == 1 and cs[-1][1] == 100, cs
    assert all(s <= e for s, e, _ in cs) and len(cs) > 1, cs
    assert all(cs[i + 1][0] <= cs[i][1] for i in range(len(cs) - 1)), "chunks must overlap, not gap"
    assert chunk("") == [] and chunk("   \n  \n") == []

    assert {"allowdelete", "allow", "delete"} <= set(tokens("  allowDelete: true")), "camelCase"
    assert {"non-exposed", "non", "exposed"} <= set(tokens("zoneclass: non-exposed")), "hyphen"
    assert "zoneclass" in set(tokens("zoneclass: non-exposed")), "key must survive whole"
    assert sparse("a a b").values[0] > sparse("a b").values[0], "tf must saturate upward"

    config.PREFIX = "selftest" + uuid.uuid4().hex[:6] + "-"
    try:
        _selftest_store()
    finally:
        # Every collection this run made is PREFIX-namespaced, so sweeping the prefix
        # cleans up whatever a failed assert skipped past. Keyed on the prefix rather
        # than on the state file: an early failure means there is no state file yet.
        for c in client().get_collections().collections:
            if c.name.startswith(config.PREFIX):
                client().delete_collection(c.name)
    print("selftest ok")


def _selftest_store():
    """The half that needs Qdrant and Ollama. Split out so selftest() can wrap it in the
    cleanup that deletes this run's collections even when an assertion fails."""
    import tempfile


    # a pre-hybrid collection must be rebuilt, not left to fail on every upsert
    legacy = slug("legacy-schema")
    client().create_collection(legacy, vectors_config=qm.VectorParams(size=DIM, distance=qm.Distance.COSINE))
    ensure(legacy)
    assert is_hybrid(legacy), "ensure must migrate an old single-vector collection"
    client().delete_collection(legacy)

    with tempfile.TemporaryDirectory() as tmp:
        config.STATE = Path(tmp) / "state.json"                 # never touch the real .rag.json
        root, root2 = Path(tmp) / "p1", Path(tmp) / "p2"
        alpha, beta, delta = root / "alpha-repo", root / "beta-repo", root2 / "delta-repo"
        for d in (alpha, beta, delta):
            d.mkdir(parents=True)
        (alpha / "a.md").write_text("# Alpha\nthe alpha document about penguins\n"
                                    "see also beta-repo for the other half\n")
        (alpha / "conf.yaml").write_text("zoneclass: non-exposed\nallowDelete: true\n")
        (beta / "b.md").write_text("# Beta\nthe beta document about turbines\n")
        (delta / "d.md").write_text("# Delta\nthe delta document about accordions\n")

        projects = {"one": [str(root)], "two": [str(root2)]}
        ctx = rediscover({"projects": projects, "roots": [str(root), str(root2)],
                          "desc": {}, "graph": {}})
        A, B, D = slug("alpha-repo"), slug("beta-repo"), slug("delta-repo")
        assert set(ctx["repos"]) == {A, B, D}, ctx["repos"]
        assert ctx["proj"] == {A: "one", B: "one", D: "two"}, ctx["proj"]
        st = load_state()                  # context() persists this in real use
        st["projects"] = projects
        save_state(st)

        assert sync(ctx, quiet=True) == (4, 0)
        assert sync(ctx, quiet=True) == (0, 0), "second sync must be a no-op"
        assert all(ctx["desc"].get(n) for n in (A, B, D)), "sync must describe every collection"

        # syncing one project must not evict another project's collections
        one = rediscover({"projects": {"one": projects["one"]}, "roots": [str(root)],
                          "desc": ctx["desc"], "graph": {}})
        assert sync(one, quiet=True) == (0, 0)
        assert client().collection_exists(D), "project-scoped sync must not prune other projects"

        # project is a search filter, not a separate index
        assert scope(ctx, project="two") == [D]
        assert "accordions" not in t_search(ctx, "accordions", project="one")
        assert "accordions" in t_search(ctx, "accordions", project="two")

        # topology: a repo naming another is a directed edge, and ranks first as related
        g = build_graph(ctx, quiet=True)
        ab = next(e for e in g["edges"] if {e["a"], e["b"]} == {A, B})
        assert ab["mentions"][A] == 1 and ab["mentions"][B] == 0, ab
        assert g["related"][A][0] == B, g["related"]
        assert f"{A} -->|1|" not in graph_mermaid(g), "mermaid must use node ids, not names"
        assert "graph LR" in graph_mermaid(g) and "-->" in graph_mermaid(g)
        CG = collection_graph(g, ctx)
        assert set(CG) == set(g["nodes"]), "every collection must become a node"
        assert CG.has_edge(A, B) and CG[A][B]["weight"] == 1, "a mention must become an edge"
        page = Path(draw_graph(CG, Path(tmp) / "c", "test"))
        assert page.suffix == ".html" and page.stat().st_size > 20000, "pyvis must inline vis.js"
        body = page.read_text()
        assert all(n in body for n in g["nodes"]), "every collection must reach the page"
        assert "__TITLE__" not in body and "one" in body, "legend must be filled in"
        assert "jsdelivr" not in body and "unpkg" not in body, "page must not call out"

        # topics: clustered chunk vectors, labelled from the corpus, drawn the same way
        TG = topic_graph(ctx, k=2, budget=50, floor=.0, quiet=True)
        assert TG.number_of_nodes() >= 1, "clustering must yield at least one topic"
        assert sum(d["size"] for _, d in TG.nodes(data=True)) == 4, "every chunk lands once"
        assert all(d["label"] for _, d in TG.nodes(data=True)), "every topic needs a label"
        assert all(d["project"] in ("one", "two") for _, d in TG.nodes(data=True))
        assert Path(draw_graph(TG, Path(tmp) / "t", "test")).stat().st_size > 20000

        # caching: a second call is served from state, and survives the JSON round trip
        P = dict(k=2, budget=50, floor=.0)
        first = topics(ctx, quiet=True, **P)
        assert cached("topics", cache_key(ctx, **P)), "topics must be cached after a run"
        again = topics(ctx, quiet=True, **P)
        assert tg_to_json(again) == tg_to_json(first), "cache must round-trip exactly"
        assert not cached("topics", cache_key(ctx, k=3, budget=50, floor=.0)), \
            "a different -k must not be served from the cache"
        assert cached("graph", cache_key(ctx)), "graph must be cached after build_graph"

        # a consistent index must be clean; PREFIX scoping keeps the real one out of view
        assert doctor(ctx, quiet=True) == 0, "healthy index must report no problems"

        # include globs: one root, two projects, disjoint repo sets
        assert set(discover([str(root)], ["alpha-*"])) == {A}, "glob must filter by dir name"
        assert set(discover([str(root)], ["beta-repo"])) == {B}, "a literal is a valid glob"
        assert set(discover([str(root)], ["nothing-*"])) == set()
        assert set(discover([str(root)])) == {A, B}, "no globs means every subdirectory"
        assert set(discover([str(root) + "-does-not-exist"])) == set(), "missing root, no crash"
        split = rediscover({"projects": {"x": [str(root)], "y": [str(root)]},
                            "include": {"x": ["alpha-*"], "y": ["beta-*"]}})
        assert split["proj"] == {A: "x", B: "y"}, split["proj"]

        # a repo cloned while watch is running must get chunks AND a catalogue entry
        gamma = root / "gamma-repo"
        gamma.mkdir()
        (gamma / "c.md").write_text("# Gamma\nthe gamma document about telescopes\n")
        rediscover(ctx)
        rev = load_state().get("revision", 0)
        assert sync(ctx, quiet=True) == (1, 0)
        assert ctx["desc"].get(slug("gamma-repo")), "new repo must be described, not just indexed"
        # indexing a repo must invalidate both derived views, not serve a stale one
        assert load_state()["revision"] > rev, "a sync that changed the index must bump it"
        assert not cached("topics", cache_key(ctx, k=2, budget=50, floor=.0))
        assert not cached("graph", cache_key(ctx)), "new collection must invalidate the graph"

        # collections are isolated: a beta term must not surface when searching alpha
        assert "penguins" in t_search(ctx, "penguins", collections=[A])
        assert "turbines" not in t_search(ctx, "turbines", collections=[A])
        assert "turbines" in t_search(ctx, "turbines")           # ...but does across all

        # the lexical half: an exact identifier must rank first, which is the whole
        # reason for the sparse vector — dense alone blurs rare literals into their topic
        top = search_hits(ctx, "allowDelete", k=1, collections=[A], mode="sparse")
        assert top and top[0][0].payload["rel"].endswith("conf.yaml"), top
        assert "allowDelete" in t_search(ctx, "allowDelete", k=2, collections=[A])

        (alpha / "a.md").write_text("# Alpha\nrewritten to talk about submarines\n")
        assert sync(ctx, quiet=True) == (1, 0), "edit must reindex exactly one file"
        assert "submarines" in t_search(ctx, "submarines", collections=[A])
        assert "penguins" not in t_search(ctx, "penguins", k=2, collections=[A])

        (beta / "b.md").unlink()
        assert sync(ctx, quiet=True) == (0, 1), "delete must evict from the index"
        assert "turbines" not in t_search(ctx, "turbines", k=5)

        # forget must reach collections sync can no longer see. Project "two" is absent
        # from the state file's `projects` here, which is exactly the hand-edited shape
        # that used to orphan collections in Qdrant forever.
        st = load_state()
        st["projects"].pop("two")          # the hand-edit that used to strand collections
        save_state(st)
        rediscover(one)                    # ...it has gained gamma-repo since it was built
        assert sync(one, quiet=True) == (0, 0) and client().collection_exists(D), \
            "precondition: a scoped sync cannot reach another project's collections"
        assert doctor(one, quiet=True), "doctor must notice the orphan sync cannot see"
        assert forget("two", quiet=True) == [D]
        assert not client().collection_exists(D), "forget must delete the collections"
        st = load_state()
        assert D not in st["collections"] and D not in st["desc"], st
        assert "graph" not in st, "a stale graph must not outlive a removed node"
