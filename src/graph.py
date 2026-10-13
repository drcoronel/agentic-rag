"""Derived views over the index: the repository topology and the topic clustering."""
import math, re, sys
from collections import Counter
from pathlib import Path

from .config import DIM
from .qdrant import client
from .state import cache_key, cached, scope, store
from .text import tokens

def build_graph(ctx, quiet=False):
    """A graph over the collections, from two signals already present in the index:

    - **mentions** (directed, exact): chunks in A that name B. This is the sharp edge —
      a repo's docs pointing at another repo is a real dependency, not a guess.
    - **affinity** (undirected): cosine between the mean dense vector of each collection.
      Blurry by construction — a whole repo averaged to one point — so it is only ever
      read as a ranking among a node's own neighbours, never as an absolute number.

    No LLM and no second store: one scroll per collection over Qdrant.
    """
    names = sorted(ctx["repos"])
    alias = {n: {n.lower(), Path(ctx["repos"][n]).name.lower()} for n in names}
    cent = {n: [0.0] * DIM for n in names}
    ment = {n: Counter() for n in names}
    sizes = {}

    for n in names:
        offset, seen = None, 0
        while True:
            pts, offset = client().scroll(n, limit=512, offset=offset,
                                          with_payload=["text"], with_vectors=["dense"])
            for pt in pts:
                seen += 1
                c, v = cent[n], pt.vector["dense"]
                for i, x in enumerate(v):
                    c[i] += x
                t = pt.payload["text"].lower()
                for m in names:
                    if m != n and any(a in t for a in alias[m]):
                        ment[n][m] += 1
            if offset is None:
                break
        sizes[n] = seen
        if not quiet:
            print(f"  {n}: {seen} chunks, {len(ment[n])} outbound", file=sys.stderr)
        norm = math.sqrt(sum(x * x for x in cent[n])) or 1.0
        cent[n] = [x / norm for x in cent[n]]

    edges = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            edges.append({"a": a, "b": b,
                          "sim": round(sum(x * y for x, y in zip(cent[a], cent[b])), 4),
                          "mentions": {a: ment[a][b], b: ment[b][a]}})
    edges.sort(key=lambda e: -e["sim"])

    related = {}
    for n in names:
        nb = [(e, e["b"] if e["a"] == n else e["a"]) for e in edges if n in (e["a"], e["b"])]
        # mentions first (hard evidence), affinity only to fill the remaining slots
        ranked = (sorted((x for x in nb if sum(x[0]["mentions"].values())),
                         key=lambda x: -sum(x[0]["mentions"].values()))
                  + sorted(nb, key=lambda x: -x[0]["sim"]))
        related[n] = list(dict.fromkeys(o for _, o in ranked))[:4]

    g = store("graph", cache_key(ctx),
              {"nodes": {n: {"chunks": sizes[n], "project": ctx["proj"][n]} for n in names},
               "edges": edges, "related": related})
    ctx["graph"] = g
    return g


def graph_table(g, top=25):
    rows = sorted(g["edges"], key=lambda e: (-sum(e["mentions"].values()), -e["sim"]))[:top]
    w = max((len(n) for n in g["nodes"]), default=10) + 2
    out = [f"{len(g['nodes'])} collections, {len(g['edges'])} pairs "
           f"(top {len(rows)} by mentions, then affinity)\n",
           f"{'':<{w}}{'':<{w}}{'a→b':>6}{'b→a':>6}{'affinity':>10}"]
    for e in rows:
        out.append(f"{e['a']:<{w}}{e['b']:<{w}}{e['mentions'][e['a']]:>6}"
                   f"{e['mentions'][e['b']]:>6}{e['sim']:>10.3f}")
    return "\n".join(out)


def graph_mermaid(g, sim_top=10):
    """Mentions as labelled arrows; the strongest affinity pairs that have no mention
    edge as dotted lines — otherwise the affinity half is a complete graph and unreadable."""
    ids, out = {}, ["graph LR"]
    nid = lambda n: ids.setdefault(n, f"n{len(ids)}")
    for n, meta in g["nodes"].items():
        out.append(f'  {nid(n)}["{n}<br/><i>{meta["project"]}</i>"]')
    quiet = []
    for e in g["edges"]:
        if sum(e["mentions"].values()):
            for src, dst in ((e["a"], e["b"]), (e["b"], e["a"])):
                if e["mentions"][src]:
                    out.append(f'  {nid(src)} -->|{e["mentions"][src]}| {nid(dst)}')
        else:
            quiet.append(e)
    for e in sorted(quiet, key=lambda e: -e["sim"])[:sim_top]:
        out.append(f'  {nid(e["a"])} -. "{e["sim"]:.2f}" .- {nid(e["b"])}')
    return "\n".join(out)


# ---------------------------------------------------------------- topics

STOP = set("the a an and or of to in for is are be it this that with as on at by from "
           "if then else not no yes you we they can will should must may use used using "
           "see https http com www md yaml json true false null none name type value "
           "run all any new set get add https org io dev test".split())


def sample_chunks(ctx, budget=3000, project=None):
    """(vectors, texts, collections) spread evenly over the collections.

    Qdrant scrolls in point-id order, and ids are uuid5 of the file path, so taking the
    first slice of each collection is arbitrary with respect to content — good enough for
    a shape, and far cheaper than reading every chunk back out.
    """
    import numpy as np
    names = scope(ctx, project=project)
    per = max(1, budget // max(1, len(names)))
    V, T, C = [], [], []
    for n in names:
        offset = None
        while len(T) < per * (names.index(n) + 1):
            pts, offset = client().scroll(n, limit=256, offset=offset,
                                          with_payload=["text"], with_vectors=["dense"])
            for p in pts:
                V.append(p.vector["dense"])
                T.append(p.payload["text"])
                C.append(n)
            if offset is None or not pts:
                break
    return np.asarray(V, dtype="float32"), T, C


def label_topic(members, T, df, total, n_terms=3):
    """Top distinctive terms for one cluster: how much more often a term appears in this
    cluster than in the corpus. Plain tf-idf over the BM25 tokenizer already in use."""
    here = Counter()
    for i in members:
        here.update(set(tokens(T[i])) - STOP)      # presence per chunk, not repetition
    scored = [(c / len(members) * math.log(total / (1 + df[t])), t)
              for t, c in here.items() if c >= max(2, len(members) * .15) and len(t) > 2]
    scored.sort(reverse=True)
    return " / ".join(t for _, t in scored[:n_terms]) or "(no distinctive terms)"


def tg_to_json(TG):
    return {"nodes": [[n, d] for n, d in TG.nodes(data=True)],
            "edges": [[a, b, d["weight"]] for a, b, d in TG.edges(data=True)]}


def tg_from_json(j):
    import networkx as nx
    TG = nx.Graph()
    for n, d in j["nodes"]:
        TG.add_node(n, **d)
    for a, b, w in j["edges"]:
        TG.add_edge(a, b, weight=w)
    return TG


def topics(ctx, rebuild=False, quiet=False, **params):
    """Cached topic graph. Clustering is the expensive half — a scroll of every sampled
    chunk plus an n² similarity matrix — and it only changes when the index does."""
    key = cache_key(ctx, project=params.get("project"),
                    **{p: params.get(p) for p in ("k", "budget", "floor")})
    hit = None if rebuild else cached("topics", key)
    if hit:
        TG = tg_from_json(hit)
        if not quiet:
            print_topics(TG, len(hit["nodes"]), cached_from=key["revision"])
        return TG
    TG = topic_graph(ctx, quiet=quiet, **params)
    store("topics", key, tg_to_json(TG))
    return TG


def print_topics(TG, n_topics, cached_from=None):
    head = f"{TG.number_of_nodes()} topics, {TG.number_of_edges()} links"
    print(head + (f"  (cached at revision {cached_from} — --rebuild to recompute)\n"
                  if cached_from is not None else "\n"))
    print(f"{'#':>3}{'chunks':>8}{'repos':>7}  {'topic':<44}dominant repo")
    for t, d in sorted(TG.nodes(data=True), key=lambda x: -x[1]["size"]):
        print(f"{t:>3}{d['size']:>8}{d['spread']:>7}  {d['label'][:42]:<44}{d['top']}")


def topic_graph(ctx, k=8, budget=3000, floor=.55, project=None, quiet=False,
                min_size=3, keep_links=3):
    """Cluster the chunk vectors into topics, then graph how the topics connect.

    Nodes are topics rather than repositories: a k-nearest-neighbour graph over sampled
    chunk vectors, cut into communities by Louvain (NetworkX ships it, so no extra
    dependency and no LLM). The interesting ones are the topics that span repositories —
    those are the places the corpus is really joined, which a per-repo graph cannot show.
    """
    import numpy as np, networkx as nx

    V, T, C = sample_chunks(ctx, budget, project)
    if len(T) < k + 2:
        sys.exit(f"only {len(T)} chunks sampled — index something first")
    V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-9
    S = V @ V.T
    np.fill_diagonal(S, -1.0)

    kk = min(k, len(T) - 1)
    nn = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
    G = nx.Graph()
    G.add_nodes_from(range(len(T)))
    for i, row in enumerate(nn):
        for j in row:
            if S[i, j] >= floor:
                G.add_edge(int(i), int(j), weight=float(S[i, j]))

    comms = sorted(nx.community.louvain_communities(G, weight="weight", seed=0),
                   key=len, reverse=True)
    # Singletons are chunks nothing else resembles. They carry no topic, and because they
    # are disconnected the spring layout flings them to the corners and collapses the rest
    # into an unreadable blob — so they are dropped, not drawn.
    dropped = sum(1 for c in comms if len(c) < min_size)
    comms = [c for c in comms if len(c) >= min_size]
    where = {i: t for t, c in enumerate(comms) for i in c}

    df = Counter()
    for text in T:
        df.update(set(tokens(text)))

    TG = nx.Graph()
    for t, members in enumerate(comms):
        cols = Counter(C[i] for i in members)
        TG.add_node(t, size=len(members), label=label_topic(members, T, df, len(T)),
                    collections=dict(cols), top=cols.most_common(1)[0][0],
                    project=ctx["proj"][cols.most_common(1)[0][0]], spread=len(cols))
    for u, v in G.edges():
        a, b = where.get(u), where.get(v)
        if a is not None and b is not None and a != b:
            TG.add_edge(a, b, weight=TG.get_edge_data(a, b, {}).get("weight", 0) + 1)

    # Every topic links to nearly every other, so the full edge set is a hairball that no
    # layout can spread. Keep each topic's strongest few links — the backbone, not the mesh.
    keep = {tuple(sorted((n, m))) for n in TG
            for m, _ in sorted(TG[n].items(), key=lambda kv: -kv[1]["weight"])[:keep_links]}
    TG.remove_edges_from([e for e in list(TG.edges()) if tuple(sorted(e)) not in keep])

    if not quiet:
        print(f"{len(T)} chunks sampled from {len(set(C))} collections → ", end="")
        print_topics(TG, len(comms))
        if dropped:
            print(f"\n({dropped} singleton topics dropped)")
    return TG


PALETTE = ["#2d9c8f", "#4a7fd4", "#d98324", "#9b59b6", "#d1495b", "#5c8001"]

LEGEND = """<div style="position:fixed;top:12px;left:12px;padding:10px 14px;z-index:9;
 border-radius:8px;background:rgba(255,255,255,.93);box-shadow:0 1px 6px rgba(0,0,0,.18);
 font:13px/1.5 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif">
 <b>__TITLE__</b><br>__KEYS__
 <div style="opacity:.6;font-size:11px;margin-top:4px">drag to pin · hover for detail</div>
</div>"""


def draw_graph(G, path, title, label_key="label"):
    """Interactive page via pyvis — a Python API over vis.js, so the physics, scaling and
    hover behaviour are configured here in Python rather than in hand-written JS.

    `cdn_resources="in_line"` inlines the library, so the file is standalone and opens
    with no network. It is one HTML file with no server and nothing installed.
    """
    from pyvis.network import Network

    projects = sorted({d["project"] for _, d in G.nodes(data=True)})
    color = dict(zip(projects, PALETTE * (len(projects) // len(PALETTE) + 1)))
    path = Path(path).with_suffix(".html")

    net = Network(height="100vh", width="100%", directed=False, bgcolor="#ffffff",
                  font_color="#222", cdn_resources="in_line", notebook=False)
    for n, d in G.nodes(data=True):
        spread = d.get("spread", 1)
        detail = "<br>".join(f"{k}: {v}" for k, v in sorted(
            d.get("collections", {}).items(), key=lambda kv: -kv[1]))
        net.add_node(
            str(n), label=d[label_key], value=d["size"], color=color[d["project"]],
            borderWidth=3 if spread > 1 else 1,
            title=f"{d[label_key]}\n{d['size']} chunks · {d['project']}"
                  + (f"\nspans {spread} repositories" if spread > 1 else "")
                  + (f"\n\n{detail}" if detail else ""),
            font={"size": 16 if spread > 1 else 13,
                  "face": "-apple-system", "strokeWidth": 4, "strokeColor": "#fff"})
    for a, b, d in G.edges(data=True):
        net.add_edge(str(a), str(b), value=d["weight"], title=str(d["weight"]))

    net.set_options("""{
      "nodes": {"shape": "dot", "scaling": {"min": 8, "max": 52},
                "color": {"border": "#3338"}},
      "edges": {"color": {"color": "#999", "opacity": 0.4},
                "scaling": {"min": 1, "max": 10}, "smooth": {"type": "continuous"}},
      "physics": {"barnesHut": {"gravitationalConstant": -14000, "springLength": 220,
                                "avoidOverlap": 0.55},
                  "stabilization": {"iterations": 500}},
      "interaction": {"hover": true, "tooltipDelay": 120}
    }""")

    net.write_html(str(path), notebook=False, open_browser=False)
    keys = " ".join(
        f'<span style="display:inline-block;width:10px;height:10px;border-radius:2px;'
        f'background:{color[p]};margin:0 5px 0 10px"></span>{p}' for p in projects)
    html = path.read_text().replace(
        "</body>", LEGEND.replace("__TITLE__", title).replace("__KEYS__", keys) + "</body>")
    # pyvis's template still pulls Bootstrap off a CDN for filter/config panels we never
    # enable. Dropping those two tags is what makes the file genuinely offline.
    html = re.sub(r'\s*<(?:script|link)[^>]*(?:cdn\.jsdelivr|cdnjs|unpkg)[^>]*>'
                  r'(?:</script>)?', "", html)
    path.write_text(html)
    return path


def collection_graph(g, ctx):
    """The per-repository graph as a NetworkX object, so it draws through the same path."""
    import networkx as nx
    G = nx.Graph()
    for n, m in g["nodes"].items():
        G.add_node(n, size=m["chunks"], label=n, project=m["project"], spread=1)
    for e in g["edges"]:
        w = sum(e["mentions"].values())
        if w:
            G.add_edge(e["a"], e["b"], weight=w)
    return G
