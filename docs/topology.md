# Topology notes

Implementation rationale for `graph.py`. The README's "Topology" and "Topics" sections
cover what the two views are for and what they showed on the real corpus.

## Two signals, different sharpness

`build_graph()` derives a graph over the collections from two things already in the
index, with no LLM and no second store — one scroll per collection:

- **mentions** (directed, exact): chunks in A that name B, matched against the
  collection name and the directory name. This is the sharp edge — a repo's docs
  pointing at another repo is a real dependency, not a guess.
- **affinity** (undirected): cosine between the mean dense vector of each collection.
  Blurry by construction, since a whole repo is averaged to one point. It is only ever
  read as a ranking among a node's own neighbours, never as an absolute number.

`related` ranks mentions first as hard evidence, and uses affinity only to fill the
remaining slots. It is capped at four, because it ends up in the agent's system prompt.

## Drawing

`graph_mermaid()` draws mentions as labelled arrows, and only the strongest affinity
pairs that have *no* mention edge as dotted lines. The affinity half is otherwise a
complete graph, and unreadable.

`draw_graph()` goes through pyvis — a Python API over vis.js, so physics, scaling and
hover behaviour are configured in Python rather than hand-written JS.
`cdn_resources="in_line"` inlines the library so the page is standalone, and the regex
afterwards strips the Bootstrap `<script>`/`<link>` tags that pyvis's template emits for
filter and config panels we never enable. Removing those two tags is what actually makes
the file work offline.

## Topics

Nodes are topics rather than repositories: a k-nearest-neighbour graph over sampled
chunk vectors, cut into communities by Louvain (NetworkX ships it, so no extra
dependency and no LLM). The interesting ones are the topics that span repositories —
those are where the corpus is really joined, which a per-repo graph cannot show.

`sample_chunks()` takes the first slice of each collection. Qdrant scrolls in point-id
order and ids are a uuid5 of the file path, so that slice is arbitrary with respect to
content — good enough for a shape, and far cheaper than reading every chunk back out.

`label_topic()` is plain tf-idf over the BM25 tokenizer already in use: how much more
often a term appears in this cluster than in the corpus, counting presence per chunk
rather than repetition. The stopword list is English only; a corpus in another
language needs its own, or its stopwords win every cluster.

Two prunings keep the drawing readable:

- **Singletons are dropped.** They are chunks nothing else resembles, so they carry no
  topic, and because they are disconnected the spring layout flings them to the corners
  and collapses everything else into a blob.
- **Only each topic's strongest few links survive.** Every topic links to nearly every
  other, so the full edge set is a hairball no layout can spread. Keeping the top
  `keep_links` per node leaves the backbone instead of the mesh.

## Caching

Clustering is the expensive half — a scroll of every sampled chunk plus an n² similarity
matrix — and it only changes when the index does.

`cache_key()` (in `state.py`) is what a derived view was computed from: the `revision`
counter covers any change to the index itself, the collection scope covers a project
being added or a repo cloned, and the explicit params cover the knobs, so a `-k 12` run
cannot be served from a `-k 8` cache.

The topic graph round-trips through JSON to live in `.rag.json`; `tg_to_json` /
`tg_from_json` are exact inverses, which the selftest asserts.
