"""The Qdrant client and collection schema.

One collection per repository, carrying a named dense vector and a `bm25` sparse vector
with the IDF modifier. See docs/indexing.md.
"""
import sys

from qdrant_client import QdrantClient, models as qm

from .config import settings


_client = None


def client():
    """The process-wide Qdrant client, connected lazily."""
    global _client
    if _client is None:
        _client = QdrantClient(url=settings.qdrant, api_key=settings.qdrant_key, timeout=120)
    return _client


def is_hybrid(name):
    """Whether a collection already has the current schema and embedding dimension."""
    p = client().get_collection(name).config.params
    return (isinstance(p.vectors, dict) and "bm25" in (p.sparse_vectors or {})
            and getattr(p.vectors.get("dense"), "size", None) == settings.dim)


def ensure(name):
    """Create the collection, or rebuild it if its schema or dimension is stale.

    Everything here is rebuildable from disk, so dropping a mismatched collection and
    letting the same sync pass refill it is always cheaper than migrating it.
    """
    c = client()
    if c.collection_exists(name):
        if is_hybrid(name):
            return c
        c.delete_collection(name)
        print(f"  {name}: rebuilding (schema or embedding dim changed)", file=sys.stderr)
    c.create_collection(
        name,
        vectors_config={"dense": qm.VectorParams(size=settings.dim, distance=qm.Distance.COSINE)},
        sparse_vectors_config={"bm25": qm.SparseVectorParams(modifier=qm.Modifier.IDF)})
    c.create_payload_index(name, "path", qm.PayloadSchemaType.KEYWORD)
    return c


def by_path(path):
    """A filter selecting every point belonging to one file."""
    return qm.Filter(must=[qm.FieldCondition(key="path", match=qm.MatchValue(value=path))])


def indexed_files(name):
    """path -> (hash, mtime, size) for everything currently in one collection."""
    seen, offset = {}, None
    while True:
        points, offset = client().scroll(name, limit=4096, offset=offset,
                                         with_payload=["path", "hash", "mtime", "size"],
                                         with_vectors=False)
        for p in points:
            seen[p.payload["path"]] = (p.payload["hash"], p.payload["mtime"], p.payload["size"])
        if offset is None:
            return seen
