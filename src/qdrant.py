"""The Qdrant client and collection schema."""
import sys

from qdrant_client import QdrantClient, models as qm

from .config import DIM, QDRANT, QDRANT_KEY

_client = None


def client():
    global _client
    if _client is None:
        _client = QdrantClient(url=QDRANT, api_key=QDRANT_KEY, timeout=120)
    return _client


def is_hybrid(name):
    p = client().get_collection(name).config.params
    return (isinstance(p.vectors, dict) and "bm25" in (p.sparse_vectors or {})
            and getattr(p.vectors.get("dense"), "size", None) == DIM)


def ensure(name):
    c = client()
    if c.collection_exists(name):
        if is_hybrid(name):
            return c
        c.delete_collection(name)       # wrong schema or wrong dim; sync refills it
        print(f"  {name}: rebuilding (schema or embedding dim changed)", file=sys.stderr)
    c.create_collection(
        name,
        vectors_config={"dense": qm.VectorParams(size=DIM, distance=qm.Distance.COSINE)},
        sparse_vectors_config={"bm25": qm.SparseVectorParams(modifier=qm.Modifier.IDF)})
    c.create_payload_index(name, "path", qm.PayloadSchemaType.KEYWORD)
    return c


def by_path(path):
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
