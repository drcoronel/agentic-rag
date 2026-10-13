"""Chunking and the BM25 sparse-vector encoding."""
import hashlib, re
from collections import Counter

from qdrant_client import models as qm

from .config import CHUNK_CHARS, OVERLAP_LINES

def chunk(text):
    """[(start_line, end_line, text)], 1-based inclusive, with a small overlap."""
    lines = text.splitlines()
    out, buf, start = [], [], 0
    for i, ln in enumerate(lines):
        buf.append(ln)
        if sum(map(len, buf)) >= CHUNK_CHARS:
            out.append((start + 1, i + 1, "\n".join(buf)))
            buf, start = buf[-OVERLAP_LINES:], max(0, i - OVERLAP_LINES + 1)
    if any(l.strip() for l in buf):
        out.append((start + 1, len(lines), "\n".join(buf)))
    return out


# ---------------------------------------------------------------- bm25

# Deliberately not fastembed's Bm25: its English stemmer and stopword list are tuned for
# prose, and this corpus is mostly YAML keys, project ids and identifiers. Splitting those
# the way a reader would is worth more here than stemming, and costs no dependency.
_WORD = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.\-/]*")
_PART = re.compile(r"[a-z0-9]+|[A-Z][a-z0-9]+|[A-Z]+(?![a-z])")
BM25_K1, BM25_B = 1.2, 0.75
BM25_AVGDL = 180            # ≈ tokens in a CHUNK_CHARS chunk; only the ratio matters


def tokens(text):
    """Whole identifiers, plus their camelCase / snake_case / dotted / hyphenated parts.

    `allowDelete` also yields allow and delete; `zoneclass` stays whole. Exact literals are
    precisely what dense vectors are worst at, so they have to survive intact.
    """
    for w in _WORD.findall(text):
        yield w.lower()
        parts = _PART.findall(w)
        if len(parts) > 1:
            for p in parts:
                if len(p) > 1:
                    yield p.lower()


def term_id(tok):
    """Stable u32 id for a token — Qdrant sparse indices are u32, and a hash means no
    vocabulary file to keep in sync with the index."""
    return int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=4).digest(), "big")


def sparse(text, query=False):
    """BM25 term weights. Qdrant's IDF modifier supplies the corpus half at query time,
    so a document stores only the saturated term frequency, and a query only its terms."""
    tf = Counter(term_id(t) for t in tokens(text))
    if query:
        return qm.SparseVector(indices=list(tf), values=[1.0] * len(tf))
    n = sum(tf.values()) or 1
    norm = BM25_K1 * (1 - BM25_B + BM25_B * n / BM25_AVGDL)
    return qm.SparseVector(indices=list(tf), values=[f / (f + norm) for f in tf.values()])
