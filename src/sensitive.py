"""The secret-file guardrail: one predicate, and a scan of what the index already holds.

`is_secret()` is a denylist over the *filename*, checked before the extension allowlist
everywhere a file can reach the model — indexing, grep and read_file. It is deliberately
broad: a false positive costs one unsearchable file, a false negative embeds a
credential into a store with no auth in front of it.

`secret_in()` is the second layer: vendor-shaped key material anywhere in a file's
text keeps the whole file out, whatever it is called. Together they miss a hand-rolled
secret in a conventionally-named file — this is a guardrail, not a scanner. See
docs/indexing.md.
"""
import fnmatch
import re
from pathlib import Path

from .config import settings
from .qdrant import by_path, client
from .state import scope


def is_secret(name):
    """Whether a bare filename looks like it carries credentials.

    `settings.secret_allow` wins, so `.env.example` and `config.tfvars.template` stay
    indexable while `.env` and `config.tfvars` do not.
    """
    n = Path(name).name.lower()
    if any(fnmatch.fnmatch(n, g) for g in settings.secret_allow):
        return False
    return any(fnmatch.fnmatch(n, g) for g in settings.secret_globs)


_PATTERNS = None


def secret_in(text):
    """The label of the first key-material pattern found in `text`, else None.

    Never returns the match itself: the point is to report a file without copying its
    secret into a log.
    """
    global _PATTERNS
    if _PATTERNS is None:
        _PATTERNS = [(k, re.compile(v)) for k, v in settings.secret_patterns.items()]
    return next((label for label, rx in _PATTERNS if rx.search(text)), None)


def scan(ctx, project=None):
    """Points already in the index whose file looks sensitive.

    Checks both layers against what is stored: the filename, and the chunk text that
    is already in the payload — so this also catches files indexed before either guard
    existed, without re-reading the corpus.

    Returns {collection: {path: (chunks, reason)}}, one scroll per collection.
    """
    found = {}
    for name in scope(ctx, project=project):
        hits = {}
        offset = None
        while True:
            points, offset = client().scroll(name, limit=4096, offset=offset,
                                             with_payload=["path", "text"], with_vectors=False)
            for p in points:
                path = p.payload["path"]
                why = ("filename" if is_secret(path)
                       else secret_in(p.payload.get("text", "")))
                if why:
                    n, prev = hits.get(path, (0, why))
                    hits[path] = (n + 1, prev)
            if offset is None:
                break
        if hits:
            found[name] = hits
    return found


def purge(found, quiet=False):
    """Delete every point in `found` from Qdrant. Returns the chunk count removed."""
    gone = 0
    for name, hits in found.items():
        for path, (n, why) in hits.items():
            client().delete(name, points_selector=by_path(path))
            gone += n
            if not quiet:
                print(f"  purged {n} chunks ({why}): {path}")
    return gone
