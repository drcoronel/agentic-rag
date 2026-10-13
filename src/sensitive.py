"""The secret-file guardrail: one predicate, and a scan of what the index already holds.

`is_secret()` is a denylist over the *filename*, checked before the extension allowlist
everywhere a file can reach the model — indexing, grep and read_file. It is deliberately
broad: a false positive costs one unsearchable file, a false negative embeds a
credential into a store with no auth in front of it.

It cannot read minds. It does not inspect contents, so a secret pasted into `notes.md`
is still indexed. See docs/indexing.md.
"""
import fnmatch
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


def scan(ctx, project=None):
    """Points already in the index whose file looks sensitive.

    Returns {collection: {path: chunks}}. Reads only the `path` payload, so this is one
    cheap scroll per collection rather than a re-read of the corpus.
    """
    found = {}
    for name in scope(ctx, project=project):
        hits = {}
        offset = None
        while True:
            points, offset = client().scroll(name, limit=4096, offset=offset,
                                             with_payload=["path"], with_vectors=False)
            for p in points:
                path = p.payload["path"]
                if is_secret(path):
                    hits[path] = hits.get(path, 0) + 1
            if offset is None:
                break
        if hits:
            found[name] = hits
    return found


def purge(found, quiet=False):
    """Delete every point in `found` from Qdrant. Returns the chunk count removed."""
    gone = 0
    for name, hits in found.items():
        for path, n in hits.items():
            client().delete(name, points_selector=by_path(path))
            gone += n
            if not quiet:
                print(f"  purged {n} chunks: {path}")
    return gone
