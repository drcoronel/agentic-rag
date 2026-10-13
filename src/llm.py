"""Embeddings and chat completion. Ollama by default, Vertex when RAG_VERTEX_EMBED is set.

See docs/indexing.md for the task-prefix and token-caching rationale.
"""
import json, os, subprocess, sys, time, urllib.request

from .config import settings


def _post(url, payload, timeout=600, headers=None):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json", **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _get(url, headers, timeout=2):
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as r:
        return json.loads(r.read())


_token = (0.0, "")


def gcp_token():
    """Access token for Vertex, cached for 45 minutes.

    Tried in order: the GOOGLE_ACCESS_TOKEN override, the metadata server (so this works
    unchanged on Cloud Run), then the local gcloud.
    """
    global _token
    if _token[0] > time.time():
        return _token[1]
    t = os.environ.get("GOOGLE_ACCESS_TOKEN")
    if not t:
        try:
            t = _get("http://metadata.google.internal/computeMetadata/v1/instance/"
                     "service-accounts/default/token", {"Metadata-Flavor": "Google"})["access_token"]
        except Exception:
            t = subprocess.run(["gcloud", "auth", "print-access-token"],
                               capture_output=True, text=True).stdout.strip()
    if not t:
        sys.exit("RAG_VERTEX_EMBED is set but no GCP credentials — run `gcloud auth login`")
    _token = (time.time() + 2700, t)
    return t


def embed(texts, prefix):
    """Dense vectors of `settings.dim` dimensions, batched 64 at a time.

    `prefix` is one of the task strings in config; embeddinggemma needs them and recall
    drops measurably without one. On Vertex the equivalent is the task_type parameter.
    """
    out = []
    for i in range(0, len(texts), 64):
        batch = texts[i:i + 64]
        if settings.vertex_embed:
            task = "RETRIEVAL_QUERY" if prefix == settings.query_prefix else "RETRIEVAL_DOCUMENT"
            region = settings.vertex_embed.split("/locations/")[1].split("/")[0]
            r = _post(f"https://{region}-aiplatform.googleapis.com/v1/{settings.vertex_embed}:predict",
                      {"instances": [{"task_type": task, "content": t} for t in batch],
                       "parameters": {"outputDimensionality": settings.dim}},
                      headers={"Authorization": f"Bearer {gcp_token()}"})
            out += [p["embeddings"]["values"] for p in r["predictions"]]
        else:
            out += _post(f"{settings.ollama}/api/embed",
                         {"model": settings.embed_model, "input": [prefix + t for t in batch]})["embeddings"]
    return out


def chat(messages, tools=None):
    """One completion at temperature 0. Returns the message dict, tool calls included."""
    body = {"model": settings.chat_model, "messages": messages, "stream": False, "options": {"temperature": 0}}
    if tools:
        body["tools"] = tools
    return _post(f"{settings.ollama}/api/chat", body)["message"]
