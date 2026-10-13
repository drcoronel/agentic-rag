"""Embeddings and chat completion. Ollama by default, Vertex when RAG_VERTEX_EMBED is set."""
import json, os, subprocess, sys, time, urllib.request

from .config import CHAT_MODEL, DIM, EMBED_MODEL, OLLAMA, QUERY_PREFIX, VERTEX_EMBED

def _post(url, payload, timeout=600, headers=None):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json", **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


_token = (0.0, "")


def gcp_token():
    """Access token for Vertex: env override, then the metadata server (Cloud Run), then
    the local gcloud. Cached — tokens last an hour and an index pass is hundreds of calls."""
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


def _get(url, headers, timeout=2):
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as r:
        return json.loads(r.read())


def embed(texts, prefix):
    """768-dim vectors. Ollama locally; Vertex when RAG_VERTEX_EMBED is set — the only
    part of the pipeline that is not already a URL in the environment."""
    out = []
    for i in range(0, len(texts), 64):
        batch = texts[i:i + 64]
        if VERTEX_EMBED:
            task = "RETRIEVAL_QUERY" if prefix == QUERY_PREFIX else "RETRIEVAL_DOCUMENT"
            region = VERTEX_EMBED.split("/locations/")[1].split("/")[0]
            r = _post(f"https://{region}-aiplatform.googleapis.com/v1/{VERTEX_EMBED}:predict",
                      {"instances": [{"task_type": task, "content": t} for t in batch],
                       "parameters": {"outputDimensionality": DIM}},
                      headers={"Authorization": f"Bearer {gcp_token()}"})
            out += [p["embeddings"]["values"] for p in r["predictions"]]
        else:
            out += _post(f"{OLLAMA}/api/embed",
                         {"model": EMBED_MODEL, "input": [prefix + t for t in batch]})["embeddings"]
    return out


def chat(messages, tools=None):
    body = {"model": CHAT_MODEL, "messages": messages, "stream": False, "options": {"temperature": 0}}
    if tools:
        body["tools"] = tools
    return _post(f"{OLLAMA}/api/chat", body)["message"]
