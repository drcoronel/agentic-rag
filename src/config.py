"""Environment knobs and corpus constants.

`PREFIX` and `STATE` are rebound at runtime (selftest namespaces its collections and
redirects the state file), so read them as `config.PREFIX` / `config.STATE` rather than
importing the names — a `from config import STATE` captures the value, not the box.
"""
import os
from pathlib import Path

OLLAMA = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
QDRANT = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_KEY = os.environ.get("QDRANT_API_KEY") or None   # managed Qdrant wants one; local doesn't
EMBED_MODEL = os.environ.get("RAG_EMBED_MODEL", "embeddinggemma:300m")
CHAT_MODEL = os.environ.get("RAG_CHAT_MODEL", "qwen2.5:14b")
PREFIX = os.environ.get("RAG_PREFIX", "")        # collection namespace; selftest overrides it
DIM = int(os.environ.get("RAG_EMBED_DIM", "768"))
# Set to a Vertex model path to embed in GCP instead of Ollama, e.g.
# projects/<id>/locations/europe-north1/publishers/google/models/text-embedding-005
VERTEX_EMBED = os.environ.get("RAG_VERTEX_EMBED", "")

STATE = Path(__file__).parent.parent / ".rag.json"

# embeddinggemma wants these task prefixes; skipping them measurably hurts recall.
DOC_PREFIX = "title: none | text: "
QUERY_PREFIX = "task: search result | query: "

TEXT_EXT = {".md", ".mdx", ".yaml", ".yml", ".py", ".ts", ".tsx", ".js", ".mjs", ".vue",
            ".json", ".tf", ".tfvars", ".sql", ".sh", ".html", ".css", ".toml", ".txt",
            ".cfg", ".ini", ".rs", ".go", ".java"}
SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", "dist", "build",
             ".next", ".nuxt", ".turbo", ".pytest_cache", ".mypy_cache", ".ruff_cache",
             "target", ".terraform"}
TEXT_NAMES = {"Makefile", "Dockerfile", "Justfile", "Procfile", "CODEOWNERS", "Caddyfile",
              ".gitignore", ".dockerignore", ".npmrc", ".nvmrc", ".env.example"}
DOC_FILES = ("AGENTS.md", "CLAUDE.md", "README.md")
MAX_BYTES = 200_000
CHUNK_CHARS = 1200
OVERLAP_LINES = 2
