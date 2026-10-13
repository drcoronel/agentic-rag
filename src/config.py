"""Settings, read from the environment once at import.

`settings` is a single mutable instance that every module holds a reference to, which is
what lets the selftest redirect `state` and namespace `prefix` for the duration of a run
without the real index ever being touched.

Corpus constants (the extension and skip lists, chunk sizes) live here too. They are
fields rather than module constants so they are overridable from the environment and
validated the same way, even though nothing normally sets them.
"""
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


_ROOT = Path(__file__).parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", extra="ignore",
                                      validate_assignment=True)

    ollama: str = Field("http://localhost:11434", validation_alias="OLLAMA_HOST")
    qdrant: str = Field("http://localhost:6333", validation_alias="QDRANT_URL")
    qdrant_key: str | None = Field(None, validation_alias="QDRANT_API_KEY")
    """Managed Qdrant wants an API key; a local one does not."""

    embed_model: str = "embeddinggemma:300m"
    chat_model: str = "qwen2.5:14b"
    dim: int = Field(768, validation_alias="RAG_EMBED_DIM")

    vertex_embed: str = ""
    """Full Vertex model path, e.g.
    `projects/<id>/locations/<region>/publishers/google/models/text-embedding-005`.
    Set it to embed in GCP instead of Ollama."""

    prefix: str = ""
    """Collection namespace. The selftest overrides it so its collections are its own."""

    state: Path = _ROOT / ".rag.json"

    doc_prefix: str = "title: none | text: "
    query_prefix: str = "task: search result | query: "
    """embeddinggemma's task prefixes. Skipping them measurably hurts recall."""

    text_ext: set[str] = {
        ".md", ".mdx", ".yaml", ".yml", ".py", ".ts", ".tsx", ".js", ".mjs", ".vue",
        ".json", ".tf", ".tfvars", ".sql", ".sh", ".html", ".css", ".toml", ".txt",
        ".cfg", ".ini", ".rs", ".go", ".java"}
    skip_dirs: set[str] = {
        ".git", "node_modules", ".venv", "venv", "__pycache__", "dist", "build",
        ".next", ".nuxt", ".turbo", ".pytest_cache", ".mypy_cache", ".ruff_cache",
        "target", ".terraform", ".secrets", ".gnupg", ".ssh"}
    text_names: set[str] = {
        "Makefile", "Dockerfile", "Justfile", "Procfile", "CODEOWNERS", "Caddyfile",
        ".gitignore", ".dockerignore", ".nvmrc", ".env.example"}

    secret_globs: set[str] = {
        # environment and shell-sourced config
        ".env", ".env.*", "*.env", ".envrc",
        # private keys, certificates, keystores
        "*.key", "*.pem", "*.p12", "*.pfx", "*.pkcs12", "*.jks", "*.keystore", "*.ppk",
        "*.asc", "*.gpg", "*.kdbx", "id_rsa*", "id_dsa*", "id_ecdsa*", "id_ed25519*",
        # anything that names itself
        "*secret*", "*credential*", "*password*", "*passwd*", "*apikey*", "*api_key*",
        # tool config that holds tokens
        ".netrc", ".pgpass", ".htpasswd", ".npmrc", ".pypirc", ".dockercfg",
        ".git-credentials", ".s3cfg", "kubeconfig", "*.kubeconfig",
        # terraform variable files are a routine credential home
        "*.tfvars", "*.tfvars.json",
        # service-account key material
        "service-account*.json", "serviceaccount*.json", "*-key.json", "*-sa.json"}
    """Filenames never indexed, grepped or read. Broad on purpose — see
    docs/indexing.md. A false positive loses one searchable file; a false negative
    embeds a credential into a store with no auth in front of it."""

    secret_allow: set[str] = {"*.example", "*.sample", "*.template", "*.dist",
                              "*.example.*", "*.sample.*",
                              "*.md", "*.mdx", "*.rst", "*.txt"}
    """Checked first, so placeholder files and prose stay indexable.

    Prose is exempt because the name hints (`*secret*`, `*credential*`) are what catch
    documentation *about* credentials — `sealedsecrets-create.md` is a how-to, not a
    key. Nothing is lost by exempting it: detection is filename-only, so a credential
    pasted into a `.md` was never going to be caught either way. The hard patterns
    (`.env`, `*.key`, `.netrc`) cannot be prose, so they are unaffected."""
    doc_files: tuple[str, ...] = ("AGENTS.md", "CLAUDE.md", "README.md")

    secret_patterns: dict[str, str] = {
        "private key":       r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
        "aws access key":    r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b",
        "github token":      r"\bgh[pousr]_[A-Za-z0-9]{36}\b|\bgithub_pat_[A-Za-z0-9_]{60,}\b",
        "gitlab token":      r"\bglpat-[A-Za-z0-9_-]{20}\b",
        "slack token":       r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b",
        "google api key":    r"\bAIza[0-9A-Za-z_-]{35}\b",
        "openai key":        r"\bsk-[A-Za-z0-9]{32,}\b",
        "anthropic key":     r"\bsk-ant-[A-Za-z0-9_-]{20,}\b",
        "npm token":         r"\bnpm_[A-Za-z0-9]{36}\b",
        "pypi token":        r"\bpypi-AgEIcHlwaS5vcmc[A-Za-z0-9_-]{20,}\b",
        "json web token":    r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b",
    }
    """Content patterns that keep a whole file out of the index, label -> regex.

    Deliberately only vendor-shaped key material, which is close to zero false
    positives. A generic `password: "..."` rule was left out: it fires on every
    placeholder in every how-to, and a skipped doc is a silent hole in the index."""

    tracked_only: bool = False
    """Index only files `git ls-files` reports. Inherits the repo's own .gitignore,
    which is the strongest available signal for "this was never meant to be read".
    Non-git directories are indexed in full regardless."""

    max_bytes: int = 200_000
    chunk_chars: int = 1200
    overlap_lines: int = 2


settings = Settings()
