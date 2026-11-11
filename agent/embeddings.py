"""Pluggable embedding backends.

auto  -> fastembed (light ONNX, used in the Docker image) -> sentence-transformers
         (handy on a dev laptop) -> None (retrieval falls back to keyword-only BM25).
hash  -> deterministic hashing embedder, for tests/offline use only.
openai-> OpenAI embeddings API (uses OPENAI_API_KEY).
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import threading
from typing import Protocol

import numpy as np
import requests

from .config import Settings

log = logging.getLogger("agent.embeddings")
_TOKEN = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    id: str

    def embed_docs(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


def _normalize(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    if m.ndim == 1:
        m = m[None, :]
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


class HashEmbedder:
    """Bag-of-words hashing trick. Not semantic, but deterministic and dependency-free."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim
        self.id = f"hash-{dim}"

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in _TOKEN.findall(text.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest()[:8], 16)
            v[h % self.dim] += 1.0 if (h >> 31) & 1 else -1.0
        return v

    def embed_docs(self, texts):
        return _normalize(np.stack([self._vec(t) for t in texts])) if texts else np.zeros((0, self.dim), np.float32)

    def embed_query(self, text):
        return _normalize(self._vec(text))[0]


class FastEmbedEmbedder:
    def __init__(self, model_name: str) -> None:
        from fastembed import TextEmbedding  # lazy: heavy import

        self._model = TextEmbedding(model_name=model_name, cache_dir=os.getenv("FASTEMBED_CACHE_PATH") or None)
        self.id = f"fastembed:{model_name}"
        self._lock = threading.Lock()

    def embed_docs(self, texts):
        with self._lock:
            return _normalize(np.array(list(self._model.embed(texts))))

    def embed_query(self, text):
        with self._lock:
            query_embed = getattr(self._model, "query_embed", None)
            out = query_embed(text) if query_embed else self._model.embed([text])
            return _normalize(np.array(list(out)))[0]


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer  # lazy: heavy import

        self._model = SentenceTransformer(model_name)
        self.id = f"st:{model_name}"
        self._lock = threading.Lock()

    def embed_docs(self, texts):
        with self._lock:
            return _normalize(self._model.encode(texts, show_progress_bar=False))

    def embed_query(self, text):
        with self._lock:
            return _normalize(self._model.encode([text], show_progress_bar=False))[0]


class OpenAIEmbedder:
    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self._s = settings
        self.id = f"openai:{settings.openai_embed_model}"

    def _call(self, texts: list[str]) -> np.ndarray:
        out: list[list[float]] = []
        for i in range(0, len(texts), 64):
            r = requests.post(
                f"{self._s.openai_base_url}/embeddings",
                headers={"Authorization": f"Bearer {self._s.openai_api_key}"},
                json={"model": self._s.openai_embed_model, "input": texts[i : i + 64]},
                timeout=60,
            )
            if r.status_code != 200:
                raise RuntimeError(f"Embedding request failed (HTTP {r.status_code})")
            out.extend(d["embedding"] for d in r.json()["data"])
        return _normalize(np.array(out))

    def embed_docs(self, texts):
        return self._call(texts)

    def embed_query(self, text):
        return self._call([text])[0]


_cache: dict[tuple, Embedder | None] = {}
_cache_lock = threading.Lock()


def _key(settings: Settings) -> tuple:
    return (settings.embedder, settings.fastembed_model, settings.st_model, settings.openai_embed_model)


def embedding_state(settings: Settings) -> str:
    """'ready' | 'none' | 'pending' - never blocks (safe to call on every page load)."""
    key = _key(settings)
    if key not in _cache:
        return "pending"
    return "ready" if _cache[key] is not None else "none"


def get_embedder(settings: Settings) -> Embedder | None:
    """Return the configured embedder, or None if semantic search is unavailable."""
    key = _key(settings)
    with _cache_lock:
        if key in _cache:
            return _cache[key]
        emb = _build(settings)
        _cache[key] = emb
        return emb


def _build(settings: Settings) -> Embedder | None:
    choice = settings.embedder
    if choice == "none":
        return None
    if choice == "hash":
        return HashEmbedder()
    order = {
        "auto": ["fastembed", "sentence-transformers"],
        "fastembed": ["fastembed"],
        "sentence-transformers": ["sentence-transformers"],
        "openai": ["openai"],
    }.get(choice, ["fastembed", "sentence-transformers"])
    for name in order:
        try:
            if name == "fastembed":
                return FastEmbedEmbedder(settings.fastembed_model)
            if name == "sentence-transformers":
                return SentenceTransformerEmbedder(settings.st_model)
            if name == "openai":
                return OpenAIEmbedder(settings)
        except Exception as exc:  # missing package, no wheel, model download failed
            log.warning("Embedder '%s' unavailable (%s: %s)", name, exc.__class__.__name__, exc)
    log.warning("No embedding backend available; using keyword-only (BM25) retrieval.")
    return None
