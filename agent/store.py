"""File-based, per-user brand knowledge store with hybrid (semantic + BM25) retrieval.

Layout:  <DATA_DIR>/workspaces/<username>/<brand_id>/{meta.json, chunks.json, vectors.npy}

Every user only ever touches their own folder (username is validated upstream and
brand ids are strict hex), which gives tenant isolation on a shared public host.
No database server and no native extensions are needed; a few thousand chunks per
brand is far below where this approach gets slow.
"""
from __future__ import annotations

import json
import math
import os
import re
import secrets
import shutil
import threading
import time
from pathlib import Path

import numpy as np

from .config import Settings
from .embeddings import Embedder

BRAND_ID_RE = re.compile(r"^[0-9a-f]{12}$")
_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset("a an and are as at be by for from has have in is it its of on or that the this to was were will with we our you your".split())
_LOCK = threading.RLock()


class StoreError(ValueError):
    """User-facing validation problem (limit reached, unknown brand, ...)."""


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


def bm25_scores(query: list[str], docs: list[list[str]], k1: float = 1.5, b: float = 0.75) -> np.ndarray:
    n = len(docs)
    if n == 0 or not query:
        return np.zeros(n, dtype=np.float32)
    avgdl = sum(len(d) for d in docs) / n or 1.0
    df: dict[str, int] = {}
    for d in docs:
        for t in set(d):
            df[t] = df.get(t, 0) + 1
    scores = np.zeros(n, dtype=np.float32)
    for t in set(query):
        if t not in df:
            continue
        idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
        for i, d in enumerate(docs):
            f = d.count(t)
            if f:
                scores[i] += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * len(d) / avgdl))
    return scores


def _minmax(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    lo, hi = float(x.min()), float(x.max())
    return np.zeros_like(x) if hi - lo < 1e-9 else (x - lo) / (hi - lo)


def _atomic_write(path: Path, writer) -> None:
    tmp = path.with_name(path.name + f".{secrets.token_hex(4)}.tmp")
    with open(tmp, "wb") as fh:
        writer(fh)
    os.replace(tmp, path)


class WorkspaceStore:
    def __init__(self, settings: Settings, username: str) -> None:
        self.s = settings
        self.root = settings.data_dir / "workspaces" / username
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ paths
    def _dir(self, brand_id: str) -> Path:
        if not BRAND_ID_RE.match(brand_id or ""):
            raise StoreError("Unknown brand.")
        d = self.root / brand_id
        if not (d / "meta.json").exists():
            raise StoreError("Unknown brand.")
        return d

    def _read_meta(self, d: Path) -> dict:
        return json.loads((d / "meta.json").read_text(encoding="utf-8"))

    def _write_meta(self, d: Path, meta: dict) -> None:
        _atomic_write(d / "meta.json", lambda f: f.write(json.dumps(meta, indent=2).encode()))

    def _read_chunks(self, d: Path) -> list[dict]:
        p = d / "chunks.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []

    def _write_chunks(self, d: Path, chunks: list[dict]) -> None:
        _atomic_write(d / "chunks.json", lambda f: f.write(json.dumps(chunks).encode()))

    def _read_vectors(self, d: Path) -> np.ndarray | None:
        p = d / "vectors.npy"
        return np.load(p) if p.exists() else None

    def _write_vectors(self, d: Path, vecs: np.ndarray) -> None:
        _atomic_write(d / "vectors.npy", lambda f: np.save(f, vecs.astype(np.float32)))

    # ----------------------------------------------------------------- brands
    def list_brands(self) -> list[dict]:
        out = []
        for d in sorted(self.root.iterdir()):
            if d.is_dir() and BRAND_ID_RE.match(d.name) and (d / "meta.json").exists():
                m = self._read_meta(d)
                out.append({"brand_id": m["brand_id"], "name": m["name"], "industry": m["industry"], "docs": len(m["docs"]), "created": m["created"]})
        return sorted(out, key=lambda b: b["created"])

    def get_brand(self, brand_id: str) -> dict:
        m = self._read_meta(self._dir(brand_id))
        return {"brand_id": m["brand_id"], "name": m["name"], "industry": m["industry"], "docs": len(m["docs"])}

    def create_brand(self, name: str, industry: str) -> dict:
        name, industry = (name or "").strip(), (industry or "").strip()
        if not (2 <= len(name) <= 80):
            raise StoreError("Brand name must be 2-80 characters.")
        if len(industry) > 80:
            raise StoreError("Industry must be at most 80 characters.")
        with _LOCK:
            if len(self.list_brands()) >= self.s.max_brands_per_user:
                raise StoreError(f"Limit reached: at most {self.s.max_brands_per_user} brands per user. Delete one first.")
            brand_id = secrets.token_hex(6)
            d = self.root / brand_id
            d.mkdir(parents=True)
            self._write_meta(d, {"brand_id": brand_id, "name": name, "industry": industry or "General", "created": time.time(), "docs": [], "embedder_id": None})
        return self.get_brand(brand_id)

    def delete_brand(self, brand_id: str) -> None:
        with _LOCK:
            shutil.rmtree(self._dir(brand_id))

    # -------------------------------------------------------------- documents
    def list_documents(self, brand_id: str) -> list[dict]:
        return self._read_meta(self._dir(brand_id))["docs"]

    def add_document(self, brand_id: str, name: str, kind: str, chunks: list[str], embedder: Embedder | None, source: str = "") -> dict:
        if not chunks:
            raise StoreError("No usable text found.")
        with _LOCK:
            d = self._dir(brand_id)
            meta = self._read_meta(d)
            if len(meta["docs"]) >= self.s.max_docs_per_brand:
                raise StoreError(f"Limit reached: at most {self.s.max_docs_per_brand} documents per brand.")
            existing = self._read_chunks(d)
            if len(existing) + len(chunks) > self.s.max_chunks_per_brand:
                raise StoreError("This brand's knowledge base is full. Remove some documents first.")
            # skip exact re-ingests of the same source
            if source and any(doc.get("source") == source for doc in meta["docs"]):
                raise StoreError("That source is already in this brand's knowledge base.")
            doc_id = secrets.token_hex(6)
            new_rows = [{"doc_id": doc_id, "text": t, "source": source or name} for t in chunks]
            vecs = self._read_vectors(d)
            if embedder is not None:
                if vecs is None or meta.get("embedder_id") != embedder.id or len(vecs) != len(existing):
                    vecs = embedder.embed_docs([c["text"] for c in existing]) if existing else None
                new_vecs = embedder.embed_docs(chunks)
                vecs = new_vecs if vecs is None or len(vecs) == 0 else np.vstack([vecs, new_vecs])
                meta["embedder_id"] = embedder.id
                self._write_vectors(d, vecs)
            self._write_chunks(d, existing + new_rows)
            doc = {"doc_id": doc_id, "name": name, "kind": kind, "source": source, "chunks": len(chunks), "added": time.time()}
            meta["docs"].append(doc)
            self._write_meta(d, meta)
            return doc

    def delete_document(self, brand_id: str, doc_id: str) -> None:
        with _LOCK:
            d = self._dir(brand_id)
            meta = self._read_meta(d)
            if not any(x["doc_id"] == doc_id for x in meta["docs"]):
                raise StoreError("Unknown document.")
            chunks = self._read_chunks(d)
            keep = [i for i, c in enumerate(chunks) if c["doc_id"] != doc_id]
            vecs = self._read_vectors(d)
            if vecs is not None and len(vecs) == len(chunks):
                self._write_vectors(d, vecs[keep] if keep else np.zeros((0, vecs.shape[1]), np.float32))
            self._write_chunks(d, [chunks[i] for i in keep])
            meta["docs"] = [x for x in meta["docs"] if x["doc_id"] != doc_id]
            self._write_meta(d, meta)

    # -------------------------------------------------------------- retrieval
    def retrieve(self, brand_id: str, query: str, top_k: int, hybrid_weight: float, embedder: Embedder | None) -> list[dict]:
        """hybrid_weight: 1.0 = pure semantic, 0.0 = pure keyword (BM25)."""
        with _LOCK:
            d = self._dir(brand_id)
            chunks = self._read_chunks(d)
            if not chunks:
                return []
            meta = self._read_meta(d)
            vecs = self._read_vectors(d)
            if embedder is not None and (vecs is None or meta.get("embedder_id") != embedder.id or len(vecs) != len(chunks)):
                vecs = embedder.embed_docs([c["text"] for c in chunks])  # backend changed: rebuild once
                self._write_vectors(d, vecs)
                meta["embedder_id"] = embedder.id
                self._write_meta(d, meta)
        n = len(chunks)
        kw = bm25_scores(tokenize(query), [tokenize(c["text"]) for c in chunks])
        use_sem = embedder is not None and vecs is not None and len(vecs) == n
        sem = (vecs @ embedder.embed_query(query)).astype(np.float32) if use_sem else np.zeros(n, np.float32)
        w = min(1.0, max(0.0, hybrid_weight)) if use_sem else 0.0
        combined = w * _minmax(sem) + (1 - w) * _minmax(kw)
        order = np.argsort(-combined, kind="stable")[: max(1, min(top_k, n))]
        return [
            {"text": chunks[i]["text"], "source": chunks[i]["source"], "score": round(float(combined[i]), 4), "semantic": round(float(sem[i]), 4), "keyword": round(float(kw[i]), 4)}
            for i in order
        ]
