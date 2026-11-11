"""Application service: everything the UI can do, with validation and usage limits.

All dependencies (LLM, fetcher, search, embedder) are injectable so the whole
workflow can be tested offline.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from urllib.parse import urlparse

from . import generate as gen
from .chunking import chunk_text
from .config import Settings, get_settings
from .discover import Candidate, discover_sources
from .embeddings import embedding_state, get_embedder
from .extract import ExtractError, bytes_to_text, file_to_text
from .limits import LimitReached, UsageLimiter
from .llm import LLMError, chat
from .security import FetchError, UrlBlocked, safe_get
from .store import StoreError, WorkspaceStore

# Exceptions whose message is safe and useful to show to the end user.
USER_ERRORS = (StoreError, LimitReached, LLMError, ExtractError, UrlBlocked, FetchError)
_UNSET = object()
THIN_PAGE_CHARS = 400
MAX_URLS_PER_CALL = 15


@dataclass
class IngestResult:
    name: str
    ok: bool
    chunks: int = 0
    message: str = ""


class Service:
    def __init__(self, settings: Settings | None = None, *, llm_fn=None, fetch_fn=None, search_fn=None, wiki_fn=None, embedder=_UNSET):
        self.s = settings or get_settings()
        self.limiter = UsageLimiter(self.s)
        self._llm = llm_fn or chat
        self._fetch = fetch_fn or safe_get
        self._search_fn = search_fn
        self._wiki_fn = wiki_fn
        self._embedder_override = embedder

    # -------------------------------------------------------------- plumbing
    @property
    def embedder(self):
        return get_embedder(self.s) if self._embedder_override is _UNSET else self._embedder_override

    def store(self, user: str) -> WorkspaceStore:
        return WorkspaceStore(self.s, user)

    def usage(self, user: str) -> dict:
        if self._embedder_override is _UNSET:  # never block a page load on model loading
            semantic = {"ready": True, "none": False, "pending": None}[embedding_state(self.s)]
        else:
            semantic = self._embedder_override is not None
        return {
            "generations_left": self.limiter.remaining(user, "generate", self.s.gen_per_user_day),
            "searches_left": self.limiter.remaining(user, "search", self.s.search_per_user_day),
            "urls_left": self.limiter.remaining(user, "ingest_url", self.s.ingest_url_per_user_day),
            "semantic_search": semantic,  # True | False | None (model still loading)
        }

    # ---------------------------------------------------------------- brands
    def list_brands(self, user: str) -> list[dict]:
        return self.store(user).list_brands()

    def create_brand(self, user: str, name: str, industry: str, voice_notes: str = "") -> dict:
        st = self.store(user)
        brand = st.create_brand(name, industry)
        notes = (voice_notes or "").strip()
        if len(notes) >= 20:
            try:
                st.add_document(brand["brand_id"], "Brand voice & guidelines (typed)", "notes", chunk_text(notes[:20000]), self.embedder)
            except StoreError:
                pass
        return st.get_brand(brand["brand_id"])

    def delete_brand(self, user: str, brand_id: str) -> None:
        self.store(user).delete_brand(brand_id)

    def list_documents(self, user: str, brand_id: str) -> list[dict]:
        return self.store(user).list_documents(brand_id)

    def delete_document(self, user: str, brand_id: str, doc_id: str) -> None:
        self.store(user).delete_document(brand_id, doc_id)

    # ---------------------------------------------------------------- ingest
    def ingest_files(self, user: str, brand_id: str, paths: list[str]) -> list[IngestResult]:
        st = self.store(user)
        st.get_brand(brand_id)  # validates ownership
        results = []
        for path in paths or []:
            name = path.replace("\\", "/").rsplit("/", 1)[-1]
            try:
                _, text = file_to_text(path, self.s.max_upload_mb)
                chunks = chunk_text(text)
                doc = st.add_document(brand_id, name, "file", chunks, self.embedder, source=f"file:{name}")
                results.append(IngestResult(name, True, doc["chunks"]))
            except (StoreError, ExtractError) as exc:
                results.append(IngestResult(name, False, message=str(exc)))
        return results

    def ingest_urls(self, user: str, brand_id: str, urls: list[str]) -> list[IngestResult]:
        st = self.store(user)
        st.get_brand(brand_id)
        urls = [u.strip() for u in urls if u and u.strip()][:MAX_URLS_PER_CALL]
        results = []
        for url in urls:
            try:
                self.limiter.consume(user, "ingest_url", self.s.ingest_url_per_user_day)
            except LimitReached as exc:
                results.append(IngestResult(url, False, message=str(exc)))
                break
            try:
                res = self._fetch(url, self.s)
                path_name = urlparse(res.url).path
                title, text = bytes_to_text(res.data, res.content_type, path_name)
                chunks = chunk_text(text)
                if not chunks:
                    raise ExtractError("No readable text found (the page is probably built with JavaScript). Upload a brand PDF/Word file instead.")
                doc = st.add_document(brand_id, title or url, "web", chunks, self.embedder, source=res.url)
                note = ""
                if len(text) < THIN_PAGE_CHARS:
                    note = "thin page (likely built with JavaScript) - consider uploading a brand PDF/Word file instead"
                results.append(IngestResult(url, True, doc["chunks"], note))
            except USER_ERRORS as exc:
                results.append(IngestResult(url, False, message=str(exc)))
        return results

    def discover(self, user: str, brand_id: str, domain: str = "") -> tuple[list[Candidate], list[str]]:
        brand = self.store(user).get_brand(brand_id)
        self.limiter.consume(user, "search", self.s.search_per_user_day)
        return discover_sources(brand["name"], brand["industry"], domain, self.s, max_results=20,
                                search_fn=self._search_fn, wiki_fn=self._wiki_fn)

    def _verify(self, variations: list[dict], sources: list[dict], gen_model: str) -> None:
        """Second pass: flag claims the retrieved sources do not support. Best-effort, never blocks output."""
        vm = self.s.verifier_model
        for v in variations:
            v.setdefault("unsupported", [])
            v.setdefault("verified", False)
        if vm.lower() == "off" or not variations or variations[0]["angle"].startswith("Model output"):
            return
        model = vm if (vm and vm in self.s.allowed_models) else gen_model
        try:
            raw = self._llm(gen.build_verify_messages(variations, sources), model, self.s, max_output_tokens=2000, json_mode=True)
            gen.apply_verification(variations, raw)
        except Exception:  # verification must never break generation
            pass

    # -------------------------------------------------------------- generate
    def generate(self, user: str, brand_id: str, platform: str, goal: str, brief: str, n: int, model: str,
                 top_k: int | None = None, hybrid_weight: float | None = None) -> dict:
        goal, brief = (goal or "").strip(), (brief or "").strip()
        if platform not in gen.PLATFORMS:
            raise StoreError("Choose a platform.")
        if len(goal) < 5:
            raise StoreError("Describe the campaign goal (at least a few words).")
        if len(goal) > 500 or len(brief) > 2000:
            raise StoreError("Goal is limited to 500 characters and the brief to 2000.")
        if model not in self.s.allowed_models:
            raise StoreError("That model is not enabled on this server.")
        n = max(1, min(5, int(n)))
        top_k = max(2, min(12, int(self.s.default_top_k if top_k is None else top_k)))
        hybrid_weight = self.s.default_hybrid if hybrid_weight is None else hybrid_weight

        st = self.store(user)
        brand = st.get_brand(brand_id)
        t0 = time.time()
        sources = st.retrieve(brand_id, f"{goal} {brief}", top_k, float(hybrid_weight), self.embedder)
        retrieval_ms = (time.time() - t0) * 1000
        if not sources:
            raise StoreError("This brand has no knowledge yet. Upload a document or add web sources first.")

        self.limiter.consume(user, "generate", self.s.gen_per_user_day, self.s.gen_global_day)
        t1 = time.time()
        try:
            raw = self._llm(gen.build_messages(brand, platform, goal, brief, sources, n), model, self.s, max_output_tokens=2500 + 1200 * n, json_mode=True)
        except LLMError:
            self.limiter.refund(user, "generate")
            raise
        gen_ms = (time.time() - t1) * 1000
        variations = gen.parse_variations(raw, platform, n, len(sources))
        t2 = time.time()
        self._verify(variations, sources, model)
        verify_ms = (time.time() - t2) * 1000
        return {
            "brand": brand,
            "platform": platform,
            "model": model,
            "variations": variations,
            "verify_ms": verify_ms,
            "sources": sources,
            "retrieval_ms": retrieval_ms,
            "generation_ms": gen_ms,
            "semantic": self.embedder is not None,
        }
