"""Runtime settings, read from environment variables (never hard-coded).

Secrets (OPENAI_API_KEY, APP_ACCESS_CODE, search keys) must come from the
hosting platform's secret manager or a local .env file that is git-ignored.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:  # optional convenience for local development
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - python-dotenv not installed
    pass


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


def _list(name: str, default: str) -> tuple[str, ...]:
    raw = os.getenv(name, "").strip() or default
    return tuple(x.strip() for x in raw.split(",") if x.strip())


@dataclass(frozen=True)
class Settings:
    # --- secrets / access -------------------------------------------------
    openai_api_key: str
    openai_base_url: str
    access_code: str
    users: dict  # username -> password; when non-empty, replaces the shared access code
    allow_no_auth: bool
    tavily_api_key: str
    brave_api_key: str

    # --- models -------------------------------------------------------------
    allowed_models: tuple[str, ...]
    default_model: str
    verifier_model: str  # '' = use the generation model; 'off' = skip verification
    default_top_k: int
    default_hybrid: float
    embedder: str  # auto | fastembed | sentence-transformers | openai | hash | none
    fastembed_model: str
    st_model: str
    openai_embed_model: str
    search_provider: str  # auto | ddg | tavily | brave | none

    # --- storage ------------------------------------------------------------
    data_dir: Path

    # --- abuse limits (per UTC day unless noted) ---------------------------------
    gen_per_user_day: int
    gen_global_day: int
    search_per_user_day: int
    ingest_url_per_user_day: int
    eval_per_user_day: int
    eval_global_day: int
    eval_users: tuple  # empty = every signed-in user may run the evaluation
    max_upload_mb: int
    max_brands_per_user: int
    max_docs_per_brand: int
    max_chunks_per_brand: int

    # --- network ------------------------------------------------------------
    fetch_timeout: int
    max_fetch_bytes: int
    llm_timeout: int


def _parse_users(raw: str) -> dict:
    """APP_USERS="priya:pass1,rahul:pass2" -> {"priya": "pass1", "rahul": "pass2"}.
    Usernames follow the display-name rules (lowercase letters/digits/._-, 2-32 chars).
    Passwords may not contain commas. A malformed entry stops startup instead of being skipped."""
    from .security import normalize_username  # local import: security imports config

    users: dict = {}
    for item in [i.strip() for i in (raw or "").split(",") if i.strip()]:
        name, sep, pw = item.partition(":")
        norm = normalize_username(name)
        if not sep or norm is None or len(pw) < 6:
            raise ValueError("APP_USERS entries must look like username:password (password at least 6 characters, no commas). Problem entry starts with: " + (name[:12] or "?"))
        users[norm] = pw
    return users


def get_settings() -> Settings:
    models = _list("ALLOWED_MODELS", "gpt-5-nano,gpt-5.4-nano,gpt-6-luna,gpt-5.6-luna")
    default_model = os.getenv("DEFAULT_MODEL", "").strip() or models[0]
    if default_model not in models:
        models = (default_model,) + models
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY", "").strip().strip('"').strip("'"),
        openai_base_url=(os.getenv("OPENAI_BASE_URL", "").strip() or "https://api.openai.com/v1").rstrip("/"),
        access_code=os.getenv("APP_ACCESS_CODE", "").strip(),
        users=_parse_users(os.getenv("APP_USERS", "")),
        allow_no_auth=os.getenv("ALLOW_NO_AUTH", "").strip() == "1",
        tavily_api_key=os.getenv("TAVILY_API_KEY", "").strip(),
        brave_api_key=os.getenv("BRAVE_API_KEY", "").strip(),
        allowed_models=models,
        default_model=default_model,
        verifier_model=os.getenv("VERIFIER_MODEL", "gpt-5.4-nano").strip(),
        default_top_k=_int("DEFAULT_TOP_K", 6),
        default_hybrid=float(os.getenv("DEFAULT_HYBRID", "0.5") or 0.5),
        embedder=os.getenv("EMBEDDER", "auto").strip().lower() or "auto",
        fastembed_model=os.getenv("FASTEMBED_MODEL", "").strip() or "BAAI/bge-small-en-v1.5",
        st_model=os.getenv("ST_MODEL", "").strip() or "all-MiniLM-L6-v2",
        openai_embed_model=os.getenv("OPENAI_EMBED_MODEL", "").strip() or "text-embedding-3-small",
        search_provider=os.getenv("SEARCH_PROVIDER", "auto").strip().lower() or "auto",
        data_dir=Path(os.getenv("DATA_DIR", "").strip() or "./data").resolve(),
        gen_per_user_day=_int("GEN_PER_USER_DAY", 25),
        gen_global_day=_int("GEN_GLOBAL_DAY", 400),
        search_per_user_day=_int("SEARCH_PER_USER_DAY", 30),
        ingest_url_per_user_day=_int("INGEST_URL_PER_USER_DAY", 80),
        eval_per_user_day=_int("EVAL_PER_USER_DAY", 400),
        eval_global_day=_int("EVAL_GLOBAL_DAY", 800),
        eval_users=_list("EVAL_USERS", ""),
        max_upload_mb=_int("MAX_UPLOAD_MB", 10),
        max_brands_per_user=_int("MAX_BRANDS_PER_USER", 5),
        max_docs_per_brand=_int("MAX_DOCS_PER_BRAND", 40),
        max_chunks_per_brand=_int("MAX_CHUNKS_PER_BRAND", 4000),
        fetch_timeout=_int("FETCH_TIMEOUT", 15),
        max_fetch_bytes=_int("MAX_FETCH_MB", 3) * 1024 * 1024,
        llm_timeout=_int("LLM_TIMEOUT", 150),
    )
