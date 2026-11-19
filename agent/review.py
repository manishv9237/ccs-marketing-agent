"""Paragraph-level review helpers: which paragraphs of a draft overlap with claims the fact-check could not find."""
import re


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(w) > 3}


def paragraph_status(content: str, unsupported: list[str], verified: bool) -> list[dict]:
    """[{'text', 'status'}] with status 'grounded' | 'flagged' | 'unchecked'."""
    flagged = [_words(u) for u in (unsupported or []) if u]
    out = []
    for p in [x.strip() for x in re.split(r"\n+", content or "") if x.strip()]:
        pw = _words(p)
        bad = any(f and len(pw & f) >= min(3, len(f)) for f in flagged)
        out.append({"text": p, "status": "flagged" if bad else ("grounded" if verified else "unchecked")})
    return out
