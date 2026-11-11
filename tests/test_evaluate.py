"""Evaluation harness: balanced test set, judge parsing, brand setup, full run, CSV, limits."""
import csv
import json
from collections import Counter

import pytest

from agent import evaluate as ev
from agent.embeddings import HashEmbedder
from agent.llm import LLMError
from agent.security import FetchResult
from agent.service import Service
from agent.store import StoreError

HTML = (b"<html><head><title>About</title></head><body><main><p>Acme is a family owned maker founded in 1998 in Pune by two sisters.</p>"
        b"<p>Our factories run on solar power and every ingredient comes from nearby farms across the state.</p></main></body></html>")


def fake_llm(messages, model, s, max_output_tokens=0, json_mode=False):
    if messages[0]["content"].startswith("You are a strict, impartial evaluator"):
        return json.dumps({"grounded": 4, "on_brand": 5, "goal_relevance": 3, "platform_fit": 4, "unsupported_claims": [], "reason": "fine"})
    return json.dumps({"variations": [{"angle": "Benefit-led", "content": "Short post about solar factories #Brand", "sources": [1], "brand_note": "x"}]})


@pytest.fixture
def svc(settings):
    return Service(settings, llm_fn=fake_llm, fetch_fn=lambda url, s, session=None: FetchResult(url, "text/html", HTML),
                   embedder=HashEmbedder(), search_fn=lambda q, n: [], wiki_fn=lambda n: [])


def test_items_are_balanced():
    items = ev.build_items()
    assert len(items) == 32
    for key in ("brand", "platform", "goal_id"):
        assert set(Counter(i[key] for i in items).values()) == {8}, key
    assert len({(i["brand"], i["platform"], i["goal_id"]) for i in items}) == 32  # no duplicates
    half = Counter(i["platform"] for i in ev.build_items(16))
    assert set(half.values()) == {4}  # a partial run stays balanced across platforms


def test_parse_scores_clamps_and_requires_numbers():
    s = ev.parse_scores('{"grounded": 9, "on_brand": 0, "goal_relevance": "x", "platform_fit": 3.6, "reason": "ok"}')
    assert s["grounded"] == 5 and s["on_brand"] == 1 and "goal_relevance" not in s and s["platform_fit"] == 4
    assert ev.parse_scores("not json") == {}


def test_setup_run_summary_and_csv(svc, tmp_path):
    texts = list(ev.setup_brands_stream(svc, "priya"))
    assert "Done" in texts[-1]
    assert {b["name"] for b in svc.list_brands("priya")} == {b["name"] for b in ev.EVAL_BRANDS}
    # running setup again must not duplicate brands
    list(ev.setup_brands_stream(svc, "priya"))
    assert len(svc.list_brands("priya")) == 4

    params = {"gen_model": "gpt-5-nano", "judge_model": "gpt-5.4-nano", "top_k": 6, "hybrid": 0.5}
    last = None
    for last in ev.evaluate_stream(svc, "priya", **{k: v for k, v in params.items() if k in ("gen_model", "judge_model")}, n_items=32):
        pass
    done, total, rows = last
    assert (done, total) == (32, 32) and not any(r["error"] for r in rows)
    assert all(r["grounded"] == 4 for r in rows)
    md = ev.summary_markdown(rows, params)
    assert "By brand" in md and "By platform" in md and "By campaign goal" in md and "Swiggy" in md and "4.00" in md

    path = ev.write_csv(svc, "priya", rows, params)
    data = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    assert len(data) == 32 and data[0]["gen_model"] == "gpt-5-nano" and data[0]["judge_model"] == "gpt-5.4-nano"
    assert svc.limiter.used("priya", "eval") == 32


def test_validation_errors(svc):
    with pytest.raises(StoreError, match="different model"):
        list(ev.evaluate_stream(svc, "priya", "gpt-5-nano", "gpt-5-nano"))
    with pytest.raises(StoreError, match="not enabled|allowed"):
        list(ev.evaluate_stream(svc, "priya", "gpt-5-nano", "gpt-evil"))
    with pytest.raises(StoreError, match="not found"):
        list(ev.evaluate_stream(svc, "priya", "gpt-5-nano", "gpt-5.4-nano"))  # brands not set up yet
    assert svc.limiter.used("priya", "eval") == 0


def test_failed_items_are_refunded_and_do_not_crash(settings):
    def broken(messages, model, s, max_output_tokens=0, json_mode=False):
        raise LLMError("model down")

    svc = Service(settings, llm_fn=broken, fetch_fn=lambda url, s, session=None: FetchResult(url, "text/html", HTML),
                  embedder=HashEmbedder(), search_fn=lambda q, n: [], wiki_fn=lambda n: [])
    list(ev.setup_brands_stream(svc, "priya"))
    rows = None
    for _, _, rows in ev.evaluate_stream(svc, "priya", "gpt-5-nano", "gpt-5.4-nano", n_items=8):
        pass
    assert len(rows) == 8 and all("model down" in r["error"] for r in rows)
    assert svc.limiter.used("priya", "eval") == 0  # refunded
    assert "0 scored items" in ev.summary_markdown(rows, {"gen_model": "a", "judge_model": "b", "top_k": 6, "hybrid": 0.5})


def test_pick_sources_stays_on_brand_domain():
    from agent.discover import Candidate
    cands = [Candidate("a", "https://www.brand.com/about", "", "search"), Candidate("b", "https://evil.wixsite.com/brand", "", "search"),
             Candidate("w", "https://en.wikipedia.org/wiki/Brand", "", "wikipedia"), Candidate("s", "https://brand.com/", "", "site")]
    urls = ev.pick_sources(cands, "brand.com")
    assert urls == ["https://www.brand.com/about", "https://en.wikipedia.org/wiki/Brand", "https://brand.com/"]
