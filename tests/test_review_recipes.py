import json

import pytest

from agent import review
from agent.embeddings import HashEmbedder
from agent.service import Service
from agent.store import StoreError


def test_paragraph_status():
    content = "We founded the company in 1998.\nOur factories reach net zero carbon by 2030."
    rows = review.paragraph_status(content, ["factories reach net zero carbon by 2030"], True)
    assert [r["status"] for r in rows] == ["grounded", "flagged"]
    assert [r["status"] for r in review.paragraph_status(content, [], False)] == ["unchecked", "unchecked"]


def llm(messages, model, s, max_output_tokens=0, json_mode=False):
    if "REWRITING" in messages[0]["content"] or messages[0]["content"].startswith("You are a senior marketing copywriter. Rewrite"):
        return json.dumps({"content": "Shorter and friendlier post."})
    return json.dumps({"variations": []})


def test_recipes_and_rewrite(settings):
    svc = Service(settings, llm_fn=llm, embedder=HashEmbedder())
    b = svc.create_brand("u1", "Acme", "FMCG & Consumer Brands")["brand_id"]
    svc.save_recipe("u1", b, "Launch kit", {"g_chan": ["linkedin"], "g_n": 3})
    assert svc.list_recipes("u1", b)["Launch kit"]["g_n"] == 3
    with pytest.raises(StoreError):
        svc.save_recipe("u1", b, "x", {})
    svc.delete_recipe("u1", b, "Launch kit")
    assert svc.list_recipes("u1", b) == {}
    var = {"number": 1, "angle": "A", "content": "Old post", "length": "ok", "sources": [1], "brand_note": "", "unsupported": [], "verified": True}
    new = svc.rewrite("u1", b, "linkedin", var, "shorter and friendlier", [{"text": "fact", "source": "s", "score": 1}], settings.default_model)
    assert new["content"] == "Shorter and friendlier post." and var["content"] == "Old post"
    with pytest.raises(StoreError):
        svc.rewrite("u1", b, "linkedin", var, "x", [], settings.default_model)
