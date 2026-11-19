import dataclasses

import pytest

from agent.accounts import AccountError, AccountStore
from agent.embeddings import HashEmbedder
from agent.service import Service


def test_brand_kit_create_edit(settings):
    svc = Service(settings, llm_fn=lambda *a, **k: "{}", embedder=HashEmbedder())
    kit = {"audience": "Indian households 25-45", "positioning": "Everyday wellness for families", "avoid": "cheap, miracle",
           "tone": {"formal_casual": 80, "reserved_bold": 50, "factual_emotive": 20}, "colours": ["#00843d", "bad", "#f4a900"], "notes": ""}
    b = svc.create_brand("u1", "Marico", "FMCG & Consumer Brands", kit=kit)
    assert b["kit"]["colours"] == ["#00843d", "#f4a900"] and b["kit"]["tone"]["formal_casual"] == 80
    docs = svc.list_documents("u1", b["brand_id"])
    assert len(docs) == 1 and docs[0]["name"] == "Brand voice & guidelines (typed)"
    kit["audience"] = "Young professionals"
    b2 = svc.update_brand("u1", b["brand_id"], "Marico Ltd", "FMCG & Consumer Brands", kit)
    assert b2["name"] == "Marico Ltd" and b2["kit"]["audience"] == "Young professionals"
    assert len(svc.list_documents("u1", b["brand_id"])) == 1  # replaced, not duplicated


def test_profile_and_password(settings):
    s = dataclasses.replace(settings, signup_enabled=True, signup_code="")
    a = AccountStore(s)
    acct = a.sign_up("Priya", "Shah", "priya@example.com", "longenough1", "longenough1")
    assert acct["full_name"] == "Priya Shah" and a.get(acct["id"])["email"] == "priya@example.com"
    assert a.update_profile(acct["id"], "Priyanka", "S")["full_name"] == "Priyanka S"
    with pytest.raises(AccountError):
        a.change_password(acct["id"], "wrongpass", "newpassword1", "newpassword1")
    with pytest.raises(AccountError):
        a.change_password(acct["id"], "longenough1", "newpassword1", "different")
    a.change_password(acct["id"], "longenough1", "newpassword1", "newpassword1")
    assert a.verify("priya@example.com", "longenough1") is None
    assert a.verify("priya@example.com", "newpassword1")["full_name"] == "Priyanka S"
    assert a.get("priya") is None
    with pytest.raises(AccountError):
        a.update_profile("priya", "X", "Y")
