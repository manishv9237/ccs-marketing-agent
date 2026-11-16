"""Sign-up accounts: hashing, duplicates, validation, invite code, persistence through the backend."""
import dataclasses
import json
import shutil

import pytest

from agent import persist
from agent.accounts import AccountError, AccountStore, account_id


def store(settings, sync=None, **over):
    return AccountStore(dataclasses.replace(settings, **over) if over else settings, sync)


def test_sign_up_and_verify(settings):
    a = store(settings)
    acct = a.sign_up("Priya", "S", "Priya@Example.com", "longenough1", "longenough1")
    assert acct["id"] == account_id("priya@example.com") and acct["name"] == "Priya"
    assert a.verify("priya@example.com", "longenough1")["id"] == acct["id"]
    assert a.verify("PRIYA@example.com", "longenough1") is not None
    assert a.verify("priya@example.com", "wrong-password") is None
    assert a.verify("nobody@example.com", "longenough1") is None


def test_password_is_not_stored_in_clear(settings):
    a = store(settings)
    a.sign_up("Priya", "", "p@example.com", "S3cretPassw0rd")
    raw = next(a.dir.glob("u*.json")).read_text()
    assert "S3cretPassw0rd" not in raw and json.loads(raw)["hash"]


@pytest.mark.parametrize("args,msg", [
    (("", "x", "a@b.co", "longenough1", "longenough1"), "first name"),
    (("A", "x", "not-an-email", "longenough1", "longenough1"), "e-mail"),
    (("A", "x", "a@b.co", "short", "short"), "at least"),
    (("A", "x", "a@b.co", "longenough1", "different1"), "do not match"),
])
def test_validation(settings, args, msg):
    with pytest.raises(AccountError, match=msg):
        store(settings).sign_up(*args)


def test_duplicate_email_rejected(settings):
    a = store(settings)
    a.sign_up("A", "", "a@b.co", "longenough1")
    with pytest.raises(AccountError, match="already exists"):
        a.sign_up("A", "", "A@B.co", "anotherpass1")


def test_invite_code_and_closed_signup(settings):
    a = store(settings, signup_code="class-2026")
    with pytest.raises(AccountError, match="invite code"):
        a.sign_up("A", "", "a@b.co", "longenough1", code="wrong")
    assert a.sign_up("A", "", "a@b.co", "longenough1", code="class-2026")
    with pytest.raises(AccountError, match="closed"):
        store(settings, signup_enabled=False).sign_up("B", "", "b@b.co", "longenough1")


def test_accounts_survive_wipe_with_backend(settings):
    mem = persist.MemoryBackend()
    store(settings, mem).sign_up("Priya", "", "p@example.com", "longenough1")
    shutil.rmtree(settings.data_dir / "accounts")
    b = store(settings, mem)
    assert b.verify("p@example.com", "longenough1") is not None
    assert ("_accounts", f"accounts/{account_id('p@example.com')}.json") in mem.rows
