"""Self-service accounts (sign up / log in) with hashed passwords.

Each account is one small JSON file: <DATA_DIR>/accounts/<id>.json holding names, e-mail and a
salted scrypt hash (never the password). When Supabase is configured (agent.persist) every account
file is also copied to the `workspace_files` table under the reserved owner "_accounts", and restored
from there on start, so accounts survive restarts of a free host. Without Supabase, accounts live only
on the server's disk and disappear when the host resets it.

The `id` doubles as the workspace name (private brands per account): "u" + 12 hex chars of the
SHA-256 of the lower-cased e-mail, so it is stable, valid for the store, and reveals nothing.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import threading
import time
from pathlib import Path

from .config import Settings

EMAIL_RE = re.compile(r"^[^@\s]{1,64}@[^@\s]{1,190}\.[A-Za-z]{2,24}$")
OWNER = "_accounts"
MIN_PASSWORD = 8
_LOCK = threading.RLock()


class AccountError(ValueError):
    """User-facing sign-up / login problem."""


def account_id(email: str) -> str:
    return "u" + hashlib.sha256(email.strip().lower().encode()).hexdigest()[:12]


def _hash(password: str, salt: bytes) -> str:
    return hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1, dklen=32).hex()


class AccountStore:
    def __init__(self, settings: Settings, sync=None) -> None:
        self.s = settings
        self.sync = sync
        self.dir = settings.data_dir / "accounts"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._restored = False

    def _restore(self) -> None:
        if self.sync is not None and not self._restored:
            self.sync.restore(OWNER, "accounts", self.dir)
            self._restored = True

    def _path(self, aid: str) -> Path:
        return self.dir / f"{aid}.json"

    def _read(self, aid: str) -> dict | None:
        self._restore()
        p = self._path(aid)
        if not p.exists():
            if self.sync is not None:  # created on another instance since we last restored
                self.sync.restore(OWNER, "accounts", self.dir)
            if not p.exists():
                return None
        return json.loads(p.read_text(encoding="utf-8"))

    def count(self) -> int:
        self._restore()
        return len(list(self.dir.glob("u*.json")))

    # ------------------------------------------------------------------ sign up
    def sign_up(self, first: str, last: str, email: str, password: str, confirm: str | None = None, code: str = "") -> dict:
        first, last, email = (first or "").strip(), (last or "").strip(), (email or "").strip().lower()
        if not self.s.signup_enabled:
            raise AccountError("Sign-up is closed. Ask the project team for a login.")
        if self.s.signup_code and not hmac.compare_digest((code or "").strip().encode(), self.s.signup_code.encode()):
            raise AccountError("That invite code is not valid.")
        if not (1 <= len(first) <= 40) or len(last) > 40:
            raise AccountError("Please enter your first name (up to 40 characters).")
        if not EMAIL_RE.match(email):
            raise AccountError("Please enter a valid e-mail address.")
        if len(password or "") < MIN_PASSWORD or len(password) > 128:
            raise AccountError(f"Choose a password of at least {MIN_PASSWORD} characters.")
        if confirm is not None and confirm != password:
            raise AccountError("The two passwords do not match.")
        aid = account_id(email)
        with _LOCK:
            if self._read(aid) is not None:
                raise AccountError("An account with this e-mail already exists. Please log in.")
            salt = secrets.token_bytes(16)
            rec = {"id": aid, "first_name": first, "last_name": last, "email": email, "salt": salt.hex(),
                   "hash": _hash(password, salt), "created": time.time()}
            tmp = self._path(aid).with_suffix(".tmp")
            tmp.write_text(json.dumps(rec), encoding="utf-8")
            os.replace(tmp, self._path(aid))
            if self.sync is not None:
                self.sync.push_file(OWNER, f"accounts/{aid}.json", self._path(aid))
        return {"id": aid, "name": first, "full_name": f"{first} {last}".strip()}

    # ------------------------------------------------------------------- profile
    def get(self, aid: str) -> dict | None:
        """{"first_name", "last_name", "email", "full_name"} for a self-service account, else None."""
        if not re.match(r"^u[0-9a-f]{12}$", aid or ""):
            return None
        rec = self._read(aid)
        if not rec:
            return None
        return {"first_name": rec["first_name"], "last_name": rec["last_name"], "email": rec["email"],
                "full_name": f"{rec['first_name']} {rec['last_name']}".strip()}

    def _save(self, rec: dict) -> None:
        tmp = self._path(rec["id"]).with_suffix(".tmp")
        tmp.write_text(json.dumps(rec), encoding="utf-8")
        os.replace(tmp, self._path(rec["id"]))
        if self.sync is not None:
            self.sync.push_file(OWNER, f"accounts/{rec['id']}.json", self._path(rec["id"]))

    def update_profile(self, aid: str, first: str, last: str) -> dict:
        first, last = (first or "").strip(), (last or "").strip()
        if not (1 <= len(first) <= 40) or len(last) > 40:
            raise AccountError("Please enter your first name (up to 40 characters) and a last name of up to 40.")
        with _LOCK:
            rec = self._read(aid) if re.match(r"^u[0-9a-f]{12}$", aid or "") else None
            if not rec:
                raise AccountError("This login does not have an editable profile.")
            rec["first_name"], rec["last_name"] = first, last
            self._save(rec)
        return {"name": first, "full_name": f"{first} {last}".strip()}

    def change_password(self, aid: str, old: str, new: str, confirm: str) -> None:
        with _LOCK:
            rec = self._read(aid) if re.match(r"^u[0-9a-f]{12}$", aid or "") else None
            if not rec:
                raise AccountError("This login does not have an editable password.")
            if not hmac.compare_digest(_hash(old or "", bytes.fromhex(rec["salt"])), rec["hash"]):
                time.sleep(0.5)
                raise AccountError("Your current password is not correct.")
            if len(new or "") < MIN_PASSWORD or len(new) > 128:
                raise AccountError(f"Choose a new password of at least {MIN_PASSWORD} characters.")
            if new != confirm:
                raise AccountError("The two new passwords do not match.")
            salt = secrets.token_bytes(16)
            rec["salt"], rec["hash"] = salt.hex(), _hash(new, salt)
            self._save(rec)

    # ------------------------------------------------------------------- log in
    def verify(self, email: str, password: str) -> dict | None:
        """Returns {"id", "name"} for correct credentials, else None (same cost either way)."""
        email = (email or "").strip().lower()
        rec = self._read(account_id(email)) if EMAIL_RE.match(email) else None
        salt = bytes.fromhex(rec["salt"]) if rec else b"\x00" * 16
        digest = _hash(password or "", salt)
        if rec and hmac.compare_digest(digest, rec["hash"]):
            return {"id": rec["id"], "name": rec["first_name"], "full_name": f"{rec['first_name']} {rec['last_name']}".strip()}
        time.sleep(0.5)
        return None
