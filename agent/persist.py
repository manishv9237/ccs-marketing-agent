"""Optional durable storage so brands and evaluation results survive restarts of a free host.

The app keeps working from local files (fast, simple). When SUPABASE_URL and SUPABASE_SERVICE_KEY
are set, every change is also copied into one Supabase table (`workspace_files`) and anything
missing locally is restored from it when a user's workspace is opened.
Without those variables nothing here is active (from_settings returns None).

Remote layout: one row per file: (username, path, data_b64, updated_at), path like
"workspaces/<brand_id>/meta.json" or "evals/evaluation-<stamp>.csv".
"""
from __future__ import annotations

import base64
import logging
from pathlib import Path

log = logging.getLogger("persist")
TABLE = "workspace_files"
MAX_FILE_BYTES = 20 * 1024 * 1024


class MemoryBackend:
    """In-memory stand-in with the same interface; used by the tests (and as a reference)."""

    def __init__(self) -> None:
        self.rows: dict[tuple[str, str], str] = {}

    # raw row operations -------------------------------------------------
    def _put(self, user: str, path: str, data: bytes) -> None:
        self.rows[(user, path)] = base64.b64encode(data).decode()

    def _list(self, user: str, prefix: str) -> list[str]:
        return sorted(p for (u, p) in self.rows if u == user and p.startswith(prefix))

    def _get(self, user: str, path: str) -> bytes | None:
        v = self.rows.get((user, path))
        return base64.b64decode(v) if v is not None else None

    def _delete_prefix(self, user: str, prefix: str) -> None:
        for k in [k for k in self.rows if k[0] == user and k[1].startswith(prefix)]:
            del self.rows[k]

    # public interface ----------------------------------------------------
    def push_dir(self, user: str, remote_prefix: str, local_dir: Path) -> None:
        """Copy every file in local_dir to remote_prefix/<name> (best effort, never raises)."""
        try:
            for f in Path(local_dir).iterdir():
                if f.is_file() and not f.name.endswith(".tmp") and f.stat().st_size <= MAX_FILE_BYTES:
                    self._put(user, f"{remote_prefix}/{f.name}", f.read_bytes())
        except Exception:  # noqa: BLE001 - persistence must never break the app
            log.exception("push failed for %s/%s", user, remote_prefix)

    def push_file(self, user: str, remote_path: str, local_file: Path) -> None:
        try:
            self._put(user, remote_path, Path(local_file).read_bytes())
        except Exception:  # noqa: BLE001
            log.exception("push failed for %s/%s", user, remote_path)

    def delete_prefix(self, user: str, prefix: str) -> None:
        try:
            self._delete_prefix(user, prefix.rstrip("/") + "/")
        except Exception:  # noqa: BLE001
            log.exception("delete failed for %s/%s", user, prefix)

    def restore(self, user: str, remote_prefix: str, local_root: Path) -> int:
        """Download files under remote_prefix that are missing locally. Returns how many were written.
        remote_prefix "workspaces" maps to local_root itself (files land at local_root/<brand_id>/<name>)."""
        n = 0
        try:
            for path in self._list(user, remote_prefix + "/"):
                rel = path[len(remote_prefix) + 1:]
                if ".." in rel.split("/") or rel.startswith("/"):
                    continue
                dest = Path(local_root) / rel
                if dest.exists():
                    continue
                data = self._get(user, path)
                if data is None:
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
                n += 1
        except Exception:  # noqa: BLE001
            log.exception("restore failed for %s/%s", user, remote_prefix)
        return n


class SupabaseBackend(MemoryBackend):
    """Same interface, rows stored through Supabase's REST API (PostgREST) with the service key."""

    def __init__(self, url: str, key: str, session=None) -> None:
        import requests

        self.base = url.rstrip("/") + "/rest/v1/" + TABLE
        self.http = session or requests.Session()
        self.http.headers.update({"apikey": key, "Authorization": f"Bearer {key}"})

    def _put(self, user: str, path: str, data: bytes) -> None:
        r = self.http.post(self.base + "?on_conflict=username,path", timeout=30,
                           headers={"Prefer": "resolution=merge-duplicates,return=minimal", "Content-Type": "application/json"},
                           json={"username": user, "path": path, "data_b64": base64.b64encode(data).decode()})
        r.raise_for_status()

    def _list(self, user: str, prefix: str) -> list[str]:
        r = self.http.get(self.base, timeout=30, params={"select": "path", "username": f"eq.{user}", "path": f"like.{_like(prefix)}*"})
        r.raise_for_status()
        return sorted(x["path"] for x in r.json())

    def _get(self, user: str, path: str) -> bytes | None:
        r = self.http.get(self.base, timeout=60, params={"select": "data_b64", "username": f"eq.{user}", "path": f"eq.{path}"})
        r.raise_for_status()
        rows = r.json()
        return base64.b64decode(rows[0]["data_b64"]) if rows else None

    def _delete_prefix(self, user: str, prefix: str) -> None:
        r = self.http.delete(self.base, timeout=30, params={"username": f"eq.{user}", "path": f"like.{_like(prefix)}*"})
        r.raise_for_status()


def _like(prefix: str) -> str:
    return prefix.replace("%", r"\%").replace("_", r"\_")


def from_settings(s):
    if getattr(s, "supabase_url", "") and getattr(s, "supabase_service_key", ""):
        return SupabaseBackend(s.supabase_url, s.supabase_service_key)
    return None
