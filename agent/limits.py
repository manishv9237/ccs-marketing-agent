"""Daily usage caps so a public deployment cannot drain the shared API budget."""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone

from .config import Settings

_LOCK = threading.Lock()


class LimitReached(RuntimeError):
    pass


class UsageLimiter:
    def __init__(self, settings: Settings) -> None:
        self.path = settings.data_dir / "usage.json"
        settings.data_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _today() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _load(self) -> dict:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        today = self._today()
        return {today: data.get(today, {})}  # drop old days

    def _save(self, data: dict) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        os.replace(tmp, self.path)

    def used(self, user: str, action: str) -> int:
        with _LOCK:
            return self._load()[self._today()].get(user, {}).get(action, 0)

    def remaining(self, user: str, action: str, per_user: int) -> int:
        return max(0, per_user - self.used(user, action))

    def consume(self, user: str, action: str, per_user: int, global_cap: int | None = None, amount: int = 1) -> None:
        """Reserve `amount` uses or raise LimitReached. Call BEFORE doing the paid work."""
        with _LOCK:
            data = self._load()
            day = data[self._today()]
            u = day.setdefault(user, {})
            if u.get(action, 0) + amount > per_user:
                raise LimitReached(f"Daily limit reached for this action ({per_user}/day). Try again tomorrow (UTC).")
            if global_cap is not None:
                total = sum(v.get(action, 0) for k, v in day.items() if k != "_all")
                if total + amount > global_cap:
                    raise LimitReached("The shared demo has hit its daily budget. Please try again tomorrow.")
            u[action] = u.get(action, 0) + amount
            self._save(data)

    def refund(self, user: str, action: str, amount: int = 1) -> None:
        with _LOCK:
            data = self._load()
            u = data[self._today()].setdefault(user, {})
            u[action] = max(0, u.get(action, 0) - amount)
            self._save(data)
