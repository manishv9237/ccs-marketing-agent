"""Entry point.  Local:  python app.py      Docker/Render/HF Spaces: see README."""
from __future__ import annotations

import inspect
import logging
import os
import sys
import threading

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")  # no telemetry from a public demo

from agent.config import get_settings
from agent.security import check_login
from agent.service import Service

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")


def main() -> None:
    s = get_settings()

    if not s.access_code and not s.users and not s.allow_no_auth:
        sys.exit("Refusing to start: set APP_USERS (username:password list) or APP_ACCESS_CODE (shared code). For local-only testing set ALLOW_NO_AUTH=1.")
    if not s.openai_api_key:
        log.warning("OPENAI_API_KEY is not set - the app will start but generation will fail.")
    if s.allow_no_auth and not s.access_code and not s.users:
        log.warning("Running WITHOUT login (ALLOW_NO_AUTH=1). Never do this on a public host.")

    svc = Service(s)
    threading.Thread(target=lambda: svc.embedder, daemon=True, name="warm-embedder").start()  # load model in background

    from ui import build_ui  # imported late so config errors above show first

    demo = build_ui(svc)
    auth = (lambda user, pw: check_login(user, pw, s)) if (s.access_code or s.users) else None
    launch_kwargs = dict(
        server_name=os.getenv("HOST", "0.0.0.0"),
        server_port=int(os.getenv("PORT", "7860")),
        auth=auth,
        auth_message=("Log in with the username and password you were given. Your brands are private to your username." if s.users else
                      "Enter any display name (letters/numbers) and the shared access code you were given. Your brands are private to your display name."),
        show_api=False,
        max_file_size=f"{s.max_upload_mb}mb",
    )
    # Gradio renames/removes launch() options between versions: drop unknown cosmetic ones,
    # but never start without login when a login is required.
    supported = set(inspect.signature(demo.launch).parameters)
    if auth is not None and "auth" not in supported:
        sys.exit("This Gradio version does not support login (auth). Refusing to start unprotected.")
    dropped = [k for k in launch_kwargs if k not in supported]
    if dropped:
        log.warning("Ignoring launch options this Gradio version doesn't support: %s", ", ".join(dropped))
    demo.queue(default_concurrency_limit=4, max_size=40).launch(**{k: v for k, v in launch_kwargs.items() if k in supported})


if __name__ == "__main__":
    main()
