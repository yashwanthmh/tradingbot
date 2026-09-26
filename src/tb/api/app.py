"""The dashboard server: read-only views of the ledger, one button that only stops.

What it must never become is a second way to trade. Nothing here places,
cancels or releases anything. The one write is the kill switch, which can only
stop trading; releasing it stays with `tb resume` at a terminal, with a reason.

Three things keep it that way from outside the process:

* **Where it answers.** Without a token it serves loopback peers only,
  whatever address it was bound to, and `tb dashboard` refuses to listen
  anywhere else without one.
* **Who may ask.** With `TB_DASHBOARD_TOKEN` set, every `/api` request needs it
  as a bearer token, and the kill switch needs it always: a control that
  writes the ledger answers only to a credential. The token comes from the
  environment alone and appears in no response, log line or error.
* **What a browser may do.** A Host allow-list stops DNS rebinding — a hostile
  page pointing its own name at 127.0.0.1 to read this one as same-origin —
  and every response carries a policy that runs only the dashboard's own
  script, so a string from the ledger that reaches the page stays text.
"""

from __future__ import annotations

import mimetypes
from pathlib import Path

from fastapi import FastAPI
from starlette.datastructures import MutableHeaders
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.staticfiles import StaticFiles
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from tb.api.deps import DashboardSettings, check_token
from tb.api.routes import control, read

STATIC_DIR = Path(__file__).resolve().parent.parent / "ops" / "dashboard"

# The page is served with `nosniff`, under which a browser runs a script only
# if it arrives labelled as one. The platform's type map decides the label,
# and some map `.js` to text/plain; these two are not left to it.
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/css", ".css")
CONTENT_SECURITY_POLICY = "; ".join(
    (
        "default-src 'none'",
        "script-src 'self'",
        "style-src 'self'",
        "connect-src 'self'",
        "img-src 'self' data:",
        "base-uri 'none'",
        "form-action 'none'",
        "frame-ancestors 'none'",
    )
)
SECURITY_HEADERS: dict[str, str] = {
    "Content-Security-Policy": CONTENT_SECURITY_POLICY,
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-origin",
}


class SecurityHeaders:
    """The same headers on every response, the static page's included."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def sending(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                for name, value in SECURITY_HEADERS.items():
                    headers[name] = value
            await send(message)

        await self.app(scope, receive, sending)


def create_app(settings: DashboardSettings) -> FastAPI:
    """The dashboard, bound to one ledger and one set of limits."""
    if settings.token is not None:
        check_token(settings.token)
    app = FastAPI(
        title="tb dashboard",
        # The generated docs pull scripts from a CDN, which the policy above
        # forbids, and a schema browser is not something this surface needs.
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.settings = settings
    app.include_router(read.router)
    app.include_router(control.router)
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="dashboard")
    # Added innermost first: the host check runs before anything else, and the
    # headers wrap even its refusal.
    app.add_middleware(
        TrustedHostMiddleware, allowed_hosts=list(settings.allowed_hosts), www_redirect=False
    )
    app.add_middleware(SecurityHeaders)
    return app
