"""``POST /login``: trade a single-use login nonce for the browser token.

The browser a run opens automatically is launched with a URL on a command line
(``webbrowser.open`` runs ``xdg-open``, or the browser itself, with the URL as
an argument), and a command line is readable by every process on the machine
through ``ps`` for as long as that process lives, which for a browser started
this way is the whole session. The browser token in that URL would let
anything that read it, the agent's own shell included, open ``/ws`` and
approve its own permission cards.

So the URL the run opens carries a nonce instead (``#n=<nonce>``), and the page
trades it here, once, for the browser token, which it then keeps in memory
exactly as it keeps a token read from ``#t=``. A nonce is spent by its first
use, successful or not for anyone else, and expires after ``NONCE_TTL``
seconds whether or not it was used, so what a later reader of the command line
finds is a string that opens nothing. The window that remains is a process
racing the browser to the nonce in the first seconds; it can win it only by
also making the human's own tab fail, which the human sees.

The URL printed in the startup banner still carries the reusable ``#t=`` token:
a second tab, another device on the tailnet and a phone all need a link that
works more than once, and the banner goes to the terminal, not to a command
line.

This route accepts nothing but an outstanding nonce. It does not accept the
browser token (a caller holding it has no need of this route) and never the
agent token.
"""

import hmac
import secrets
import time

from . import read_json_body
from .ws import _origin_is_allowed, refusal

#: How long an issued nonce stays redeemable. Long enough for a browser that
#: was not running to start and load the page; short enough that a nonce read
#: off a command line later is worthless.
NONCE_TTL = 60.0


class LoginNonces:
    """The outstanding single-use login nonces for one run.

    ``clock`` is the time source, ``time.monotonic`` unless a test supplies one,
    so expiry can be tested without sleeping.
    """

    def __init__(self, ttl=NONCE_TTL, clock=time.monotonic):
        self._ttl = ttl
        self._clock = clock
        self._pending = {}

    def issue(self):
        """A fresh nonce, redeemable once within ``ttl`` seconds from now."""
        nonce = secrets.token_urlsafe(16)
        self._pending[nonce] = self._clock() + self._ttl
        return nonce

    def redeem(self, supplied):
        """Spend ``supplied`` if it is an outstanding, unexpired nonce, and say
        whether it was. Expired nonces are dropped on the way, and each
        comparison is constant-time, so a wrong guess learns nothing from how
        long the refusal took."""
        now = self._clock()
        for nonce, expires in list(self._pending.items()):
            if expires <= now:
                del self._pending[nonce]
        if not isinstance(supplied, str) or not supplied:
            return False
        for nonce in list(self._pending):
            if hmac.compare_digest(
                nonce.encode("utf-8"), supplied.encode("utf-8", "surrogatepass")
            ):
                del self._pending[nonce]
                return True
        return False


def register_login_routes(app, *, token, nonces, allowed_origins=()):
    """Register ``POST /login`` on ``app``.

    The body is ``{"nonce": "<nonce>"}``; the answer is
    ``{"ok": true, "token": <browser token>}`` for an outstanding nonce, and
    the same refusal every other auth failure returns for anything else. A run
    with no browser token (``token`` of ``None``) has nothing to hand out and
    refuses every request.
    """

    @app.post("/login")
    async def login(req):
        if not _origin_is_allowed(req, allowed_origins):
            return refusal()
        data, error = await read_json_body(req)
        if error is not None:
            return refusal()
        nonce = data.get("nonce") if isinstance(data, dict) else None
        if not token or not nonces.redeem(nonce):
            return refusal()
        return {"ok": True, "token": token}, 200
