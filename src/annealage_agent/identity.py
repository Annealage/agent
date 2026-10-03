"""Who a request came from: a browser token, tailnet login or hosted principal.

Every browser-gated route asks ``BrowserAuth.authenticate`` and gets a
``Human`` or ``None``. Standalone deployments keep their browser-token and
Tailscale rules. A hosted worker accepts only a short-lived operation-scoped
delegation token verified by its injected ``HostedVerifier``; the agent
package does not parse hosted tokens or depend on the hosted control plane.

The Tailscale headers are ambient authority, like a cookie: serve adds them to
whatever the browser sends, including a request a page on another site makes it
send. The Origin check prevents cross-site requests, but not programs on a
trusted device. Such an app must be bound to loopback and must not use an agent
backend with a shell of its own, which could forge a login and approve its own
permission cards.
"""

import dataclasses
import os
import re
import stat
import unicodedata
from email.errors import HeaderParseError
from email.header import decode_header, make_header
from pathlib import Path
from typing import FrozenSet, Iterable, Optional, Protocol

from .http.ws import _origin_is_allowed, _token_is_allowed


class HostedClaims(Protocol):
    """The validated, job-scoped identity and authority from the hosted front."""

    sub: str
    wsp: str
    prj: str
    ops: tuple[str, ...]
    job: str
    rev: str
    exp: int


class HostedVerifier(Protocol):
    """Verify a hosted delegation token for the operation being attempted."""

    def verify(self, token: str, requested_operation: str) -> HostedClaims: ...


DELEGATION_HEADER = "Authorization"
PRINCIPAL_DISPLAY_HEADER = "X-Annealage-Principal-Display"
_DISPLAY_LIMIT = 120

#: The header serve puts the tailnet login in (``andrew@example.com``, or
#: ``someone@github`` for a GitHub-backed tailnet).
#: The header serve puts the account's display name in, RFC 2047 encoded when
#: it is not ASCII.
#: The users-file entry that allows any login serve vouches for.
# One login: printable ASCII with no space and no comma, which is what a proxy
# that folded two copies of the header into one line would leave. ASCII only,
# so comparing without regard to case is plain ASCII case folding: Unicode
# ``lower()`` maps some other characters onto ASCII letters (the Kelvin sign
# onto ``k``), which would let a login an identity provider issued stand in for
# a different, allowed one. Tailnet logins are email-like and ASCII in practice.
LOGIN_HEADER = "Tailscale-User-Login"
NAME_HEADER = "Tailscale-User-Name"
ANY_LOGIN = "*"
_LOGIN_RE = re.compile(r"[\x21-\x2b\x2d-\x7e]+")
_SAFE_METHODS = frozenset(("GET", "HEAD"))


@dataclasses.dataclass(frozen=True)
class Human:
    """The human a request came from, retaining local and hosted authorship."""

    login: Optional[str] = None
    name: Optional[str] = None
    principal_id: Optional[str] = None
    workspace_id: Optional[str] = None
    project_id: Optional[str] = None
    hosted_claims: Optional[HostedClaims] = None

    @property
    def label(self) -> Optional[str]:
        """What to call them, with the hosted display snapshot first."""
        return self.name or self.login or None


class TailscaleIdentity:
    """The tailnet logins allowed to act as the human.

    ``allowed`` is the logins, compared without regard to ASCII case, or the
    single entry ``*`` for any login serve vouches for: every untagged device
    on the tailnet (serve adds no login for a tagged device), and also anyone
    outside it the device has been shared with, whatever identity provider
    they signed in with. List the logins unless the device is not shared. A
    login that is not printable ASCII is never allowed, ``*`` included.
    """

    def __init__(self, allowed: Iterable[str]):
        if isinstance(allowed, str):
            # Checked before iterating: a bare login would otherwise become
            # one allowed "login" per character.
            raise TypeError("allowed is a collection of logins, not one login: %r" % allowed)
        logins = set()
        for login in allowed:
            if not isinstance(login, str) or not (login == ANY_LOGIN or _LOGIN_RE.fullmatch(login)):
                raise ValueError("not a tailnet login: %r" % (login,))
            logins.add(login.lower())
        if ANY_LOGIN in logins and len(logins) > 1:
            raise ValueError(
                "%s allows every login, so it must be the only entry, not one among %d"
                % (ANY_LOGIN, len(logins))
            )
        self._logins: FrozenSet[str] = frozenset(logins)

    @property
    def logins(self) -> FrozenSet[str]:
        """The allowed logins, lowercased; ``{"*"}`` when any is."""
        return self._logins

    def allows(self, login: str) -> bool:
        """Whether ``login`` may act as the human."""
        if not isinstance(login, str) or not _LOGIN_RE.fullmatch(login):
            return False
        if ANY_LOGIN in self._logins:
            return True
        return login.lower() in self._logins

    def from_request(self, req) -> Optional[Human]:
        """The ``Human`` serve says ``req`` came from, if they are allowed.

        ``Tailscale-User-Login`` is required: one login, nothing else in the
        value. microdot keeps only the last of a repeated header line, and
        serve replaces whatever the client sent, so a value that looks like
        two joined by a comma is refused rather than split. The name is
        optional and RFC 2047 decoded; one that does not decode, or decodes to
        something with a control character in it, is replaced by the login.

        Says nothing about whether the request itself may be trusted to carry
        the header: ``BrowserAuth.authenticate`` decides that from its Origin.
        """
        login = req.headers.get(LOGIN_HEADER)
        if not isinstance(login, str):
            return None
        login = login.strip()
        if not _LOGIN_RE.fullmatch(login) or not self.allows(login):
            return None
        return Human(login=login, name=_display_name(req.headers.get(NAME_HEADER)) or login)


def _display_name(raw) -> Optional[str]:
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        name = str(make_header(decode_header(raw))).strip()
    except (HeaderParseError, LookupError, UnicodeError, ValueError):
        return None
    if not name or any(unicodedata.category(ch) == "Cc" for ch in name):
        return None
    return name


def load_users(path) -> TailscaleIdentity:
    """The ``TailscaleIdentity`` of the users file at ``path``: one login per
    line, ``#`` to the end of a line a comment, blank lines ignored.

    The file grants authority, so it is refused (``ValueError``) when it is a
    symlink, is not a regular file, is owned by another user, or is writable by
    anyone but its owner, the checks ``net.load_token`` makes. Others may read
    it: a login is not a secret. An empty file is allowed: nobody signs in by
    login, and the browser token still works. The checks and the read go
    through one file descriptor, so the file checked is the file read.
    """
    path = Path(path)
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise ValueError(
            "the users file %s cannot be opened as a plain file (%s); it must exist and "
            "must not be a symlink" % (path, exc.strerror or exc)
        ) from None
    with os.fdopen(fd, "r", encoding="utf-8") as f:
        st = os.fstat(f.fileno())
        if not stat.S_ISREG(st.st_mode):
            raise ValueError("the users file %s is not a regular file" % path)
        if hasattr(os, "getuid") and st.st_uid != os.getuid():
            raise ValueError("the users file %s is owned by another user" % path)
        if st.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
            raise ValueError(
                "the users file %s is writable by other users (mode %o); chmod 644 or "
                "600 it" % (path, stat.S_IMODE(st.st_mode))
            )
        try:
            text = f.read()
        except UnicodeDecodeError:
            raise ValueError("the users file %s is not UTF-8 text" % path) from None
    logins = []
    for number, line in enumerate(text.splitlines(), start=1):
        entry = line.split("#", 1)[0].strip()
        if not entry:
            continue
        if entry != ANY_LOGIN and not _LOGIN_RE.fullmatch(entry):
            raise ValueError(
                "the users file %s, line %d, is not one tailnet login: %r" % (path, number, entry)
            )
        logins.append(entry)
    try:
        return TailscaleIdentity(logins)
    except ValueError as exc:
        raise ValueError("the users file %s: %s" % (path, exc)) from None


class BrowserAuth:
    """The one check every browser-gated route makes: ``authenticate(req)``.

    Hosted mode accepts only a verified, operation-scoped delegation token.
    Local browser-token and Tailscale authentication remain unchanged.
    """

    def __init__(
        self,
        token: Optional[str],
        identity: Optional[TailscaleIdentity] = None,
        *,
        allowed_origins: Iterable[str] = (),
        hosted_mode: bool = False,
        hosted_verifier: Optional[HostedVerifier] = None,
    ):
        if hosted_mode and hosted_verifier is None:
            raise ValueError("hosted mode requires a delegation-token verifier")
        self._token = token
        self.identity = identity
        self._allowed_origins = frozenset(allowed_origins)
        self._hosted_mode = hosted_mode
        self._hosted_verifier = hosted_verifier

    def authenticate(self, req, requested_operation: Optional[str] = None) -> Optional[Human]:
        """Return the authenticated author, or ``None`` to refuse the request.

        Hosted callers must name the operation being attempted. The injected
        verifier checks the signature, expiry, audience and operation scope.
        """
        if not _origin_is_allowed(req, self._allowed_origins):
            return None
        if self._hosted_mode:
            if requested_operation is None:
                return None
            authorization = req.headers.get(DELEGATION_HEADER)
            if not isinstance(authorization, str) or not authorization.startswith("Bearer "):
                return None
            token = authorization[7:]
            if not token or token.strip() != token:
                return None
            try:
                claims = self._hosted_verifier.verify(token, requested_operation)
            except Exception:
                return None
            if requested_operation not in claims.ops:
                return None
            return Human(
                name=_hosted_display(req.headers.get(PRINCIPAL_DISPLAY_HEADER)),
                principal_id=claims.sub,
                workspace_id=claims.wsp,
                project_id=claims.prj,
                hosted_claims=claims,
            )
        has_origin = req.headers.get("Origin") is not None
        if self.identity is not None and (has_origin or _is_plain_read(req)):
            human = self.identity.from_request(req)
            if human is not None:
                return human
        if _token_is_allowed(req, self._token):
            return Human()
        return None


def _hosted_display(raw) -> Optional[str]:
    """Accept only the front's short plain-text author snapshot."""
    if not isinstance(raw, str):
        return None
    value = raw.strip()[:_DISPLAY_LIMIT]
    if not value or any(unicodedata.category(char).startswith("C") for char in value):
        return None
    return value


#: ``Sec-Fetch-Site`` values of a request the page itself (or the human, typing
#: the address) made. Absent too: a browser too old to send the header.
_OWN_FETCH_SITES = frozenset((None, "same-origin", "none"))


def _is_plain_read(req) -> bool:
    """A ``GET`` or ``HEAD`` that is not a WebSocket upgrade and that no other
    site made the browser send: a request whose answer changes nothing, and
    which a browser sends without an ``Origin`` when it is the page's own.

    Another site's ``<img>``, ``<link>`` or navigation carries no ``Origin``
    either, and though it cannot read the answer it can see whether an image
    loaded and how big it is. Every current browser says where a request came
    from in ``Sec-Fetch-Site``, so a cross-site or same-site one is not taken
    as the page's own."""
    return (
        req.method in _SAFE_METHODS
        and "Upgrade" not in req.headers
        and req.headers.get("Sec-Fetch-Site") in _OWN_FETCH_SITES
    )


def via(human: Human) -> str:
    """How ``human`` was authenticated, as ``GET /whoami`` reports it."""
    if human.hosted_claims is not None:
        return "hosted"
    return "tailscale" if human.login is not None else "token"


def check_bind(identity: Optional[TailscaleIdentity], bind) -> None:
    """Refuse (``ValueError``) an ``identity`` on a ``net.ResolvedBind`` that
    is not loopback, for ``create_app`` and ``FrontDoor``: serve's login
    headers can be trusted only on a port nothing but serve and this host can
    reach, and on any other bind anyone who reaches it can write them."""
    if identity is not None and not bind.is_loopback:
        raise ValueError(
            "a tailnet identity needs a loopback bind, not %s: tailscale serve's login "
            "headers can be trusted only on a port nothing but serve and this host can "
            "reach" % bind.address
        )


__all__ = [
    "ANY_LOGIN",
    "DELEGATION_HEADER",
    "LOGIN_HEADER",
    "NAME_HEADER",
    "PRINCIPAL_DISPLAY_HEADER",
    "BrowserAuth",
    "HostedClaims",
    "HostedVerifier",
    "Human",
    "TailscaleIdentity",
    "check_bind",
    "load_users",
    "via",
]
