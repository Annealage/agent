"""Bind-mode resolution, the per-run token, the Origin allowlist and the
startup banner.

These four things belong together and belong away from both routing and
argument parsing, because they are one decision with one output: what this
process is reachable on, and therefore what it should accept. The Origin
allowlist is computed from the resolved bind rather than hardcoded to
localhost, so remote access works at all; the banner reports the same
resolved bind back to the human, so exposure is never something they have to
infer. Putting this in a product's CLI (Mesh's ``cli.py``) would make it
untestable without argparse, and duplicated per product, and putting it in
``app.py`` would mix it with request handling.

``resolve_bind`` takes its two collaborators, ``run`` and ``which``, as
parameters so a test can supply canned command output instead of depending on
whether a real ``tailscale`` or ``ip`` binary happens to be installed on the
machine running the suite.
"""

import dataclasses
import ipaddress
import json
import secrets
import shutil
import socket
import subprocess
from typing import Sequence

from . import product

# Tailscale hands out addresses from the RFC 6598 carrier-grade NAT range.
# An address found while looking for a tailnet interface that falls outside
# this range is a false positive (a container bridge, another VPN) and is
# refused rather than bound: binding it would expose this process on a
# network the user did not ask for, which is the one outcome the alias
# exists to prevent.
TAILNET_RANGE = ipaddress.ip_network("100.64.0.0/10")

# Interface names whose addresses are noise in a reachable-address list: a
# loopback address tells the reader nothing about who else can reach the
# socket, and docker's bridge is reachable only by containers on this host.
_UNINTERESTING_INTERFACES = ("lo",)

DEFAULT_HOST = "127.0.0.1"
TAILSCALE_ALIAS = "tailscale"


class BindError(Exception):
    """A requested bind could not be resolved.

    Raised, never worked around. A caller that catches this must fail
    startup: the alternative, falling back to a default, hands the user a
    wider bind than they asked for, which for ``--host tailscale`` means
    exposing the tool on every interface after a request to expose it on
    exactly one.
    """


@dataclasses.dataclass(frozen=True)
class ResolvedBind:
    """What one ``--host`` value resolved to, and what follows from it.

    ``address`` is the literal string to hand to ``bind()``. ``mode`` is
    ``"loopback"``, ``"explicit"`` or ``"tailscale"``, which is what decides
    the banner's wording rather than re-deriving intent from the address.
    ``is_loopback`` gates both the banner's no-TLS warning and the Origin
    allowlist's localhost forms. ``reachable`` is ``[(interface, address)]``
    for a banner to enumerate, and is empty for a loopback bind because
    there is nothing for such a list to add.

    Frozen so a resolved bind cannot be edited after the banner has reported
    it: the printed exposure and the actual bind must not be able to drift.
    """

    address: str
    mode: str
    is_loopback: bool
    reachable: Sequence = ()


def resolve_bind(host_arg, *, run=subprocess.run, which=shutil.which):
    """Resolve a ``--host`` value to a ``ResolvedBind``, or raise ``BindError``.

    ``None`` means the default, loopback. ``"tailscale"`` is the alias
    resolved below. Anything else is taken literally, after being checked:
    an IP address is used as given, and a hostname must resolve, because a
    name that does not resolve would otherwise reach ``bind()`` as an
    unhandled ``socket.gaierror`` at startup rather than as an explanation.

    ``0.0.0.0`` is deliberately not treated as loopback even though it also
    answers there. A request arriving over the loopback path of a wildcard
    bind says nothing about who else can reach the same listening socket, so
    calling it loopback would suppress the banner's no-TLS warning for a
    bind that is in fact reachable from the whole LAN.
    """
    if host_arg is None:
        return ResolvedBind(DEFAULT_HOST, "loopback", True, ())
    host_arg = host_arg.strip()
    if not host_arg:
        raise BindError("--host was given an empty value")
    if host_arg == TAILSCALE_ALIAS:
        return _resolve_tailscale(run=run, which=which)

    address = _literal_or_resolved_address(host_arg)
    if _is_loopback_address(address):
        return ResolvedBind(address, "loopback", True, ())
    return ResolvedBind(address, "explicit", False, _reachable_addresses(run=run))


def _literal_or_resolved_address(host_arg):
    """Return the address to bind for an explicit ``--host``, or raise."""
    try:
        return str(ipaddress.ip_address(host_arg))
    except ValueError:
        pass
    try:
        infos = socket.getaddrinfo(host_arg, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise BindError("--host %s does not resolve to an address (%s)" % (host_arg, exc)) from exc
    if not infos:
        raise BindError("--host %s resolved to no addresses" % host_arg)
    return infos[0][4][0]


def _is_loopback_address(address):
    try:
        return ipaddress.ip_address(address).is_loopback
    except ValueError:
        return False


def _resolve_tailscale(*, run, which):
    """Resolve the host's own tailnet IPv4 address, or raise ``BindError``.

    Three sources, in order, and the order is the point. ``tailscale ip -4``
    is authoritative and works on every platform, which matters because the
    interface is ``tailscale0`` on Linux and a ``utun`` on macOS, so guessing
    an interface name is not portable. The two ``ip`` fallbacks exist for a
    host running tailscaled without the CLI on PATH.

    Every failure is collected and named in the raised error, because "could
    not bind" leaves the operator guessing between tailscaled being down, not
    installed, or simply not yet having an address.
    """
    attempts = []
    address = None

    if which(TAILSCALE_ALIAS):
        address, note = _try_tailscale_cli(run)
        attempts.append(note)
    else:
        attempts.append("tailscale binary not on PATH")

    if address is None:
        address, note = _try_ip_json(run)
        attempts.append(note)
    if address is None:
        address, note = _try_ip_plaintext(run)
        attempts.append(note)

    if address is None:
        raise BindError("could not resolve a tailscale address; tried: %s" % "; ".join(attempts))
    # Checked after resolution, not inside each source, so one rule covers
    # all three and no source can be added later that skips it.
    if ipaddress.ip_address(address) not in TAILNET_RANGE:
        raise BindError(
            "tailscale resolution produced %s, which is outside the tailnet range "
            "%s; refusing to bind an address that is not on the tailnet" % (address, TAILNET_RANGE)
        )
    return ResolvedBind(address, TAILSCALE_ALIAS, False, [("tailscale0", address)])


def _try_tailscale_cli(run):
    try:
        result = run(
            [TAILSCALE_ALIAS, "ip", "-4"], capture_output=True, text=True, timeout=5, check=False
        )
    except Exception as exc:
        return None, "tailscale ip -4 failed (%s)" % exc
    if getattr(result, "returncode", 1) != 0:
        return None, "tailscale ip -4 exited %s" % getattr(result, "returncode", "?")
    for line in (result.stdout or "").splitlines():
        candidate = line.strip()
        if _is_ipv4(candidate):
            return candidate, "tailscale ip -4 reported %s" % candidate
    return None, "tailscale ip -4 reported no IPv4 address"


def _try_ip_json(run):
    try:
        result = run(
            ["ip", "-json", "addr", "show"], capture_output=True, text=True, timeout=5, check=False
        )
        entries = json.loads(result.stdout or "[]")
    except Exception as exc:
        return None, "ip -json addr show failed (%s)" % exc
    for entry in entries:
        name = entry.get("ifname", "")
        for addr in entry.get("addr_info", []):
            if addr.get("family") != "inet":
                continue
            candidate = addr.get("local", "")
            if _is_ipv4(candidate) and ipaddress.ip_address(candidate) in TAILNET_RANGE:
                return candidate, "ip -json addr show found %s on %s" % (candidate, name)
    return None, "ip -json addr show found no tailnet address"


def _try_ip_plaintext(run):
    try:
        result = run(
            ["ip", "-4", "addr", "show"], capture_output=True, text=True, timeout=5, check=False
        )
    except Exception as exc:
        return None, "ip -4 addr show failed (%s)" % exc
    for _name, candidate in _parse_ip_plaintext(result.stdout or ""):
        if ipaddress.ip_address(candidate) in TAILNET_RANGE:
            return candidate, "ip -4 addr show found %s" % candidate
    return None, "ip -4 addr show found no tailnet address"


def _parse_ip_plaintext(text):
    """Yield ``(interface, address)`` from ``ip -4 addr show`` output.

    Interface lines start at column zero as ``<index>: <name>: ...``; address
    lines are indented and start with ``inet``. Parsed rather than regexed so
    the shape being relied on is visible.
    """
    name = ""
    for raw in text.splitlines():
        stripped = raw.strip()
        if raw[:1].isdigit() and ": " in raw:
            name = raw.split(": ", 2)[1].strip()
            continue
        if stripped.startswith("inet "):
            candidate = stripped.split()[1].split("/")[0]
            if _is_ipv4(candidate):
                yield name, candidate


def _is_ipv4(value):
    try:
        return ipaddress.ip_address(value).version == 4
    except ValueError:
        return False


def _reachable_addresses(*, run):
    """Best-effort ``[(interface, address)]`` for the banner to enumerate.

    Informational only: it tells the human which addresses a non-loopback
    bind is answering on, and nothing decides anything from it. So every
    failure yields an empty list rather than propagating, and the ``except``
    is deliberately broad: ``run`` is an injected collaborator that may raise
    anything at all, and neither a missing ``ip`` binary nor output in a
    shape this does not recognise may turn an otherwise valid bind into a
    startup failure.
    """
    try:
        result = run(
            ["ip", "-json", "addr", "show"], capture_output=True, text=True, timeout=5, check=False
        )
        entries = json.loads(result.stdout or "[]")
        found = []
        for entry in entries:
            name = entry.get("ifname", "")
            if name in _UNINTERESTING_INTERFACES:
                continue
            for addr in entry.get("addr_info", []):
                if addr.get("family") == "inet":
                    found.append((name, addr.get("local", "")))
        return found
    except Exception:
        return []


def bind_from_address(address):
    """A ``ResolvedBind`` for an address that is already resolved.

    What ``create_app`` needs in order to compute its allowlists, with no
    subprocess and no interface enumeration: an app is built once per test and
    must not shell out to do it. ``resolve_bind`` remains the only entry point
    that interprets a user-supplied ``--host``, so the rule for what counts as
    loopback lives in one place and both callers get the same answer.
    """
    return ResolvedBind(
        address,
        "loopback" if _is_loopback_address(address) else "explicit",
        _is_loopback_address(address),
        (),
    )


def generate_token(explicit=None):
    """The per-run token: ``explicit`` when the user supplied ``--token``,
    otherwise 16 fresh random bytes as 22 URL-safe characters.

    Regenerated every run, and never written to a config file, so a token
    that leaks is worthless as soon as the process it belonged to exits.
    """
    if explicit:
        return explicit
    return secrets.token_urlsafe(16)


def allowed_origins(bind, port, extra_origins=()):
    """The exact set of ``Origin`` values ``/ws`` accepts, as strings.

    Computed from the resolved bind, never hardcoded, because a hardcoded
    localhost allowlist would refuse every remote viewer and make the
    tailscale bind mode useless.

    A loopback bind gets all three spellings a browser may send for itself.
    Any other bind gets its own address only, and deliberately **not** the
    loopback forms: a page loaded from this server always carries this
    server's address as its Origin, so a page claiming
    ``http://127.0.0.1:<port>`` against a LAN-bound server was never served
    by it, and accepting that claim would let a page which only ever needed
    loopback access to itself drive a socket reachable from the whole
    network.

    ``extra_origins`` is taken verbatim, which is what makes a reverse proxy
    and ``tailscale serve`` work: those front the server under a name and
    scheme this process cannot derive from its own bind.
    """
    origins = set()
    if bind.is_loopback:
        origins.add("http://127.0.0.1:%d" % port)
        origins.add("http://localhost:%d" % port)
        origins.add("http://[::1]:%d" % port)
    else:
        origins.add("http://%s:%d" % (_origin_host(bind.address), port))
    origins.update(extra_origins)
    return origins


def allowed_hosts(bind, port):
    """The exact set of ``Host`` header values any route accepts.

    Same shape as the Origin set and for the same reason, minus the scheme.
    A ``Host`` naming this server is the only one that can legitimately
    describe it, so an attacker-controlled name whose DNS answer has been
    repointed at this address is refused before reaching a route: that is
    the DNS-rebinding case, which the Origin check cannot catch because
    Origin describes where a page came from and says nothing about which
    server it is now talking to.

    ``localhost`` appears only for a loopback bind. It resolves on the
    requesting browser's own machine, which is never the machine a LAN or
    tailnet bind is answering on, so accepting it there would accept a name
    that can never truthfully name this server.

    Each name appears twice, with and without the port, so that the check
    itself can be exact set membership with no parsing whatsoever. That
    matters more than the redundancy costs: a check that split a port off the
    header would have to decide what to do with userinfo, a trailing dot, and
    unbalanced IPv6 brackets, and every one of those decisions is a chance to
    extract an address from a header that does not actually name it. The
    portless spellings are safe to accept for the same reason an absent
    ``Host`` is: a browser omits the port only when it is the scheme default,
    so on any other port a portless ``Host`` cannot have come from a browser,
    and on port 80 an attacker's rebound name still arrives as that name and
    still is not in this set.
    """
    hosts = set()
    if bind.is_loopback:
        names = ["127.0.0.1", "localhost", "[::1]"]
    else:
        names = [_origin_host(bind.address)]
    for name in names:
        hosts.add(name)
        hosts.add("%s:%d" % (name, port))
    return hosts


def _origin_host(address):
    """Bracket an IPv6 literal, which an Origin and a Host header both
    require; return anything else unchanged."""
    return "[%s]" % address if ":" in address else address


def server_url(bind, port):
    """The bind's address as a URL, with no token: what can be said about a
    running instance by someone who does not hold its token (a refused second
    start, say)."""
    return "http://%s:%d/" % (_origin_host(bind.address), port)


def login_url(bind, port, nonce):
    """The URL a browser is launched with: the bind's address plus a
    single-use login nonce as a fragment, which the page trades for the token
    (``http/routes_login.py`` says why the token itself is not used)."""
    return "%s#n=%s" % (server_url(bind, port), nonce)


def viewer_url(bind, port, token):
    """The URL to open: the bind's address plus the token as a fragment.

    A fragment is never sent to a server, so the token stays out of access
    logs, out of ``Referer`` headers and out of any proxy's log, while still
    travelling in a link the human can open or bookmark.
    """
    return "%s#t=%s" % (server_url(bind, port), token)


def format_banner(bind, port, token):
    """The startup banner, printed every run whatever the bind.

    Every run, because a default or a persisted setting means the user never
    typed a flag that would have reminded them what this is reachable on.
    The wording is security-relevant output: a change that quietly drops the
    no-TLS sentence is a change that leaves a user believing a LAN bind is
    private, so the tests assert on the text.
    """
    lines = []
    lines.append(
        "%s is serving %s on %s:%d"
        % (product.current().display_name, _describe_mode(bind), bind.address, port)
    )
    lines.append("  open: %s" % viewer_url(bind, port, token))

    if bind.is_loopback:
        lines.append("  reachable from this machine only")
        return "\n".join(lines)

    if bind.reachable:
        lines.append("  reachable on:")
        for name, address in bind.reachable:
            lines.append("    %s  %s:%d" % (name, address, port))

    if bind.mode == TAILSCALE_ALIAS:
        lines.append(
            "  the tailnet carries this over WireGuard, so the link and the "
            "token in it are encrypted in transit"
        )
        lines.append(
            "  stronger still, costing no extra setup here: bind loopback "
            "instead and run  tailscale serve https / http://127.0.0.1:%d" % port
        )
    else:
        # Spelled out rather than hinted at: on a plain LAN bind the token,
        # the model bytes and the whole conversation cross the network in
        # cleartext, and anyone on that network can read them.
        lines.append(
            "  WARNING: no TLS on this bind. The token in the URL, the model "
            "files and the conversation all cross the network in cleartext "
            "and can be read by anyone on it."
        )
        lines.append(
            "  for remote review, prefer  --host tailscale  or  tailscale "
            "serve https / http://127.0.0.1:%d" % port
        )
    return "\n".join(lines)


def _describe_mode(bind):
    if bind.mode == "loopback":
        return "on loopback"
    if bind.mode == TAILSCALE_ALIAS:
        return "on the tailnet"
    if bind.address == "0.0.0.0":
        return "on every interface"
    return "on one interface"
