"""Tests for bind-mode resolution: loopback default, an explicit ``--host``,
and Tailscale discovery, plus the token and the startup banner that go with a
resolved bind.

This file was written before ``net.py`` existed (Annealage Mesh's M4
bind-mode module), so every test below imports it lazily, through the ``net``
fixture, rather than at module level: a missing module then failed each test
individually with a clear ``ModuleNotFoundError``, instead of one opaque
collection error hiding every property this file is meant to pin.

It was written against an assumed contract, inferred from the M4 brief's
bind-mode table and banner spec since no implementation existed to read
instead:

    class BindError(Exception): ...

    @dataclass(frozen=True)
    class ResolvedBind:
        address: str          # the literal address to bind()
        mode: str             # "loopback" | "explicit" | "tailscale"
        is_loopback: bool
        reachable: list       # [(iface_name, address), ...] for a banner to
                               # enumerate; empty for a loopback bind

    def resolve_bind(host_arg, *, run=subprocess.run,
                      which=shutil.which) -> ResolvedBind: ...

    def generate_token(explicit=None) -> str: ...

    def allowed_origins(bind: ResolvedBind, port: int,
                         extra_origins=()) -> set[str]: ...

    def format_banner(bind: ResolvedBind, port: int, token: str) -> str: ...

``run`` and ``which`` are dependency-injected (the same pattern
``viewers.py``'s constructor already uses for its own collaborators)
specifically so a test can supply canned subprocess output instead of
depending on a real ``tailscale`` or ``ip`` binary being present, or absent,
on whatever machine runs the suite.

Whoever implements ``net.py`` should reconcile this contract with reality;
where this file's assumptions turn out wrong, the fix is to the test file,
not a silent rename in ``net.py`` that leaves the tests wrong for a
different reason.
"""

import ipaddress
import string

import pytest

pytestmark = pytest.mark.asyncio


@pytest.fixture
def net():
    import annealage_agent.net as net_module

    return net_module


class _Completed:
    """Just enough of ``subprocess.CompletedProcess`` for a stubbed ``run``."""

    def __init__(self, stdout="", returncode=0):
        self.stdout = stdout
        self.returncode = returncode


def _dispatch(table, absent_ok=False):
    """Build a fake ``run(argv, ...)`` that answers only the argv tuples in
    ``table``. Any other argv is a test bug (an untested command actually
    being invoked), so it raises loudly rather than returning a plausible
    default that would hide the gap."""

    def run(argv, *args, **kwargs):
        key = tuple(argv)
        if key not in table:
            raise AssertionError("unstubbed command: %r" % (argv,))
        result = table[key]
        if isinstance(result, Exception):
            raise result
        return result

    return run


def _which_only(*names):
    present = set(names)
    return lambda name: ("/usr/bin/" + name) if name in present else None


# ---------------------------------------------------------------------------
# Loopback default and explicit addresses.
# ---------------------------------------------------------------------------


async def test_no_host_arg_resolves_to_loopback(net):
    bind = net.resolve_bind(None)
    assert bind.address == "127.0.0.1"
    assert bind.is_loopback is True


async def test_explicit_127_resolves_to_loopback(net):
    bind = net.resolve_bind("127.0.0.1")
    assert bind.address == "127.0.0.1"
    assert bind.is_loopback is True


async def test_localhost_resolves_to_loopback(net):
    # The brief allows "localhost" as an explicit --host value alongside a
    # literal IP or a resolvable hostname; it is still loopback traffic, so
    # the banner's no-TLS warning (which loopback binds must not carry, see
    # below) must not fire for it.
    bind = net.resolve_bind("localhost")
    assert bind.is_loopback is True


async def test_explicit_lan_address_accepted_and_not_loopback(net):
    bind = net.resolve_bind("192.0.2.10")
    assert bind.address == "192.0.2.10"
    assert bind.mode == "explicit"
    assert bind.is_loopback is False


async def test_wildcard_bind_is_not_treated_as_loopback(net):
    # 0.0.0.0 also answers on 127.0.0.1, but a client's Origin/Host arriving
    # over that loopback path tells you nothing about who else can reach the
    # same listening socket from off-box; treating the wildcard as loopback
    # would suppress the no-TLS banner warning for a bind that is, in fact,
    # reachable from the whole LAN.
    bind = net.resolve_bind("0.0.0.0")
    assert bind.address == "0.0.0.0"
    assert bind.is_loopback is False


async def test_unresolvable_host_raises_bind_error(net):
    # RFC 2606 reserves .invalid for names guaranteed never to resolve.
    with pytest.raises(net.BindError):
        net.resolve_bind("this-name.invalid")


# ---------------------------------------------------------------------------
# Tailscale resolution: binary present, binary absent with interface-scan
# fallback, and total failure.
# ---------------------------------------------------------------------------


async def test_tailscale_binary_reports_address(net):
    run = _dispatch(
        {
            ("tailscale", "ip", "-4"): _Completed(stdout="100.101.102.103\n"),
        }
    )
    bind = net.resolve_bind("tailscale", run=run, which=_which_only("tailscale"))
    assert bind.address == "100.101.102.103"
    assert bind.mode == "tailscale"
    assert bind.is_loopback is False


_IP_JSON_TAILSCALE0 = """[
  {"ifindex": 3, "ifname": "tailscale0", "operstate": "UNKNOWN",
   "addr_info": [{"family": "inet", "local": "100.64.5.6", "prefixlen": 32}]}
]"""

_IP_PLAINTEXT_TAILSCALE0 = """\
1: lo: <LOOPBACK,UP,LOWER_UP> mtu 65536 qdisc noqueue state UNKNOWN
    inet 127.0.0.1/8 scope host lo
       valid_lft forever preferred_lft forever
2: eth0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc fq_codel state UP
    inet 192.168.1.50/24 brd 192.168.1.255 scope global dynamic eth0
       valid_lft 86000sec preferred_lft 86000sec
3: tailscale0: <POINTOPOINT,MULTICAST,NOARP,UP,LOWER_UP> mtu 1280 qdisc fq_codel state UNKNOWN
    inet 100.64.7.8/32 scope global tailscale0
       valid_lft forever preferred_lft forever
"""


async def test_tailscale_binary_absent_falls_back_to_ip_json(net):
    run = _dispatch(
        {
            ("ip", "-json", "addr", "show"): _Completed(stdout=_IP_JSON_TAILSCALE0),
        }
    )
    bind = net.resolve_bind("tailscale", run=run, which=_which_only())
    assert bind.address == "100.64.5.6"
    assert bind.mode == "tailscale"


async def test_ip_json_unavailable_falls_back_to_ip_dash_4(net):
    run = _dispatch(
        {
            ("ip", "-json", "addr", "show"): FileNotFoundError("no -json support"),
            ("ip", "-4", "addr", "show"): _Completed(stdout=_IP_PLAINTEXT_TAILSCALE0),
        }
    )
    bind = net.resolve_bind("tailscale", run=run, which=_which_only())
    assert bind.address == "100.64.7.8"
    assert bind.mode == "tailscale"


async def test_tailscale_totally_unresolvable_raises_named_error(net):
    run = _dispatch(
        {
            ("ip", "-json", "addr", "show"): FileNotFoundError("no ip"),
            ("ip", "-4", "addr", "show"): FileNotFoundError("no ip"),
        }
    )
    with pytest.raises(net.BindError) as excinfo:
        net.resolve_bind("tailscale", run=run, which=_which_only())
    # Naming what was tried is the point: a bare "could not bind" leaves the
    # operator guessing whether tailscaled is down, not installed, or simply
    # has no address yet.
    assert "tailscale" in str(excinfo.value).lower()


async def test_tailscale_resolution_failure_never_returns_a_wider_bind(net):
    # A resolution failure must surface as an exception, never as a
    # returned ResolvedBind for 0.0.0.0 or any other address broader than
    # what was actually asked for: a caller that only checks "did this
    # raise" before falling through to a default bind must not be handed
    # a silent 0.0.0.0 that opens the server to the whole LAN. Raising is
    # the only outcome _dispatch's unstubbed-command guard permits here:
    # any code path that tried to compute a fallback address using a
    # command not stubbed above would fail this test via that guard
    # rather than by quietly returning something wider.
    run = _dispatch(
        {
            ("ip", "-json", "addr", "show"): FileNotFoundError("no ip"),
            ("ip", "-4", "addr", "show"): FileNotFoundError("no ip"),
        }
    )
    with pytest.raises(net.BindError):
        net.resolve_bind("tailscale", run=run, which=_which_only())


async def test_tailscale_address_outside_cgnat_range_rejected(net):
    # Tailscale's own address space is 100.64.0.0/10 (RFC 6598 CGNAT
    # range); an interface-scan false positive (a VPN or container bridge
    # that also happens to be named or numbered plausibly) must not be
    # accepted just because it was found while looking for tailscale0.
    run = _dispatch(
        {
            ("tailscale", "ip", "-4"): _Completed(stdout="10.0.0.5\n"),
        }
    )
    with pytest.raises(net.BindError):
        net.resolve_bind("tailscale", run=run, which=_which_only("tailscale"))


async def test_tailscale_address_is_verified_in_cgnat_range():
    # Companion to the rejection test above: confirms the accepted address
    # in the success-path tests actually falls inside the range being
    # enforced, so that test is not passing merely because nothing in it
    # exercises the range check at all. Takes no net fixture: this checks
    # the two literal addresses those tests use, not net.py's own logic.
    assert ipaddress.ip_address("100.101.102.103") in ipaddress.ip_network("100.64.0.0/10")
    assert ipaddress.ip_address("10.0.0.5") not in ipaddress.ip_network("100.64.0.0/10")


# ---------------------------------------------------------------------------
# Token generation.
# ---------------------------------------------------------------------------


async def test_generate_token_default_is_urlsafe_and_unpredictable(net):
    a = net.generate_token()
    b = net.generate_token()
    assert a != b
    allowed = set(string.ascii_letters + string.digits + "-_")
    assert set(a) <= allowed
    # secrets.token_urlsafe(16) yields 22 base64url characters (16 raw bytes,
    # no padding); a shorter default would make the token guessable within
    # a plausible number of /ws connection attempts.
    assert len(a) >= 22


async def test_generate_token_passes_through_explicit_value(net):
    assert net.generate_token("my-fixed-token") == "my-fixed-token"


# ---------------------------------------------------------------------------
# Allowed origins.
# ---------------------------------------------------------------------------


async def test_allowed_origins_for_loopback_bind(net):
    bind = net.resolve_bind("127.0.0.1")
    origins = net.allowed_origins(bind, 8765)
    assert origins == {
        "http://127.0.0.1:8765",
        "http://localhost:8765",
        "http://[::1]:8765",
    }


async def test_allowed_origins_for_explicit_bind_is_just_that_address(net):
    bind = net.resolve_bind("192.0.2.10")
    origins = net.allowed_origins(bind, 8765)
    assert "http://192.0.2.10:8765" in origins
    # A page loaded from this server always carries this server's own
    # address as its Origin; a page that instead claims to be
    # http://127.0.0.1:8765 was never served by this bind; and accepting it
    # anyway would let anything reachable at 192.0.2.10:8765 be driven by a
    # page that only ever needed loopback access to itself, which defeats
    # binding away from loopback in the first place.
    assert "http://127.0.0.1:8765" not in origins
    assert "http://localhost:8765" not in origins


async def test_allowed_origins_includes_extra_origins_verbatim(net):
    bind = net.resolve_bind("127.0.0.1")
    origins = net.allowed_origins(bind, 8765, extra_origins=("https://tail-abc.ts.net",))
    assert "https://tail-abc.ts.net" in origins


async def test_allowed_origins_uses_the_port_given_not_a_default(net):
    bind = net.resolve_bind("127.0.0.1")
    origins = net.allowed_origins(bind, 9999)
    assert "http://127.0.0.1:9999" in origins
    assert "http://127.0.0.1:8765" not in origins


# ---------------------------------------------------------------------------
# Startup banner.
# ---------------------------------------------------------------------------


async def test_banner_names_address_port_and_token_fragment(net):
    bind = net.resolve_bind("127.0.0.1")
    banner = net.format_banner(bind, 8765, "the-token")
    assert "127.0.0.1" in banner
    assert "8765" in banner
    assert "#t=the-token" in banner


async def test_banner_no_tls_warning_present_for_lan_bind(net):
    bind = net.resolve_bind("192.0.2.10")
    banner = net.format_banner(bind, 8765, "the-token").lower()
    assert "tls" in banner


async def test_banner_no_tls_warning_absent_for_loopback_bind(net):
    bind = net.resolve_bind("127.0.0.1")
    banner = net.format_banner(bind, 8765, "the-token").lower()
    assert "tls" not in banner


async def test_banner_mentions_tailscale_serve_for_tailscale_bind(net):
    run = _dispatch(
        {
            ("tailscale", "ip", "-4"): _Completed(stdout="100.101.102.103\n"),
        }
    )
    bind = net.resolve_bind("tailscale", run=run, which=_which_only("tailscale"))
    banner = net.format_banner(bind, 8765, "the-token").lower()
    assert "tailscale serve" in banner
    assert "wireguard" in banner


async def test_banner_lists_every_reachable_interface_for_nonloopback_bind(net):
    bind = net.resolve_bind("0.0.0.0")
    bind = _with_reachable(bind, [("eth0", "192.0.2.10"), ("wlan0", "192.0.2.11")])
    banner = net.format_banner(bind, 8765, "the-token")
    assert "eth0" in banner and "192.0.2.10" in banner
    assert "wlan0" in banner and "192.0.2.11" in banner


async def test_banner_does_not_enumerate_interfaces_for_loopback_bind(net):
    bind = net.resolve_bind("127.0.0.1")
    banner = net.format_banner(bind, 8765, "the-token")
    # A loopback bind is reachable from nowhere but this machine; there is
    # nothing for a per-interface list to add, and printing one would wrongly
    # imply other hosts on the LAN can reach it.
    assert "eth0" not in banner and "wlan0" not in banner


def _with_reachable(bind, reachable):
    """Return a copy of ``bind`` with its ``reachable`` list replaced.

    ``ResolvedBind`` is assumed frozen (see module docstring), so this
    goes through ``dataclasses.replace`` rather than mutating the instance
    the earlier ``resolve_bind`` call returned.
    """
    import dataclasses

    return dataclasses.replace(bind, reachable=reachable)
