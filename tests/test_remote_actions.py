import json

import pytest

from annealage_agent.remote_actions import RemoteActionStore


CONTEXT = {
    "project_id": "a" * 64,
    "source_manifest": "b" * 64,
    "authority_epoch": 1,
}


def _store(path, session="session"):
    return RemoteActionStore(path, project_id="/project", session=session)


def test_intent_digest_excludes_arguments_and_blocks_uncertain_across_sessions(tmp_path):
    path = tmp_path / "actions.json"
    first = _store(path)
    invocation, blocked = first.begin(
        "datum", "submit", {"token": "secret", "document": "private"},
        source="upload:submit", approval="human-1", context=CONTEXT,
    )
    assert blocked is None
    first.dispatching(invocation)
    first.outcome(invocation, "uncertain")

    restarted = _store(path, session="new-session")
    retry, blocked = restarted.begin(
        "datum", "submit", {"document": "private", "token": "secret"},
        source="upload:submit", approval="new-human", context=CONTEXT,
    )
    assert retry is None
    assert blocked["id"] == invocation["id"]
    assert blocked["outcome"] == "uncertain"
    persisted = path.read_text()
    assert "secret" not in persisted
    assert "private" not in persisted
    assert "/project" not in persisted


def test_not_sent_can_be_retried_and_reconciliation_requires_explicit_context(tmp_path):
    store = _store(tmp_path / "actions.json")
    args = {"x": 1}
    invocation, _ = store.begin("s", "write", args, source="agent", context=CONTEXT)
    store.outcome(invocation, "not_sent")
    retry, blocked = store.begin("s", "write", args, source="agent", context=CONTEXT)
    assert retry is not None and blocked is None
    store.dispatching(retry)
    store.outcome(retry, "uncertain")

    with pytest.raises(ValueError):
        store.reconcile(
            retry["id"], operator="human", risk_reason="unknown", confirm_unverified=False,
            expected_digest=retry["argument_digest"], expected_context=retry["context"],
        )
    reconciled = store.reconcile(
        retry["id"], operator="human", risk_reason="Checked manually; effect remains unknown",
        confirm_unverified=True, expected_digest=retry["argument_digest"],
        expected_context=retry["context"],
    )
    assert reconciled["outcome"] == "reported_result"
    assert reconciled["evidence"] == "operator_unverified"
    assert reconciled["reconciliation"]["outcome_unverified"] is True
    assert store.blocked("s", "write", args, context=CONTEXT) is None


def test_status_is_bounded_and_omits_approval_and_remote_arguments(tmp_path):
    path = tmp_path / "actions.json"
    store = _store(path)
    invocation, _ = store.begin(
        "s", "write", {"password": "hidden"}, source="agent", approval="secret-login",
        context=CONTEXT,
    )
    store.dispatching(invocation)
    store.outcome(invocation, "uncertain")
    status = store.status(limit=1000)
    assert len(status) == 1
    assert "approval" not in status[0]
    assert "password" not in json.dumps(status)
    assert "secret-login" not in json.dumps(status)


def test_malformed_and_symlink_ledgers_fail_closed(tmp_path):
    path = tmp_path / "actions.json"
    path.write_text('{"version":1,"records":[')
    with pytest.raises((ValueError, OSError)):
        _store(path)
    path.unlink()
    target = tmp_path / "target.json"
    target.write_text('{"version":1,"records":[]}')
    path.symlink_to(target)
    with pytest.raises(OSError):
        _store(path)


def test_preparing_reservation_blocks_concurrent_writer(tmp_path):
    path = tmp_path / "actions.json"
    store1 = _store(path)
    first, _ = store1.begin("s", "write", {"x": 1}, source="agent", context=CONTEXT)
    store2 = _store(path)
    second, blocked = store2.begin("s", "write", {"x": 1}, source="agent", context=CONTEXT)
    assert second is None
    assert blocked["id"] == first["id"]
    store1.outcome(first, "not_sent")


def test_real_context_keeps_opaque_project_hash_and_integer_epoch():
    from annealage_agent.remote_actions import safe_context

    context = {
        "project_id": "a" * 64,
        "source_manifest": "b" * 64,
        "authority_epoch": 12,
    }
    normalized = safe_context(context, "fallback")
    assert normalized["project"] == "a" * 64
    assert normalized["source_manifest"] == "b" * 64
    assert normalized["authority_epoch"] == 12
    assert normalized["context_verified"] is True
    other = dict(context, project_id="c" * 64)
    assert safe_context(other, "fallback")["project"] != normalized["project"]
    unknown = safe_context(
        {"project_id": "UNKNOWN", "source_manifest": "UNKNOWN", "authority_epoch": "UNKNOWN"},
        "fallback",
    )
    assert unknown["context_verified"] is False
    assert unknown["source_manifest"] == "UNKNOWN"
