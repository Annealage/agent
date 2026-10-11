"""Durable project-scoped intent/outcome ledger for remote side effects.

Records contain canonical argument digests and whitelisted provenance only;
remote arguments are never persisted. Writes use an atomic snapshot under an
OS file lock so concurrent workers and restarts share one replay decision.
"""
import contextlib
import fcntl
import hashlib
import json
import os
import re
import secrets
import stat
import threading
from pathlib import Path


MAX_RECORDS = 2000
MAX_FILE_BYTES = 5 * 1024 * 1024
UNKNOWN = "UNKNOWN"
_PROJECT_ID_RE = re.compile(r"[0-9a-f]{64}\Z")
_MANIFEST_RE = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_LABEL = re.compile(r"[A-Za-z0-9_.:@/-]{1,160}\Z")

STATUS_REQUEST_SCHEMA = {
    "type": "object",
    "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 100}},
    "additionalProperties": False,
}
RECONCILE_REQUEST_SCHEMA = {
    "type": "object",
    "properties": {
        "invocation_id": {"type": "string", "maxLength": 64},
        "expected_digest": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "expected_context": {"type": "object"},
        "risk_reason": {"type": "string", "minLength": 1, "maxLength": 1000},
        "confirm_unverified": {"type": "boolean", "const": True},
    },
    "required": [
        "invocation_id", "expected_digest", "expected_context",
        "risk_reason", "confirm_unverified",
    ],
    "additionalProperties": False,
}



def project_hash(project_id):
    """Opaque, stable project scope; never retain the host path or raw id."""
    return hashlib.sha256(str(project_id).encode("utf-8")).hexdigest()


def safe_context(raw, default_project):
    """Normalize a trusted host callback result to whitelisted provenance."""
    if not isinstance(raw, dict):
        raw = {}
    raw_project = raw.get("project_id")
    manifest = raw.get("source_manifest")
    epoch = raw.get("authority_epoch")
    project_valid = isinstance(raw_project, str) and _PROJECT_ID_RE.fullmatch(raw_project)
    manifest_valid = isinstance(manifest, str) and _MANIFEST_RE.fullmatch(manifest)
    epoch_valid = isinstance(epoch, int) and not isinstance(epoch, bool) and epoch >= 0
    verified = bool(project_valid and manifest_valid and epoch_valid)
    return {
        "project": raw_project if project_valid else default_project,
        "source_manifest": manifest if manifest_valid else UNKNOWN,
        "authority_epoch": epoch if epoch_valid else UNKNOWN,
        "context_verified": verified,
    }


class RemoteActionStore:
    """Atomic project-scoped records preventing blind replay of remote writes."""

    def __init__(self, path, *, project_id, session=None):
        self.path = Path(path)
        self.project = project_hash(project_id)
        self.session = str(session) if session is not None else None
        self._lock_path = self.path.with_name(self.path.name + ".lock")
        self._thread_lock = threading.RLock()
        self._owner = secrets.token_hex(16)
        self._records = []
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._check_path(self.path.parent, directory=True)
        with self._locked():
            self._refresh()
            self._recover_abandoned()

    @staticmethod
    def _check_path(path, *, directory=False):
        if path.is_symlink():
            raise OSError("remote action ledger path must not be a symlink")
        try:
            info = path.stat()
        except FileNotFoundError:
            return
        if info.st_uid != os.geteuid():
            raise PermissionError("remote action ledger path has a different owner")
        if directory:
            if not stat.S_ISDIR(info.st_mode) or info.st_mode & 0o022:
                raise PermissionError("remote action ledger directory is not privately owned")
        elif not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077:
            raise PermissionError("remote action ledger file is not private")

    @contextlib.contextmanager
    def _locked(self):
        with self._thread_lock:
            self._check_path(self.path.parent, directory=True)
            flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(self._lock_path, flags, 0o600)
            try:
                info = os.fstat(fd)
                if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid():
                    raise PermissionError("remote action ledger lock is not privately owned")
                os.fchmod(fd, 0o600)
                fcntl.flock(fd, fcntl.LOCK_EX)
                yield
            finally:
                os.close(fd)

    @staticmethod
    def intent(server, tool, arguments, project, session=None):
        encoded = json.dumps(
            arguments, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()
        key = hashlib.sha256(
            "\0".join((str(project), str(server), str(tool), digest)).encode("utf-8")
        ).hexdigest()
        return key, digest

    def _refresh(self):
        self._check_path(self.path)
        try:
            fd = os.open(self.path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        except FileNotFoundError:
            self._records = []
            return
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & 0o077:
                raise PermissionError("remote action ledger file is not private")
            if info.st_size > MAX_FILE_BYTES:
                raise OSError("remote action ledger exceeds its size limit")
            with os.fdopen(fd, "r", encoding="utf-8") as stream:
                fd = -1
                data = json.load(stream)
        finally:
            if fd >= 0:
                os.close(fd)
        if not isinstance(data, dict) or data.get("version") != 1 or not isinstance(data.get("records"), list):
            raise OSError("remote action ledger is malformed")
        rows = data["records"]
        if len(rows) > MAX_RECORDS:
            raise OSError("remote action ledger exceeds its record limit")
        required = {"id", "intent_key", "server", "tool", "argument_digest", "project",
                    "session", "source", "context", "outcome", "phase", "owner", "owner_pid"}
        for row in rows:
            if not isinstance(row, dict) or not required.issubset(row) or row["project"] != self.project:
                raise OSError("remote action ledger contains an invalid record")
        self._records = rows

    def _save(self):
        self._prune()
        payload = json.dumps({"version": 1, "records": self._records}, sort_keys=True,
                             separators=(",", ":")).encode("utf-8")
        if len(payload) > MAX_FILE_BYTES:
            raise OSError("remote action ledger exceeds its size limit")
        name = ".%s.%s.tmp" % (self.path.name, secrets.token_hex(8))
        temporary = self.path.parent / name
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            self._check_path(self.path)
            os.replace(temporary, self.path)
            dirfd = os.open(self.path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(dirfd)
            finally:
                os.close(dirfd)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    def _prune(self):
        if len(self._records) <= MAX_RECORDS:
            return
        removable = [i for i, row in enumerate(self._records)
                     if row["phase"] == "finished" and row["outcome"] != "uncertain"]
        remove_count = len(self._records) - MAX_RECORDS
        remove = set(removable[:remove_count])
        self._records = [row for i, row in enumerate(self._records) if i not in remove]
        if len(self._records) > MAX_RECORDS:
            raise OSError("remote action ledger is full of unresolved records")

    def _recover_abandoned(self):
        changed = False
        for row in self._records:
            if row["phase"] in ("preparing", "dispatching") and not _pid_alive(row.get("owner_pid")):
                if row["phase"] == "dispatching":
                    row["outcome"] = "uncertain"
                else:
                    row["outcome"] = "not_sent"
                row["phase"] = "finished"
                changed = True
        if changed:
            self._save()

    def _latest(self, intent_key):
        return next((row for row in reversed(self._records) if row["intent_key"] == intent_key), None)

    def begin(self, server, tool, arguments, *, source, context, approval=None):
        ctx = safe_context(context, self.project)
        intent_key, digest = self.intent(server, tool, arguments, self.project)
        with self._locked():
            self._refresh()
            self._recover_abandoned()
            previous = self._latest(intent_key)
            if previous and (previous["phase"] != "finished" or previous["outcome"] == "uncertain"):
                return None, dict(previous)
            row = {
                "id": secrets.token_hex(16), "intent_key": intent_key,
                "server": str(server), "tool": str(tool), "argument_digest": digest,
                "project": self.project, "session": self.session, "source": _label(source),
                "context": ctx, "approval": _approval(approval), "outcome": "not_sent",
                "phase": "preparing", "owner": self._owner, "owner_pid": os.getpid(),
            }
            self._records.append(row)
            self._save()
            return dict(row), None

    def dispatching(self, invocation):
        return self._transition(invocation, outcome="uncertain", phase="dispatching")

    def _transition(self, invocation, *, outcome, phase):
        with self._locked():
            self._refresh()
            self._recover_abandoned()
            index = next((i for i, row in enumerate(self._records) if row["id"] == invocation["id"]), None)
            if index is None or self._records[index]["owner"] != self._owner:
                raise OSError("remote action reservation is no longer owned")
            self._records[index].update(outcome=outcome, phase=phase)
            self._save()
            return dict(self._records[index])

    def outcome(self, invocation, outcome):
        if outcome not in ("not_sent", "reported_result", "uncertain"):
            raise ValueError("invalid remote action outcome")
        return self._transition(invocation, outcome=outcome, phase="finished")

    def blocked(self, server, tool, arguments, *, context=None):
        """Whether intent is in flight/uncertain; authority changes don't alter its identity."""
        key, _ = self.intent(server, tool, arguments, self.project)
        with self._locked():
            self._refresh()
            self._recover_abandoned()
            row = self._latest(key)
            return dict(row) if row and (row["phase"] != "finished" or row["outcome"] == "uncertain") else None

    def status(self, *, limit=100):
        limit = max(1, min(int(limit), 100))
        with self._locked():
            self._refresh()
            self._recover_abandoned()
            rows = self._records[-limit:]
        fields = ("id", "server", "tool", "argument_digest", "project", "session", "source",
                  "context", "outcome", "phase", "evidence")
        public = [{key: row.get(key) for key in fields} for row in rows]
        for item, row in zip(public, rows):
            reconciliation = row.get("reconciliation")
            if reconciliation:
                item["reconciliation"] = {
                    "operator": reconciliation["operator"],
                    "outcome_unverified": True,
                }
        return public

    def reconcile(self, invocation_id, *, operator, risk_reason, confirm_unverified,
                  expected_digest, expected_context):
        """Authenticated host caller records an explicitly unverified outcome."""
        risk = str(risk_reason) if isinstance(risk_reason, str) else ""
        if not operator or not _SAFE_LABEL.fullmatch(str(operator)):
            raise ValueError("authenticated operator identity is required")
        if not isinstance(risk_reason, str) or not risk.strip() or len(risk) > 1000:
            raise ValueError("a bounded risk reason is required")
        if confirm_unverified is not True:
            raise ValueError("explicit unverified-outcome confirmation is required")
        with self._locked():
            self._refresh()
            row = next((r for r in self._records if r["id"] == invocation_id), None)
            if not row or row["phase"] != "finished" or row["outcome"] != "uncertain":
                raise ValueError("no uncertain invocation to reconcile")
            if expected_digest != row["argument_digest"] or expected_context != row["context"]:
                raise ValueError("invocation changed; refresh status before reconciling")
            row.update(outcome="reported_result", evidence="operator_unverified", phase="finished",
                       reconciliation={"operator": str(operator), "risk_reason": risk.strip(),
                                       "outcome_unverified": True})
            self._save()
            return dict(row)
def _pid_alive(pid):
    if not isinstance(pid, int) or pid < 1:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _label(value):
    value = str(value)
    return value if _SAFE_LABEL.fullmatch(value) else "UNKNOWN"


def _approval(value):
    if not isinstance(value, str) or not _SAFE_LABEL.fullmatch(value):
        return None
    return value
