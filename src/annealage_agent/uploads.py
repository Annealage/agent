"""Upload actions: something the human does with a file they upload, as
themselves, through the same approval card as the agent's calls.

A product declares them for an app, ``create_app(upload_actions=...)``::

    UploadAction(
        name="datum",
        label="Submit to Datum",
        tool=("ds-wiki", "submit_datasheet"),
        build_args=lambda upload, human: {"project_tag": "psu"},
    )

``tool`` is ``(remote, tool)``: one of the remote MCP servers the product's
tool server declares (``remote.RemoteServer``, by its name) and a tool on it.
``create_app`` refuses an action naming a remote the tool server does not
declare, and an app with no agent offers no actions (nothing could approve
one).

**The flow.** The page's attach button takes a file an action accepts (only
``application/pdf`` today) and uploads it as a document (``POST
/upload?kind=document``, ``http/routes_chat.py``), which the server sniffs by
its bytes and keeps outside the served tree, in the user's cache directory
(``documents_dir``), never serving it back, and whose size and SHA-256 it
notes (``Documents``). The page shows the file with a
button per action, and a click posts ``POST /upload/action``. Only the
browser can: the route takes the browser's credentials and ``Origin`` like
every other browser POST, and the agent token opens no browser route. The
server then asks ``build_args(upload, human)`` for the call's arguments and
puts the call in front of the human as a permission card through the live
session's broker, as it does the agent's write-grade calls: the tool's
model-visible name, and the arguments the call will carry, the file's name
among them and its bytes shown only as a size, marked as the human's own
action (``PermissionRequest.action`` and ``by``). No standing grant answers
it and no "always allow" is kept for it.

Approved, the file is read here, checked against the digest taken at upload,
and its bytes put in the call as ``content_arg`` (base64), beside
``filename_arg``, so they never pass through the model or back through the
page; a file that changed since is not sent. The call goes to the remote on a
connection of its own (``remote.call``), not through the agent's proxy: the
pause switch holds the agent's tools, not the human's own action. How it
ended (``UploadActionEnded``) is published for the page's conversation, and,
unless the human declined it, queued as a note for the agent's next turn
(``ViewerBus.queue_note``), saying what the remote answered, so the agent
can follow it up (poll a job, say). However it ended, the document is then
removed. Its card is the human's, not the agent's turn's, so interrupting
the agent leaves it open (``PermissionBroker.pending_requests``).
"""

import asyncio
import base64
import dataclasses
import hashlib
import os
import re
import secrets
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Tuple

from . import files, product, sessions
from .session.base import ACTION_DENIED, ACTION_DONE, ACTION_FAILED, UploadActionEnded

#: The media types an upload action may accept: what ``/upload`` keeps as a
#: document.
DOCUMENT_MEDIA_TYPES = ("application/pdf",)

#: The most of a remote's answer a note to the agent carries.
NOTE_RESULT_LIMIT = 4000

_NAME_RE = re.compile(r"[a-z][a-z0-9-]{0,31}")


@dataclasses.dataclass(frozen=True)
class Upload:
    """A document the human uploaded: its ``id`` (the stored file's name,
    which the page sends back), ``name`` (the human's own file name, made
    safe), size in ``bytes``, ``media_type``, where it is kept, and the
    ``sha256`` of the bytes as they were uploaded."""

    id: str
    name: str
    bytes: int
    media_type: str
    path: Path
    sha256: str = ""


@dataclasses.dataclass(frozen=True)
class UploadAction:
    """One thing the human can do with an uploaded file; see the module
    docstring.

    ``name`` is a short lowercase slug the page sends back, ``label`` its
    button and card text. ``build_args(upload, human)`` returns the call's
    arguments other than the file's: a JSON object, without
    ``content_arg`` or ``filename_arg``, which the server fills in. It runs
    when the human clicks, and a ``ValueError`` it raises refuses the click
    with its message. ``human`` is the ``identity.Human`` who clicked.
    """

    name: str
    label: str
    tool: Tuple[str, str]
    build_args: Callable[[Upload, Any], Mapping]
    accepts: str = "application/pdf"
    content_arg: str = "content_base64"
    filename_arg: str = "filename"

    def to_wire(self):
        """What the page is told of it (the hello's
        ``session.upload_actions``)."""
        return {"name": self.name, "label": self.label, "accepts": self.accepts}


def check_upload_actions(actions, tools=None):
    """``actions`` as a tuple, or ``ValueError``: not a tuple of
    ``UploadAction`` (a bare one is not), a name that is not a short
    lowercase slug or is given twice, no label, a ``tool`` that is not two
    names, a media type no upload is kept as, a ``build_args`` that cannot be
    called, the two argument names missing or equal, or, given ``tools`` (the
    app's ``ToolServer``), a remote it does not declare."""
    if isinstance(actions, UploadAction):
        raise ValueError("upload_actions is a tuple of UploadAction, not one")
    actions = tuple(actions)
    seen = set()
    for action in actions:
        if not isinstance(action, UploadAction):
            raise ValueError("upload action %r is not an UploadAction" % (action,))
        if not isinstance(action.name, str) or not _NAME_RE.fullmatch(action.name):
            raise ValueError("upload action name %r is not a short lowercase slug" % (action.name,))
        if action.name in seen:
            raise ValueError("upload action %r is declared twice" % action.name)
        seen.add(action.name)
        if not isinstance(action.label, str) or not action.label.strip():
            raise ValueError("upload action %r has no label" % action.name)
        tool = action.tool
        if not (
            isinstance(tool, tuple)
            and len(tool) == 2
            and all(isinstance(part, str) and part for part in tool)
        ):
            raise ValueError("upload action %r: tool is (remote, tool name)" % action.name)
        if action.accepts not in DOCUMENT_MEDIA_TYPES:
            raise ValueError(
                "upload action %r accepts %r; an upload is kept only as %s"
                % (action.name, action.accepts, ", ".join(DOCUMENT_MEDIA_TYPES))
            )
        if not callable(action.build_args):
            raise ValueError("upload action %r: build_args is not callable" % action.name)
        names = (action.content_arg, action.filename_arg)
        if not all(isinstance(n, str) and n for n in names) or names[0] == names[1]:
            raise ValueError(
                "upload action %r: content_arg and filename_arg are two argument names"
                % action.name
            )
        if tools is not None and tool[0] not in tools.remote_servers:
            raise ValueError(
                "upload action %r calls the %s MCP server, which the tool server does not "
                "declare as a remote" % (action.name, tool[0])
            )
    return actions


#: How long a document no action took is kept: the next app built for the
#: workspace removes anything older (``Documents``).
DOCUMENT_MAX_AGE = 24 * 60 * 60


def documents_dir(serve_dir):
    """Where the documents uploaded to the workspace at ``serve_dir`` are
    kept: the user's cache directory for the product, one directory per
    workspace, outside the served tree (``files.DOCUMENTS_DIRNAME`` says
    why)."""
    import platformdirs

    return (
        Path(platformdirs.user_cache_dir(product.current().config_dirname))
        / files.DOCUMENTS_DIRNAME
        / sessions.project_key_for_directory(serve_dir)
    )


class Documents:
    """The documents uploaded to one app, in ``directory``
    (``documents_dir``).

    ``record`` notes each one's size and SHA-256 as ``/upload`` wrote it, in
    this process's memory. Only a recorded upload can be found, and the
    bytes an action sends must still have that digest when read at approval
    (``run``), so a file changed on disk in between, by anything, is not
    what leaves. ``discard`` removes one (its action ended, or the human
    removed it). Building one removes whatever an earlier process left more
    than ``DOCUMENT_MAX_AGE`` seconds ago.
    """

    def __init__(self, directory, *, max_age=DOCUMENT_MAX_AGE):
        self.directory = Path(directory)
        self._known = {}
        self.sweep(max_age)

    def create(self, name):
        """``files.create_unique_document_file`` for a file the human named
        ``name``."""
        return files.create_unique_document_file(self.directory, files.document_stem(name))

    def record(self, upload_id, size, sha256):
        self._known[upload_id] = (size, sha256)

    def find(self, upload_id):
        """The ``Upload`` ``upload_id`` names, or None when this app did not
        record it or it is gone."""
        known = self._known.get(upload_id) if isinstance(upload_id, str) else None
        if known is None:
            return None
        found = files.open_document(self.directory, upload_id)
        if found is None:
            return None
        return Upload(
            id=upload_id,
            name=files.document_name(upload_id),
            bytes=known[0],
            media_type="application/pdf",
            path=found[0],
            sha256=known[1],
        )

    def discard(self, upload_id):
        """Forget ``upload_id`` and remove its file; True if it was known."""
        if self._known.pop(upload_id, None) is None:
            return False
        try:
            os.unlink(self.directory / upload_id)
        except OSError:
            pass
        return True

    def sweep(self, max_age):
        """Remove every document here last written more than ``max_age``
        seconds ago."""
        cutoff = time.time() - max_age
        try:
            entries = list(os.scandir(self.directory))
        except OSError:
            return
        for entry in entries:
            try:
                if (
                    files.document_name(entry.name) is not None
                    and entry.is_file(follow_symlinks=False)
                    and entry.stat(follow_symlinks=False).st_mtime < cutoff
                ):
                    os.unlink(entry.path)
            except OSError:
                pass


def call_args(action, upload, human):
    """``build_args``'s arguments for ``action`` on ``upload``, checked:
    ``ValueError`` is the product refusing the click (its message is the
    human's), ``RuntimeError`` a product whose ``build_args`` returned
    something that is not a JSON object of other arguments."""
    args = action.build_args(upload, human)
    if not isinstance(args, Mapping) or not all(isinstance(key, str) for key in args):
        raise RuntimeError(
            "upload action %r: build_args returned %r, not an object" % (action.name, args)
        )
    clashing = sorted({action.content_arg, action.filename_arg} & set(args))
    if clashing:
        raise RuntimeError(
            "upload action %r: build_args set %s, which the server fills in"
            % (action.name, ", ".join(clashing))
        )
    return dict(args)


def card_input(action, upload, args):
    """What the card shows: the call's arguments as they will be sent, the
    file's name among them, and in place of its bytes, how many there are."""
    return {
        **args,
        action.filename_arg: upload.name,
        action.content_arg: "(the file's %s, read when you allow this)" % size_text(upload.bytes),
    }


def size_text(count):
    """``count`` bytes for a person: ``812 bytes``, ``48 KB``, ``1.2 MB``."""
    if count < 1024:
        return "%d bytes" % count
    if count < 1024 * 1024:
        return "%d KB" % round(count / 1024)
    return "%.1f MB" % (count / (1024 * 1024))


async def run(action, upload, human, args, *, broker, server, publish, queue_note, documents):
    """Ask the human, then make the call: the flow in the module docstring,
    from the card on, with ``args`` from ``call_args``. ``server`` is the
    ``RemoteServer`` ``action.tool`` names, ``broker`` the live session's
    ``PermissionBroker``, ``publish`` the app's event publisher,
    ``queue_note`` the bus's and ``documents`` the app's ``Documents``,
    which the upload is discarded from at the end. Never raises: however it
    ends, it ends with an ``UploadActionEnded``."""
    # Imported here: the tool layer and the MCP client come with the SDK,
    # which only an app with an agent (the only kind with actions) loads.
    from . import remote
    from .tools import namespaced

    remote_name, tool_name = action.tool
    tool = namespaced(remote_name, tool_name)

    def ended(outcome, text):
        documents.discard(upload.id)
        publish(
            UploadActionEnded(
                id="ua_%s" % secrets.token_hex(6),
                label=action.label,
                file=upload.name,
                bytes=upload.bytes,
                tool=tool,
                outcome=outcome,
                text=text,
                by=human.login,
                upload=upload.id,
            )
        )
        if outcome != ACTION_DENIED:
            queue_note(note(action, upload, human, outcome == ACTION_FAILED, text))

    decision = await broker.ask(
        tool, card_input(action, upload, args), None, action=action.label, by=human.login
    )
    if not decision.allow:
        ended(ACTION_DENIED, decision.message or "not approved")
        return
    loop = asyncio.get_running_loop()
    try:
        data = await loop.run_in_executor(None, files.read_document, upload.path, upload.bytes)
    except OSError as exc:
        ended(ACTION_FAILED, "%s could not be read, so nothing was sent: %s" % (upload.name, exc))
        return
    if len(data) != upload.bytes or hashlib.sha256(data).hexdigest() != upload.sha256:
        ended(
            ACTION_FAILED,
            "%s changed on disk after it was uploaded, so nothing was sent; upload it again"
            % upload.name,
        )
        return
    arguments = {
        **args,
        action.filename_arg: upload.name,
        action.content_arg: base64.b64encode(data).decode("ascii"),
    }
    try:
        result = await remote.call(server, tool_name, arguments)
    except Exception as exc:
        sys.stderr.write("error: upload action %s failed: %r\n" % (action.name, exc))
        ended(
            ACTION_FAILED,
            "%s failed inside %s itself (%s), which is a bug; it may or may not have reached "
            "the %s MCP server"
            % (action.label, product.current().name, type(exc).__name__, remote_name),
        )
        return
    text = "\n".join(
        item.get("text", "") for item in result.get("content", ()) if item.get("type") == "text"
    )
    ended(ACTION_FAILED if result.get("is_error") else ACTION_DONE, text)


def note(action, upload, human, failed, text):
    """What the agent is told, on its next turn, of an action that ran."""
    who = "The human (%s)" % human.label if human.label else "The human"
    remote_name, tool_name = action.tool
    if len(text) > NOTE_RESULT_LIMIT:
        text = text[:NOTE_RESULT_LIMIT] + "\n(cut at %d characters)" % NOTE_RESULT_LIMIT
    return '%s used "%s" on %s (%s), which sent it to %s on the %s MCP server. %s:\n%s' % (
        who,
        action.label,
        upload.name,
        size_text(upload.bytes),
        tool_name,
        remote_name,
        "That failed" if failed else "It answered",
        text or "(no text)",
    )
