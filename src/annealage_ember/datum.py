"""Annealage Datum as a remote MCP server a product's agent reaches.

Datum holds OCR'd component datasheets, application notes and eval-board
manuals, and the imported boards that use them. A product that wants its
agent to look parts up there declares it as a remote beside its own tools
(``remote.py``), and may expose ``upload_action`` for a human-attached PDF::

    ToolServer(tools, ..., remote=(datum.remote(url),))
    create_app(..., upload_actions=(datum.upload_action("my-project"),))

``remote`` carries the shared model-facing policy: Datum's annotations decide
which newly-added tools are read grade, explicit product choices decide which
cost-bearing tools can be exposed, and a local deny list blocks shared-state
writes. Product code uses ``remote.call`` with the same read-only guard.

``submit_datasheet`` and ``request_datasheet`` are explicit write-grade tools.
``submit_evidence_job`` costs Datum worker time, so it is product-only and must
be opted into by name. A result is a candidate, never verified: ``validated``
means structure and provenance only. ``cancel_evidence_job`` is denied because
Datum has one submit token and any caller can cancel any job.

Datum lists most of its tools only once ``getting_started`` has been called,
so that is the ``prime`` call.

What Datum returns is text the model reads, not instructions: a request's
``mpn``, ``requester``, ``log``, ``error`` and ``candidates`` come from
outside Datum. ``get_part_by_mpn`` rows carry ``match`` (``exact`` or
``family``; absent means exact), ``matched_mpn`` and ``requested_mpn``, and a
``family`` row is not the part that was asked for.

Tools the server does not annotate as read-only are withheld unless their
names are explicitly graded here.
"""

from .remote import RemoteServer
from .tools import Grading
from .uploads import UploadAction

#: Datum's name as an MCP server: the key a repository's ``.mcp.json`` gives
#: it, and so the namespace its tools have (``mcp__ds-wiki__search_parts``).
SERVER = "ds-wiki"

#: Where the hosted Datum is reached.
DEFAULT_URL = "https://ds.story-kettle.ts.net/mcp"

#: The tool that files a document into Datum.
SUBMIT_TOOL = "submit_datasheet"

#: The tool that asks Datum to find and ingest a datasheet by MPN.
REQUEST_TOOL = "request_datasheet"

#: The tool that queues a document-bound evidence job.
EVIDENCE_SUBMIT_TOOL = "submit_evidence_job"

#: Tools never exposed to the model or permitted in product code.
DENY = ("cancel_evidence_job", "publish_reference", "relate_documents", "unrelate_documents")

#: Cost-bearing tools require an explicit product opt-in for model use.
PRODUCT_ONLY = (EVIDENCE_SUBMIT_TOOL,)

#: Fallback read-name guard for direct product calls, which may not have run
#: model-facing discovery first.
READ_TOOLS = (
    "getting_started", "search_parts", "get_part_by_mpn", "list_parts_by_project_tag",
    "get_datasheet_content", "semantic_search", "get_datasheet_pages",
    "get_datasheet_outline", "get_document_relations", "get_job_status",
    "get_request_status", "get_evidence_job_status", "get_evidence_result",
    "get_evidence_usage", "list_boards", "trace_board_rail_or_net", "get_board_bom",
    "find_boards_using_part", "diff_board_versions", "find_board_parts_by_function",
    "locate_board_component", "list_vault_categories", "search_vault_components",
    "get_vault_component", "semantic_search_vault_components",
    "check_board_bom_against_vault",
)

#: Datum lists most of its tools only once ``getting_started`` has been called.
PRIME = ("getting_started", {})


def remote(url, *, view=()):
    """Datum at ``url`` with annotation grading and explicit product opt-ins."""
    return RemoteServer(
        SERVER,
        url,
        Grading((), tuple(view), (SUBMIT_TOOL, REQUEST_TOOL)),
        prime=PRIME,
        excluded=DENY,
        annotation_grading=True,
        product_only=PRODUCT_ONLY,
        read_tools=READ_TOOLS,
        denied=DENY,
    )


def upload_action(project_tag):
    """A PDF upload action that asks the human for its exact source URL."""
    return UploadAction(
        name="datum",
        label="Submit to Datum",
        tool=(SERVER, SUBMIT_TOOL),
        build_args=lambda upload, human: {
            "project_tag": project_tag,
            "doc_type_hint": "datasheet",
        },
        source_url_arg="original_url",
        source_url_label="Original HTTP(S) source URL (max 2048 characters)",
    )
