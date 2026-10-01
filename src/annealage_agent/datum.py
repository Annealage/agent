"""Annealage Datum as a remote MCP server a product's agent reaches.

Datum holds OCR'd component datasheets, application notes and eval-board
manuals, and the imported boards that use them. A product that wants its
agent to look parts up there declares it as a remote beside its own tools
(``remote.py``) and, if the human may file a datasheet from the chat pane, as
an upload action (``uploads.py``)::

    ToolServer(tools, ..., remote=(datum.remote(url),))
    create_app(..., upload_actions=(datum.upload_action("my-project"),))

``remote`` carries the grading every product shares, because it is a fact
about Datum's tools and not about any product:

* every lookup Datum marks read-only is read grade;
* ``submit_datasheet`` files a document into Datum, which outlasts the
  session, so it is write grade and reaches the human as a card;
* ``publish_reference``, ``relate_documents`` and ``unrelate_documents``
  change what Datum holds for everyone, so they are excluded: the agent never
  gets them, and a Datum that lists them is not warned about;
* Datum lists most of its tools only once ``getting_started`` has been
  called, so that is the ``prime`` call.

A tool Datum adds that this grading does not name is left out with the
remote's usual startup warning, until it is graded here.
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

GRADING = Grading(
    read=(
        "getting_started",
        "search_parts",
        "get_part_by_mpn",
        "list_parts_by_project_tag",
        "get_datasheet_content",
        "semantic_search",
        "get_datasheet_pages",
        "get_datasheet_outline",
        "get_document_relations",
        "get_job_status",
        "list_boards",
        "trace_board_rail_or_net",
        "get_board_bom",
        "find_boards_using_part",
        "diff_board_versions",
        "find_board_parts_by_function",
        "locate_board_component",
        "list_vault_categories",
        "search_vault_components",
        "get_vault_component",
        "semantic_search_vault_components",
        "check_board_bom_against_vault",
    ),
    view=(),
    write=(SUBMIT_TOOL,),
)

#: Datum's tools the agent does not get: they change what Datum holds for
#: everyone.
EXCLUDED = ("publish_reference", "relate_documents", "unrelate_documents")

#: Datum lists most of its tools only once ``getting_started`` has been called.
PRIME = ("getting_started", {})


def remote(url):
    """Datum at ``url`` as a ``RemoteServer``: ``GRADING``, ``PRIME`` and
    ``EXCLUDED`` under the name ``SERVER``."""
    return RemoteServer(SERVER, url, GRADING, prime=PRIME, excluded=EXCLUDED)


def upload_action(project_tag):
    """The chat pane's "Submit to Datum" button for a PDF the human attaches:
    ``submit_datasheet`` as a datasheet filed under ``project_tag``, the tag
    Datum lists the product's documents by."""
    return UploadAction(
        name="datum",
        label="Submit to Datum",
        tool=(SERVER, SUBMIT_TOOL),
        build_args=lambda upload, human: {
            "project_tag": project_tag,
            "doc_type_hint": "datasheet",
        },
    )
