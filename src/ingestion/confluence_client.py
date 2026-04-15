"""Confluence API wrapper for page traversal and content download."""

from typing import Any, Dict, List, Optional

from atlassian import Confluence
from rich.console import Console

from src.config.settings import ConfluenceSettings
from src.models.schemas import ConfluencePage

console = Console()


class ConfluenceClient:
    """Wraps the Confluence REST API for page traversal and content download."""

    def __init__(self, settings: ConfluenceSettings) -> None:
        self._settings = settings
        is_cloud = "atlassian.net" in settings.base_url
        self._client = Confluence(
            url=settings.base_url,
            username=settings.username,
            password=settings.api_token.get_secret_value(),
            cloud=is_cloud,
        )

    def check_connection(self) -> None:
        """Verify connectivity and credentials by probing the configured space.

        Raises RuntimeError with a diagnostic message on failure.
        """
        space_key = self._settings.space_key
        try:
            space: Optional[Dict[str, Any]] = self._client.get_space(space_key)
            if not space:
                raise RuntimeError(f"Space '{space_key}' not found or not accessible.")
            space_name = space.get("name", space_key)
            console.print(f"[green]Connected.[/green] Space: [bold]{space_name}[/bold]")
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"Cannot connect to Confluence at '{self._settings.base_url}'.\n"
                "Ensure CONFLUENCE_BASE_URL is the root domain only, e.g.:\n"
                "  CONFLUENCE_BASE_URL=https://your-domain.atlassian.net\n"
                "Check CONFLUENCE_USERNAME and CONFLUENCE_API_TOKEN as well.\n"
                f"Original error: {exc}"
            ) from exc

    def get_pages(self) -> List[ConfluencePage]:
        """Fetch pages according to settings: subtree if root_page_id is set, else full space."""
        if self._settings.root_page_id:
            console.print(
                f"Fetching subtree under page [bold]{self._settings.root_page_id}[/bold]..."
            )
            return self._get_page_subtree(self._settings.root_page_id)

        space_key = self._settings.space_key
        console.print(f"Fetching all pages from space [bold]{space_key}[/bold]...")
        raw_pages: List[Dict[str, Any]] = self._client.get_all_pages_from_space(
            space=space_key,
            start=0,
            limit=500,
            expand="ancestors",
        )
        return self._build_page_tree(raw_pages, space_key)

    def _get_page_subtree(self, root_page_id: str) -> List[ConfluencePage]:
        """Recursively collect a page and all its descendants."""
        space_key = self._settings.space_key
        collected: List[Dict[str, Any]] = []
        self._collect_descendants(root_page_id, collected)
        return self._build_page_tree(collected, space_key)

    def _collect_descendants(
        self, page_id: str, collected: List[Dict[str, Any]]
    ) -> None:
        """DFS traversal: fetch page metadata then recurse into children."""
        page: Dict[str, Any] = self._client.get_page_by_id(
            page_id=page_id,
            expand="ancestors",
        )
        collected.append(page)

        children: List[Dict[str, Any]] = self._client.get_child_pages(page_id)
        for child in children:
            self._collect_descendants(str(child["id"]), collected)

    def get_page_content(self, page_id: str) -> str:
        """Return the HTML storage-format body of a page."""
        page = self._client.get_page_by_id(
            page_id=page_id,
            expand="body.storage",
        )
        return str(page["body"]["storage"]["value"])  # type: ignore[return-value]

    def get_page_attachments(self, page_id: str) -> List[Dict[str, str]]:
        """List all attachments on a page."""
        result = self._client.get_attachments_from_content(
            page_id=page_id,
            start=0,
            limit=100,
        )
        attachments = []
        for att in result.get("results", []):
            attachments.append(
                {
                    "filename": att["title"],
                    "download_url": att["_links"]["download"],
                    "media_type": att.get("metadata", {}).get("mediaType", "unknown"),
                }
            )
        return attachments

    def download_attachment(self, download_url: str) -> bytes:
        """Download a single attachment and return raw bytes."""
        return self._client.request(  # type: ignore[no-any-return]
            method="GET",
            path=download_url,
        ).content

    def _build_page_tree(
        self,
        raw_pages: List[Dict[str, Any]],
        space_key: str,
    ) -> List[ConfluencePage]:
        """Reconstruct parent-child tree from flat API results."""
        pages_by_id: Dict[str, ConfluencePage] = {}

        for raw in raw_pages:
            page_id = str(raw["id"])
            ancestors: List[Dict[str, Any]] = raw.get("ancestors", [])
            parent_id = str(ancestors[-1]["id"]) if ancestors else None

            links: Dict[str, Any] = raw.get("_links", {})
            page = ConfluencePage(
                page_id=page_id,
                title=str(raw["title"]),
                space_key=space_key,
                url=str(links.get("webui", "")),
                parent_id=parent_id,
            )
            pages_by_id[page_id] = page

        # Wire up children_ids
        for page in pages_by_id.values():
            if page.parent_id and page.parent_id in pages_by_id:
                pages_by_id[page.parent_id].children_ids.append(page.page_id)

        return list(pages_by_id.values())
