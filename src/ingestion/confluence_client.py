"""Confluence API wrapper for page traversal and content download."""

from typing import Any, Dict, List

from atlassian import Confluence
from rich.console import Console

from src.config.settings import ConfluenceSettings
from src.models.schemas import ConfluencePage

console = Console()


class ConfluenceClient:
    """Wraps the Confluence REST API for page traversal and content download."""

    def __init__(self, settings: ConfluenceSettings) -> None:
        self._settings = settings
        self._client = Confluence(
            url=settings.base_url,
            username=settings.username,
            password=settings.api_token.get_secret_value(),
        )

    def get_space_pages(self) -> List[ConfluencePage]:
        """Fetch all pages in the configured space, preserving hierarchy."""
        space_key = self._settings.space_key
        console.print(f"Fetching pages from space [bold]{space_key}[/bold]...")

        raw_pages = self._client.get_all_pages_from_space(
            space=space_key,
            start=0,
            limit=500,
            expand="ancestors",
        )
        return self._build_page_tree(raw_pages, space_key)

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
