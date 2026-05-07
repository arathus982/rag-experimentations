"""Confluence API wrapper for page traversal and content download."""

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Set, cast

from atlassian import Confluence
from rich.console import Console

from src.config.settings import ConfluenceSettings
from src.models.schemas import ConfluencePage

console = Console()


class ConfluenceClient:
    """Wraps the Confluence REST API for page traversal and content download."""

    def __init__(self, settings: ConfluenceSettings) -> None:
        self._settings = settings
        self._is_cloud = "atlassian.net" in settings.base_url
        self._thread_local = threading.local()
        # Populate for main thread immediately
        self._thread_local.confluence = self._make_confluence()

    def _make_confluence(self) -> Confluence:
        return Confluence(
            url=self._settings.base_url,
            username=self._settings.username,
            password=self._settings.api_token.get_secret_value(),
            cloud=self._is_cloud,
        )

    @property
    def _confluence(self) -> Confluence:
        """Per-thread Confluence instance — requests.Session is not thread-safe."""
        if not hasattr(self._thread_local, "confluence"):
            self._thread_local.confluence = self._make_confluence()
        return cast(Confluence, self._thread_local.confluence)

    def check_connection(self) -> None:
        """Verify connectivity and credentials by probing the configured space.

        Raises RuntimeError with a diagnostic message on failure.
        """
        space_key = self._settings.space_key
        try:
            space: Optional[Dict[str, Any]] = self._confluence.get_space(space_key)
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
        raw_pages: List[Dict[str, Any]] = self._confluence.get_all_pages_from_space(
            space=space_key,
            start=0,
            limit=500,
            expand="ancestors",
        )
        return self._build_page_tree(raw_pages, space_key)

    def _get_page_subtree(self, root_page_id: str) -> List[ConfluencePage]:
        """Collect page subtree using two-phase parallel BFS.

        Phase 1: Discover all page IDs via parallel child-fetching per BFS level.
        Phase 2: Fetch full metadata (with ancestors) for all discovered IDs in parallel.
        """
        space_key = self._settings.space_key
        max_workers = self._settings.max_workers

        # Phase 1: BFS to collect all descendant page IDs
        all_ids: Set[str] = {root_page_id}
        frontier = [root_page_id]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            while frontier:
                child_futures = {
                    executor.submit(self._fetch_children, pid): pid for pid in frontier
                }
                frontier = []
                for child_future in as_completed(child_futures):
                    pid = child_futures[child_future]
                    try:
                        children = child_future.result()
                    except Exception as exc:
                        console.print(
                            f"[yellow]Warning: failed to fetch children of {pid}: {exc}[/yellow]"
                        )
                        continue
                    for child in children:
                        cid = str(child["id"])
                        if cid not in all_ids:
                            all_ids.add(cid)
                            frontier.append(cid)

        console.print(f"Discovered [bold]{len(all_ids)}[/bold] pages")

        # Phase 2: Fetch full page metadata (with ancestors) for all IDs in parallel
        raw_pages: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            page_futures = {
                executor.submit(self._fetch_page_with_ancestors, pid): pid
                for pid in all_ids
            }
            for page_future in as_completed(page_futures):
                pid = page_futures[page_future]
                try:
                    raw_pages.append(page_future.result())
                except Exception as exc:
                    console.print(
                        f"[yellow]Warning: failed to fetch metadata for page {pid}: {exc}[/yellow]"
                    )

        return self._build_page_tree(raw_pages, space_key)

    def _fetch_children(self, page_id: str) -> List[Dict[str, Any]]:
        """Fetch child pages for a given page ID. Called from worker threads."""
        return self._confluence.get_child_pages(page_id)  # type: ignore[no-any-return]

    def _fetch_page_with_ancestors(self, page_id: str) -> Dict[str, Any]:
        """Fetch full page metadata including ancestors. Called from worker threads."""
        return self._confluence.get_page_by_id(  # type: ignore[no-any-return]
            page_id=page_id,
            expand="ancestors",
        )

    def get_page_content(self, page_id: str) -> str:
        """Return the HTML storage-format body of a page."""
        page = self._confluence.get_page_by_id(
            page_id=page_id,
            expand="body.storage",
        )
        return str(page["body"]["storage"]["value"])  # type: ignore[return-value]

    def get_page_attachments(self, page_id: str) -> List[Dict[str, str]]:
        """List all attachments on a page."""
        result = self._confluence.get_attachments_from_content(
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
        return self._confluence.request(  # type: ignore[no-any-return]
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
