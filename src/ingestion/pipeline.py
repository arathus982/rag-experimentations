"""Orchestrates the full Confluence -> local Markdown ingestion pipeline."""

import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from rich.console import Console
from tqdm import tqdm

from src.config.settings import AppSettings
from src.ingestion.confluence_client import ConfluenceClient
from src.ingestion.html_to_markdown import HtmlToMarkdownConverter
from src.ingestion.image_downloader import ImageDownloader
from src.ingestion.metadata_manager import MetadataManager
from src.models.schemas import ConfluenceManifest, ConfluencePage

console = Console()


def _sanitize_filename(name: str) -> str:
    """Sanitize a string for use as a directory/file name.

    Preserves Hungarian characters but removes filesystem-unsafe chars.
    """
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    name = name.strip(". ")
    return name or "untitled"


class IngestionPipeline:
    """Full Confluence -> local Markdown + images pipeline."""

    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._data_dir = Path(settings.data_dir)
        self._client = ConfluenceClient(settings.confluence)
        self._converter = HtmlToMarkdownConverter()
        self._image_downloader = ImageDownloader(self._client)
        self._metadata_manager = MetadataManager(self._data_dir)

    def run(self) -> ConfluenceManifest:
        """Execute the full ingestion pipeline.

        1. Fetch page tree from Confluence
        2. For each page (respecting hierarchy):
           a. Create local directory structure
           b. Download page HTML content
           c. Download images
           d. Convert HTML to Markdown with image refs
           e. Write .md file
           f. Update manifest
        3. Save manifest

        Returns:
            The completed ConfluenceManifest.
        """
        space_key = self._settings.confluence.space_key
        console.print(f"\n[bold green]Starting ingestion for space: {space_key}[/bold green]")

        # Step 1: Fetch page tree
        pages = self._client.get_space_pages()
        console.print(f"Found [bold]{len(pages)}[/bold] pages")

        # Build lookup and ancestry paths
        pages_by_id = {p.page_id: p for p in pages}
        base_dir = self._data_dir / "confluence" / space_key

        manifest = ConfluenceManifest(
            space_key=space_key,
            download_timestamp=datetime.utcnow(),
            pages={p.page_id: p for p in pages},
        )

        # Step 2: Process each page
        for page in tqdm(pages, desc="Downloading pages"):
            try:
                self._process_page(page, pages_by_id, base_dir, manifest)
            except Exception as e:
                console.print(
                    f"[red]Error processing '{page.title}' " f"(ID: {page.page_id}): {e}[/red]"
                )

        # Step 3: Save manifest
        self._metadata_manager.save_manifest(manifest)
        console.print(
            f"\n[bold green]Ingestion complete.[/bold green] "
            f"Processed {len(pages)} pages. "
            f"Manifest saved to {self._metadata_manager._manifest_path}"
        )
        return manifest

    def _process_page(
        self,
        page: ConfluencePage,
        pages_by_id: Dict[str, ConfluencePage],
        base_dir: Path,
        manifest: ConfluenceManifest,
    ) -> None:
        """Process a single page: download content, images, convert to MD."""
        # Build local path from ancestry
        page_dir = self._build_local_path(page, pages_by_id, base_dir)
        page_dir.mkdir(parents=True, exist_ok=True)

        # Download and convert HTML to Markdown
        html_content = self._client.get_page_content(page.page_id)
        markdown = self._converter.convert(html_content)

        # Download images
        images_dir = page_dir / "images"
        downloaded_images = self._image_downloader.download_page_images(page.page_id, images_dir)

        # Write Markdown file
        md_filename = _sanitize_filename(page.title) + ".md"
        md_path = page_dir / md_filename
        md_path.write_text(markdown, encoding="utf-8")

        # Update manifest
        page_in_manifest = manifest.pages[page.page_id]
        page_in_manifest.local_path = str(md_path)
        page_in_manifest.images = downloaded_images

    def _build_local_path(
        self,
        page: ConfluencePage,
        pages_by_id: Dict[str, ConfluencePage],
        base_dir: Path,
    ) -> Path:
        """Build local path from page ancestry.

        Example: data/confluence/SPACE/Parent_Page/Child_Page/
        """
        ancestry: List[str] = []
        current = page

        while current:
            ancestry.append(_sanitize_filename(current.title))
            if current.parent_id and current.parent_id in pages_by_id:
                current = pages_by_id[current.parent_id]
            else:
                break

        ancestry.reverse()
        path = base_dir
        for segment in ancestry:
            path = path / segment

        return path
