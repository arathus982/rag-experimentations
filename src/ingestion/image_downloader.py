"""Downloads images from Confluence page attachments."""

from pathlib import Path
from typing import Dict, List

from rich.console import Console

from src.config.constants import SUPPORTED_IMAGE_EXTENSIONS
from src.ingestion.confluence_client import ConfluenceClient

console = Console()


class ImageDownloader:
    """Downloads image attachments from Confluence pages to local directories."""

    def __init__(self, client: ConfluenceClient) -> None:
        self._client = client

    def download_page_images(self, page_id: str, dest_dir: Path) -> List[str]:
        """Download all image attachments for a page.

        Args:
            page_id: Confluence page ID.
            dest_dir: Local directory to store images (e.g., page_dir/images/).

        Returns:
            List of downloaded filenames.
        """
        attachments = self._client.get_page_attachments(page_id)
        image_attachments = self._filter_images(attachments)

        if not image_attachments:
            return []

        dest_dir.mkdir(parents=True, exist_ok=True)
        downloaded: List[str] = []

        for att in image_attachments:
            filename = att["filename"]
            try:
                data = self._client.download_attachment(att["download_url"])
                file_path = dest_dir / filename
                file_path.write_bytes(data)
                downloaded.append(filename)
            except Exception as e:
                console.print(f"[yellow]Warning: Failed to download {filename}: {e}[/yellow]")

        return downloaded

    def _filter_images(self, attachments: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Filter attachments to only supported image types."""
        return [
            att
            for att in attachments
            if any(att["filename"].lower().endswith(ext) for ext in SUPPORTED_IMAGE_EXTENSIONS)
        ]
