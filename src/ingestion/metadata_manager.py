"""Manages the Confluence manifest (page ID -> local path mapping)."""

from pathlib import Path

from src.models.schemas import ConfluenceManifest


class MetadataManager:
    """Saves and loads the Confluence download manifest."""

    def __init__(self, data_dir: Path) -> None:
        self._metadata_dir = data_dir / "metadata"
        self._manifest_path = self._metadata_dir / "confluence_manifest.json"

    def save_manifest(self, manifest: ConfluenceManifest) -> None:
        """Serialize manifest to JSON file."""
        self._metadata_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_path.write_text(
            manifest.model_dump_json(indent=2),
            encoding="utf-8",
        )

    def load_manifest(self) -> ConfluenceManifest:
        """Load existing manifest from disk."""
        raw = self._manifest_path.read_text(encoding="utf-8")
        return ConfluenceManifest.model_validate_json(raw)

    def manifest_exists(self) -> bool:
        """Check if a manifest file exists."""
        return self._manifest_path.exists()

    def update_page_path(self, manifest: ConfluenceManifest, page_id: str, local_path: str) -> None:
        """Update a single page's local path in the manifest."""
        if page_id in manifest.pages:
            manifest.pages[page_id].local_path = local_path
