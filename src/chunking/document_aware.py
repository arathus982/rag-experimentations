"""Document-aware chunking for short Hungarian Confluence documents."""

import re
from pathlib import Path
from typing import List, Tuple

from llama_index.core.schema import BaseNode, Document, TextNode

from src.config.constants import DOCUMENT_AWARE_SMALL_DOC_LINE_THRESHOLD

_HEADING_PREFIXES: Tuple[str, ...] = ("## ", "### ")
_CONTEXT_TEMPLATE: str = "[Kontextus: {path}]\n[Fejezet: {section}]\n\n"

# Matches internal Markdown links: [text](path) where path is not http(s)
_INTERNAL_LINK_PATTERN: re.Pattern[str] = re.compile(r"\[(?:[^\]]*)\]\(([^)]+)\)")


class DocumentAwareChunker:
    """Structure-aware chunker for Confluence Markdown documents.

    Rules applied per document:
    1. Small docs (< DOCUMENT_AWARE_SMALL_DOC_LINE_THRESHOLD lines) → one chunk.
    2. Larger docs → split on H2/H3 headings (code fences are respected so
       heading-like comment lines inside ``` blocks are not treated as splits).
    3. Every chunk is prefixed with folder path + section title before embedding.
    4. Cross-document Markdown links are extracted and stored as metadata.
    """

    def chunk(self, documents: List[Document]) -> List[BaseNode]:
        """Chunk documents into context-enriched TextNodes."""
        nodes: List[BaseNode] = []
        for doc in documents:
            nodes.extend(self._process_document(doc))
        return nodes

    def _process_document(self, doc: Document) -> List[TextNode]:
        text = doc.get_content()
        folder_path = self._extract_folder_path(doc)
        line_count = text.count("\n") + 1

        if line_count < DOCUMENT_AWARE_SMALL_DOC_LINE_THRESHOLD:
            return [self._make_node(text, folder_path, doc, is_whole=True)]

        sections = self._split_by_headings(text)
        return [
            self._make_node(section_text, folder_path, doc, heading=heading)
            for heading, section_text in sections
            if section_text.strip()
        ]

    def _split_by_headings(self, text: str) -> List[Tuple[str, str]]:
        """Split text at H2/H3 boundaries, skipping fenced code blocks."""
        lines = text.splitlines(keepends=True)
        sections: List[Tuple[str, str]] = []
        current_heading: str = ""
        current_lines: List[str] = []
        in_code_block: bool = False

        for line in lines:
            stripped = line.rstrip()

            if stripped.startswith("```"):
                in_code_block = not in_code_block

            is_split_point = not in_code_block and any(
                stripped.startswith(prefix) for prefix in _HEADING_PREFIXES
            )

            if is_split_point:
                if current_lines:
                    sections.append((current_heading, "".join(current_lines)))
                current_heading = stripped.lstrip("#").strip()
                current_lines = [line]
            else:
                current_lines.append(line)

        if current_lines:
            sections.append((current_heading, "".join(current_lines)))

        return sections

    def _make_node(
        self,
        text: str,
        folder_path: str,
        doc: Document,
        heading: str = "",
        is_whole: bool = False,
    ) -> TextNode:
        section_label = heading or doc.metadata.get("file_name", "document")
        enriched_text = _CONTEXT_TEMPLATE.format(
            path=folder_path or "unknown",
            section=section_label,
        ) + text
        outbound_links = self._extract_links(text)

        return TextNode(
            text=enriched_text,
            metadata={
                **doc.metadata,
                "folder_path": folder_path,
                "section_title": section_label,
                "outbound_links": outbound_links,
                "is_whole_document": is_whole,
            },
        )

    @staticmethod
    def _extract_folder_path(doc: Document) -> str:
        """Extract relative folder path below 'confluence/' from doc metadata."""
        file_path = doc.metadata.get("file_path", "")
        if not file_path:
            return ""
        parts = Path(file_path).parts
        try:
            conf_idx = parts.index("confluence")
            folder_parts = parts[conf_idx + 1 : -1]
            return "/".join(folder_parts)
        except ValueError:
            return ""

    @staticmethod
    def _extract_links(text: str) -> List[str]:
        """Extract internal Markdown link targets (non-HTTP paths only)."""
        return [
            link
            for link in _INTERNAL_LINK_PATTERN.findall(text)
            if not link.startswith("http")
        ]
