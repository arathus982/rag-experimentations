"""Converts Confluence storage-format HTML into clean Markdown."""

import re
from pathlib import Path

from markdownify import markdownify


class HtmlToMarkdownConverter:
    """Converts Confluence HTML to Markdown with image reference rewriting."""

    def convert(self, html_content: str, image_dir_name: str = "images") -> str:
        """Convert HTML to Markdown, rewriting image refs to local paths.

        Args:
            html_content: Confluence storage-format HTML.
            image_dir_name: Name of the local images subdirectory.

        Returns:
            Clean Markdown string with local image references.
        """
        cleaned_html = self._clean_confluence_macros(html_content)
        markdown = markdownify(cleaned_html, heading_style="ATX", strip=["style"])
        markdown = self._rewrite_image_references(markdown, image_dir_name)
        markdown = self._clean_whitespace(markdown)
        return markdown

    def _clean_confluence_macros(self, html: str) -> str:
        """Strip Confluence-specific macros (ac:structured-macro, etc.)."""
        # Remove <ac:structured-macro> wrappers but keep inner content
        html = re.sub(
            r"<ac:structured-macro[^>]*>.*?</ac:structured-macro>",
            "",
            html,
            flags=re.DOTALL,
        )
        # Convert <ac:image> tags to standard <img> tags
        html = re.sub(
            r'<ac:image[^>]*>\s*<ri:attachment\s+ri:filename="([^"]+)"' r"\s*/>\s*</ac:image>",
            r'<img src="\1" alt="\1"/>',
            html,
            flags=re.DOTALL,
        )
        # Remove remaining ac: and ri: tags
        html = re.sub(r"</?(?:ac|ri):[^>]*>", "", html)
        return html

    def _rewrite_image_references(self, markdown: str, image_dir_name: str) -> str:
        """Replace Confluence image URLs with local relative paths."""

        # Rewrite markdown image references to point to local images dir
        # Match ![alt](url) patterns and rewrite the URL part
        def _replace_img(match: re.Match[str]) -> str:
            alt = match.group(1)
            src = match.group(2)
            # Extract just the filename from any URL or path
            filename = Path(src).name
            if filename:
                return f"![{alt}]({image_dir_name}/{filename})"
            return match.group(0)

        markdown = re.sub(
            r"!\[([^\]]*)\]\(([^)]+)\)",
            _replace_img,
            markdown,
        )
        return markdown

    def _clean_whitespace(self, markdown: str) -> str:
        """Remove excessive blank lines."""
        markdown = re.sub(r"\n{3,}", "\n\n", markdown)
        return markdown.strip() + "\n"
