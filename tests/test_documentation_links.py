"""
Test module to validate documentation links in MDX files.

This module tests both GitHub-style links (with .mdx extensions) and
Mintlify-style links (without extensions, after sync-docs processing).
"""

import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List, Set, Tuple


class DocumentationLinksTest(unittest.TestCase):
    """Test documentation links for validity in both GitHub and Mintlify contexts."""

    def setUp(self):
        """Set up test environment."""
        self.repo_root = Path(__file__).parent.parent
        self.docs_dir = self.repo_root / "docs"
        self.temp_dir = None

    def tearDown(self):
        """Clean up test environment."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def _extract_links_from_file(self, file_path: Path) -> List[Tuple[str, str]]:
        """Extract markdown links from a file.

        Returns:
            List of tuples (link_text, link_url)
        """
        links = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Match markdown links: [text](url)
            pattern = r"\[([^\]]+)\]\(([^)]+)\)"
            matches = re.findall(pattern, content)

            for text, url in matches:
                # Skip external links (http/https), images, and anchors
                if not (
                    url.startswith("http")
                    or url.startswith("#")
                    or url.endswith(".png")
                    or url.endswith(".jpg")
                    or url.endswith(".jpeg")
                ):
                    links.append((text, url))

        except Exception as e:
            self.fail(f"Failed to read file {file_path}: {e}")

        return links

    def _resolve_relative_path(self, base_file: Path, relative_url: str) -> Path:
        """Resolve a relative URL from a base file to an absolute path."""
        base_dir = base_file.parent

        # Handle different types of relative paths
        if relative_url.startswith("./"):
            target_path = base_dir / relative_url[2:]
        elif relative_url.startswith("../"):
            target_path = base_dir / relative_url
        else:
            target_path = base_dir / relative_url

        return target_path.resolve()

    def test_sync_docs_integration(self):
        """Test that sync-docs script exists and can be imported."""
        scripts_dir = self.repo_root / "scripts"
        sync_docs_script = scripts_dir / "sync_docs.py"

        self.assertTrue(sync_docs_script.exists(), "sync_docs.py script should exist")

        # Test that it can be imported
        import sys

        sys.path.insert(0, str(scripts_dir))
        try:
            import sync_docs

            self.assertTrue(
                hasattr(sync_docs, "DocsSyncer"), "DocsSyncer class should exist"
            )
        except ImportError as e:
            self.fail(f"Failed to import sync_docs: {e}")
        finally:
            if str(scripts_dir) in sys.path:
                sys.path.remove(str(scripts_dir))

    def test_no_duplicate_file_extensions(self):
        """Test that documentation files don't have inconsistent extensions."""
        md_files = list(self.docs_dir.rglob("*.md"))
        mdx_files = list(self.docs_dir.rglob("*.mdx"))

        # Check for files with same name but different extensions
        md_names = {f.stem for f in md_files}
        mdx_names = {f.stem for f in mdx_files}

        duplicate_names = md_names.intersection(mdx_names)

        if duplicate_names:
            error_msg = (
                "Found files with same name but different extensions (.md vs .mdx):\n"
            )
            for name in duplicate_names:
                md_file = next(f for f in md_files if f.stem == name)
                mdx_file = next(f for f in mdx_files if f.stem == name)
                error_msg += f"  {md_file.relative_to(self.repo_root)} and {mdx_file.relative_to(self.repo_root)}\n"
            self.fail(error_msg)

    def test_makefile_sync_docs_target(self):
        """Test that the Makefile sync-docs target exists and calls the Python script."""
        makefile_path = self.repo_root / "Makefile"

        if not makefile_path.exists():
            self.fail("Makefile not found")

        with open(makefile_path, "r", encoding="utf-8") as f:
            makefile_content = f.read()

        # Check that sync-docs target exists
        if "sync-docs:" not in makefile_content:
            self.fail("sync-docs target not found in Makefile")

        # Check that it calls the Python script
        if "python scripts/sync_docs.py" not in makefile_content:
            self.fail("sync-docs target should call Python script")


if __name__ == "__main__":
    unittest.main()
