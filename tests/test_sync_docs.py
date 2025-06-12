"""
Test module for the sync_docs.py script.
"""

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Add scripts to path for import
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from sync_docs import DocsSyncer


class TestDocsSyncerIntegration(unittest.TestCase):
    """Integration tests using real docs directory."""

    def setUp(self):
        """Set up test environment."""
        self.repo_root = Path(__file__).parent.parent
        self.docs_dir = self.repo_root / "docs"
        self.temp_target = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up test environment."""
        if self.temp_target.exists():
            shutil.rmtree(self.temp_target)

    def test_real_docs_sync(self):
        """Test syncing the actual docs directory."""
        if not self.docs_dir.exists():
            self.skipTest("Docs directory not found")

        syncer = DocsSyncer(self.docs_dir, self.temp_target)
        success = syncer.sync()

        self.assertTrue(success)
        self.assertTrue((self.temp_target / "evaluators").exists())

        # Check that some key files exist
        key_files = [
            "documentation_home.mdx",
            "api_reference/api_overview.mdx",
            "developer_guide/getting_started.mdx",
        ]

        for file_path in key_files:
            full_path = self.temp_target / "evaluators" / file_path
            self.assertTrue(full_path.exists(), f"Missing {file_path}")

    def test_no_untransformed_links(self):
        """Test that all internal links are properly transformed."""
        if not self.docs_dir.exists():
            self.skipTest("Docs directory not found")

        syncer = DocsSyncer(self.docs_dir, self.temp_target)
        success = syncer.sync()

        self.assertTrue(success)

        # Check for untransformed internal links
        untransformed_patterns = [
            r"\[([^\]]+)\]\(([^)]+\.mdx)\)",  # .mdx links not starting with /evaluators/
            r"\[([^\]]+)\]\(([^)]+\.md)\)",  # .md links not starting with /evaluators/
        ]

        evaluators_dir = self.temp_target / "evaluators"
        untransformed_links = []

        for mdx_file in evaluators_dir.rglob("*.mdx"):
            content = mdx_file.read_text(encoding="utf-8")

            for pattern in untransformed_patterns:
                import re

                matches = re.findall(pattern, content)
                for match in matches:
                    link_text, link_url = match
                    # Skip if already transformed or external
                    if (
                        link_url.startswith("/evaluators/")
                        or link_url.startswith("http")
                        or link_url.startswith("#")
                    ):
                        continue

                    untransformed_links.append(
                        {
                            "file": str(mdx_file.relative_to(evaluators_dir)),
                            "text": link_text,
                            "url": link_url,
                        }
                    )

        if untransformed_links:
            error_msg = "Found untransformed internal links:\n"
            for link in untransformed_links[:10]:  # Show first 10
                error_msg += f"  {link['file']}: [{link['text']}]({link['url']})\n"
            if len(untransformed_links) > 10:
                error_msg += f"  ... and {len(untransformed_links) - 10} more\n"
            self.fail(error_msg)


if __name__ == "__main__":
    unittest.main()
