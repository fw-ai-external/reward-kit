#!/usr/bin/env python3
"""
Sync documentation files to ~/home/docs with proper link transformations for Mintlify.

This script replaces the Makefile sync-docs target with a more maintainable Python implementation.
"""

import argparse
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Optional


class DocsSyncer:
    """Handles syncing docs with link transformations for Mintlify."""

    def __init__(self, docs_dir: Path, target_dir: Path):
        self.docs_dir = docs_dir
        self.target_dir = target_dir
        self.temp_dir: Optional[Path] = None

    def sync(self) -> bool:
        """Main sync operation."""
        try:
            print(f"Syncing docs from {self.docs_dir} to {self.target_dir}")

            # Create temporary directory for processing
            self.temp_dir = Path(tempfile.mkdtemp())
            temp_evaluators_dir = self.temp_dir / "evaluators"
            temp_evaluators_dir.mkdir(parents=True)

            # Copy all docs to temp directory
            print("Copying files...")
            shutil.copytree(self.docs_dir, temp_evaluators_dir, dirs_exist_ok=True)

            # Apply transformations
            print("Applying link transformations...")
            self._transform_links(temp_evaluators_dir)

            # Copy to final destination
            print(f"Copying to {self.target_dir}...")
            self.target_dir.mkdir(parents=True, exist_ok=True)

            # Use rsync-like behavior: delete existing and copy new
            evaluators_target = self.target_dir / "evaluators"
            if evaluators_target.exists():
                shutil.rmtree(evaluators_target)

            shutil.copytree(temp_evaluators_dir, evaluators_target)

            print("✅ Docs synced successfully!")
            return True

        except Exception as e:
            print(f"❌ Error syncing docs: {e}")
            return False

        finally:
            # Clean up temp directory
            if self.temp_dir and self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)

    def _transform_links(self, docs_dir: Path) -> None:
        """Apply all link transformations to MDX files."""
        for mdx_file in docs_dir.rglob("*.mdx"):
            self._transform_file(mdx_file)

    def _transform_file(self, file_path: Path) -> None:
        """Transform links in a single file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            original_content = content

            # Get the relative directory of this file from the docs root for context
            relative_file_path = file_path.relative_to(self.temp_dir / "evaluators")
            current_dir = (
                relative_file_path.parent
                if relative_file_path.parent != Path(".")
                else Path("")
            )

            # Apply transformations in order - IMPORTANT: More specific patterns first!

            # 1. Transform GitHub links first (most specific patterns)
            # Transform ../../../examples/path style Python files to point to GitHub
            content = re.sub(
                r"\[([^\]]+)\]\(\.\.\/\.\.\/\.\.\/examples\/([^)]+\.py)\)",
                r"[\1](https://github.com/fireworks-ai/reward-kit/blob/main/examples/\2)",
                content,
            )

            # Transform ../../examples/README.md style links to point to GitHub
            content = re.sub(
                r"\[([^\]]+)\]\(\.\.\/\.\.\/examples\/README\.md\)",
                r"[\1](https://github.com/fireworks-ai/reward-kit/blob/main/examples/README.md)",
                content,
            )

            # Transform ../../examples/path style links to point to GitHub
            content = re.sub(
                r"\[([^\]]+)\]\(\.\.\/\.\.\/examples\/([^)]+\.md)\)",
                r"[\1](https://github.com/fireworks-ai/reward-kit/blob/main/examples/\2)",
                content,
            )

            # Transform ../examples/path style links to point to GitHub
            content = re.sub(
                r"\[([^\]]+)\]\(\.\.\/examples\/([^)]+\.md)\)",
                r"[\1](https://github.com/fireworks-ai/reward-kit/blob/main/examples/\2)",
                content,
            )

            # 2. Transform relative links without leading slash or dot
            # (e.g., evaluation_workflows.mdx or developer_guide/getting_started.mdx)
            content = re.sub(
                r"\[([^\]]+)\]\(([^)#]+\.mdx?)\)",
                lambda m: self._transform_relative_link(
                    m.group(1), m.group(2), current_dir
                ),
                content,
            )

            # 3. Transform complex relative paths like ../../something (non-examples)
            content = re.sub(
                r"\[([^\]]+)\]\(\.\.\/\.\.\/([^)]+\.mdx?)\)",
                lambda m: self._transform_to_flat_mintlify_link(
                    m.group(1), m.group(2), current_dir
                ),
                content,
            )

            # 4. Transform ../ relative links (go up one level, then add evaluators prefix)
            content = re.sub(
                r"\[([^\]]+)\]\(\.\.\/([^)]+\.mdx?)\)",
                lambda m: self._transform_to_flat_mintlify_link(
                    m.group(1), m.group(2), current_dir
                ),
                content,
            )

            # 5. Transform ./ relative links (same directory)
            content = re.sub(
                r"\[([^\]]+)\]\(\.\/([^)]+\.mdx?)\)",
                lambda m: self._transform_to_flat_mintlify_link(
                    m.group(1), m.group(2), current_dir
                ),
                content,
            )

            # 9. Transform image references for Mintlify
            content = re.sub(
                r"!\[([^\]]*)\]\(images\/([^)]+)\)",
                r"![\1](/evaluators/developer_guide/images/\2)",
                content,
            )
            content = re.sub(
                r"!\[([^\]]*)\]\(main_screen\.png\)",
                r"![\1](/evaluators/main_screen.png)",
                content,
            )

            # 10. Remove .mdx extensions from internal links for Mintlify
            content = re.sub(r"(\/evaluators\/[^)]+)\.mdx\)", r"\1)", content)

            # 11. Remove .md extensions from internal links for Mintlify
            content = re.sub(r"(\/evaluators\/[^)]+)\.md\)", r"\1)", content)

            # Write back if changed
            if content != original_content:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)

        except Exception as e:
            print(f"Warning: Failed to transform {file_path}: {e}")

    def _transform_relative_link(
        self, link_text: str, link_url: str, current_dir: Path = None
    ) -> str:
        """Transform a relative link, but skip external links."""
        # Skip external links (http/https)
        if link_url.startswith(("http://", "https://")):
            return f"[{link_text}]({link_url})"

        # Skip anchor links
        if link_url.startswith("#"):
            return f"[{link_text}]({link_url})"

        # Skip links that start with / (already absolute)
        if link_url.startswith("/"):
            return f"[{link_text}]({link_url})"

        # Skip links that start with . (will be handled by other rules)
        if link_url.startswith("."):
            return f"[{link_text}]({link_url})"

        # Transform relative link - preserve directory structure for Mintlify
        # Remove extension but keep the path structure
        clean_url = link_url
        if clean_url.endswith((".mdx", ".md")):
            clean_url = clean_url[:-4] if clean_url.endswith(".mdx") else clean_url[:-3]

        # If it's a simple filename (no directories) and we have a current directory context,
        # include the current directory in the path
        if "/" not in clean_url and current_dir and current_dir != Path(""):
            full_path = current_dir / clean_url
            return f"[{link_text}](/evaluators/{full_path})"
        else:
            return f"[{link_text}](/evaluators/{clean_url})"

    def _transform_to_flat_mintlify_link(
        self, link_text: str, link_url: str, current_dir: Path = None
    ) -> str:
        """Transform a relative link preserving directory structure for Mintlify."""
        # Remove extension but keep the path structure
        clean_url = link_url
        if clean_url.endswith((".mdx", ".md")):
            clean_url = clean_url[:-4] if clean_url.endswith(".mdx") else clean_url[:-3]

        # If it's a simple filename (no directories) and we have a current directory context,
        # include the current directory in the path
        if "/" not in clean_url and current_dir and current_dir != Path(""):
            full_path = current_dir / clean_url
            return f"[{link_text}](/evaluators/{full_path})"
        else:
            return f"[{link_text}](/evaluators/{clean_url})"


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Sync docs to Mintlify format")
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=Path.cwd() / "docs",
        help="Source docs directory (default: ./docs)",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=Path.home() / "home" / "docs",
        help="Target directory (default: ~/home/docs)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )

    args = parser.parse_args()

    if not args.docs_dir.exists():
        print(f"❌ Docs directory not found: {args.docs_dir}")
        return 1

    if args.dry_run:
        print("🔍 DRY RUN - no changes will be made")
        print(f"Would sync from: {args.docs_dir}")
        print(f"Would sync to: {args.target_dir / 'evaluators'}")
        return 0

    syncer = DocsSyncer(args.docs_dir, args.target_dir)
    success = syncer.sync()

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
