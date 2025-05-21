.PHONY: clean build dist upload test lint typecheck format release sync-docs

clean:
	rm -rf build/ dist/ *.egg-info/

build: clean
	python -m build

dist: build

upload:
	twine upload dist/*

test:
	pytest

lint:
	flake8 reward_kit

typecheck:
	mypy reward_kit

format:
	black reward_kit

# Sync docs to ~/home/docs with links under 'evaluators'
sync-docs:
	@echo "Syncing docs to ~/home/docs with links under 'evaluators'..."
	@mkdir -p ~/home/docs/evaluators
	@# Create a temp directory for processed files
	@rm -rf /tmp/reward-kit-docs-processed
	@mkdir -p /tmp/reward-kit-docs-processed
	@# Copy all docs files to temp directory
	@cp -r ./docs/* /tmp/reward-kit-docs-processed/
	@# Only update links in the main documentation home file
	@if [ -f /tmp/reward-kit-docs-processed/documentation_home.mdx ]; then \
		sed -i -E 's/\[([^]]+)\]\(([^)]+\.mdx?)\)/[\1](evaluators\/\2)/g' /tmp/reward-kit-docs-processed/documentation_home.mdx; \
		sed -i -E 's/\[([^]]+)\]\(\/([^)]+\.mdx?)\)/[\1](\/evaluators\/\2)/g' /tmp/reward-kit-docs-processed/documentation_home.mdx; \
		sed -i -E 's/\.md\)/\.mdx)/g' /tmp/reward-kit-docs-processed/documentation_home.mdx; \
	fi
	@# Copy processed files to destination
	@rsync -av --delete /tmp/reward-kit-docs-processed/ ~/home/docs/evaluators/
	@echo "Docs synced successfully to ~/home/docs/evaluators"

# This help target prints all available targets
help:
	@echo "Available targets:"
	@echo "  clean      - Remove build artifacts"
	@echo "  build      - Build source and wheel distributions"
	@echo "  dist       - Alias for build"
	@echo "  upload     - Upload to PyPI (make sure to bump version first)"
	@echo "  test       - Run tests"
	@echo "  lint       - Run flake8 linter"
	@echo "  typecheck  - Run mypy type checker"
	@echo "  format     - Run black code formatter"
	@echo "  sync-docs  - Sync docs to ~/home/docs with links under 'evaluators'"
	@echo "  release    - Run lint, typecheck, test, build, then upload"
	@echo ""
	@echo "Usage examples:"
	@echo "  make release   - Full release process"
	@echo "  make lint      - Only run linting"
	@echo "  make format    - Format the code"
	@echo "  make sync-docs - Sync documentation with path adjustments"

release: lint typecheck test build upload
	@echo "Published to PyPI"

# Demo for Remote Evaluation using Serveo.net
demo-remote-eval:
	@echo "---------------------------------------------------------------------"
	@echo "Running Remote Evaluation Demo with Serveo.net..."
	@echo "This demo will:"
	@echo "1. Generate a temporary API key."
	@echo "2. Start a local mock API service."
	@echo "3. Expose the mock API service to the internet using Serveo.net via SSH."
	@echo "   (Requires a working SSH client in your PATH)"
	@echo "4. Run evaluation functions that call the tunneled mock API service."
	@echo "5. Clean up all started processes on completion or interruption."
	@echo "---------------------------------------------------------------------"
	@echo "Log files for the demo will be created in ./logs/remote_eval_demo/"
	@echo "Starting demo script..."
	python examples/remote_eval_demo/run_demo.py
	@echo "---------------------------------------------------------------------"
	@echo "Remote Evaluation Demo finished."
	@echo "---------------------------------------------------------------------"
