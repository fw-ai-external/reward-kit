.PHONY: clean build dist upload test lint typecheck format release

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
	@echo "  release    - Run lint, typecheck, test, build, then upload"
	@echo ""
	@echo "Usage examples:"
	@echo "  make release   - Full release process"
	@echo "  make lint      - Only run linting"
	@echo "  make format    - Format the code"

release: lint typecheck test build upload
	@echo "Published to PyPI"
