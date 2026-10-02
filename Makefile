.PHONY: help build test lint docs

# Default target
help:
	@echo "Available targets:"
	@echo "  build        - Build the package distribution"
	@echo "  test         - Run tests"
	@echo "  lint         - Lint code"
	@echo "  docs         - Build documentation"

# Build targets
build:
	uv run python -m build

# Testing targets
test:
	uv run pytest

# Linting and formatting targets
lint:
	uv run ruff check

# Documentation targets
docs:
	uv run mkdocs build
