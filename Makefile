# Makefile for crypto-price-prediction

# Default target
all: help

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  help        Show this help message (default target)"
	@echo "  install     Install dependencies and run pre-commit hooks"
	@echo "  run   			 Runs e2e pipeline locally"
	@echo "  lint        Run linters"
	@echo "  test        Run tests"
	@echo "  test-cov    Run tests with coverage and open reports in browser"
	@echo "  clean-test  Clean test artifacts"

install:
	@uv sync
	@uv run pre-commit install

run:
	@echo "Running pipeline locally..."
	@uv run python -m app.entrypoints.command_line.run_local

lint:
	@echo "Running linters and formatters..."
	@echo "Running black..."
	@uv run black app/ tests/
	@echo "Running isort..."
	@uv run isort app/ tests/
	@echo "Running flake8..."
	@uv run flake8 app/ tests/
	@echo "Running mypy..."
	@uv run mypy app/ tests/
	@echo "All lint checks passed!"

test:
	@echo "Running tests..."
	@uv run pytest --no-cov --quiet

test-cov:
	@echo "Running tests with coverage and opening HTML report..."
	@uv run pytest \
		--cov=app \
		--quiet \
		--strict-markers \
		--strict-config \
		--cov-report=term-missing \
		--cov-report=html:reports/htmlcov \
		--cov-report=xml:reports/coverage.xml \
		--cov-fail-under=85 \
		--html=reports/pytest_report.html \
		--self-contained-html
	@echo "Coverage report generated at reports/htmlcov/index.html"
	@open reports/htmlcov/index.html 2>/dev/null || echo "Open reports/htmlcov/index.html in your browser to view the coverage report"
	@open reports/pytest_report.html 2>/dev/null || echo "Open reports/pytest_report.html in your browser to view the test report"

clean-test:
	@echo "Cleaning test artifacts..."
	@rm -rf .pytest_cache
	@rm -rf reports
	@rm -f .coverage

.PHONY: all install run help lint test test-cov clean-test
