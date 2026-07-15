.PHONY: install test lint typecheck quality demo audit serve

install:
	uv sync --extra sequence

test:
	uv run pytest

lint:
	uv run ruff check .

typecheck:
	uv run mypy

quality: lint typecheck test

demo:
	uv run flowtwin demo --output data/processed/demo_trace_port

audit:
	uv run flowtwin audit data/processed/demo_trace_port/events.parquet

serve:
	uv run flowtwin serve
