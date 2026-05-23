.PHONY: install run test lint

install:
	pip install -r requirements.txt

run:
	uvicorn sentinelview.dashboard.app:app --reload --host 0.0.0.0 --port 8000

test:
	pytest tests/ -v

lint:
	ruff check sentinelview/ tests/
