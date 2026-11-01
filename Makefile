.PHONY: help test lint run docker-build docker-run clean

PYTHON ?= python
PIP ?= pip
DOCKER_IMAGE ?= aegis-credit-engine:latest
PORT ?= 8000

help:
	@echo "========================================================================"
	@echo "AEGIS INSTITUTIONAL CREDIT RISK & FORENSICS ENGINE — CLI"
	@echo "========================================================================"
	@echo "  make test          Run deterministic unit & integration tests"
	@echo "  make lint          Run code style and syntax checks"
	@echo "  make run           Launch FastAPI microservice locally"
	@echo "  make docker-build  Build multi-stage production Docker image"
	@echo "  make docker-run    Run containerized microservice on port $(PORT)"
	@echo "  make clean         Remove build caches, pytest artifacts, and bytecode"
	@echo "========================================================================"

test:
	$(PYTHON) -m pytest tests/test_engine.py -v --tb=short

lint:
	$(PYTHON) -m py_compile core/*.py schemas/*.py services/*.py api/*.py tests/*.py
	@echo "[AEGIS] Syntax compilation checks passed successfully."

run:
	$(PYTHON) -m uvicorn api.main:app --host 0.0.0.0 --port $(PORT) --reload

docker-build:
	docker build -t $(DOCKER_IMAGE) -f Dockerfile .

docker-run:
	docker run -p $(PORT):8000 --rm --name aegis_engine $(DOCKER_IMAGE)

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "[AEGIS] Workspace cleaned."
