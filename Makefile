.PHONY: help dev test build-index clean docker-build docker-up docker-down lint format install

# Default target
help:
	@echo "Design Analysis Multi-Agent System - Makefile"
	@echo "=============================================="
	@echo ""
	@echo "Development:"
	@echo "  make dev              - Run app in development mode"
	@echo "  make install          - Install dependencies"
	@echo ""
	@echo "Testing & Quality:"
	@echo "  make test             - Run pytest suite"
	@echo "  make test-cov         - Run tests with coverage report"
	@echo "  make lint             - Run ruff linter"
	@echo "  make format           - Format code with ruff & isort"
	@echo "  make pre-commit       - Run pre-commit hooks"
	@echo ""
	@echo "Build & Index:"
	@echo "  make build-index      - Build RAG index from data"
	@echo "  make rebuild-index    - Rebuild RAG index (force)"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build     - Build Docker images"
	@echo "  make docker-up        - Start Docker containers"
	@echo "  make docker-down      - Stop Docker containers"
	@echo "  make docker-logs      - View Docker logs"
	@echo "  make docker-prod      - Run production container"
	@echo ""
	@echo "Database & Storage:"
	@echo "  make clean            - Clean cache and build files"
	@echo "  make clean-storage    - Clear storage/data (DANGEROUS)"
	@echo ""
	@echo "Utilities:"
	@echo "  make requirements     - Generate requirements.txt"
	@echo "  make health-check     - Verify system health"
	@echo ""

# Development
dev:
	@echo "Starting development server..."
	python -m streamlit run app.py --logger.level=debug

install:
	@echo "Installing dependencies..."
	pip install --upgrade pip setuptools wheel
	pip install -r requirements.txt
	@echo "✓ Dependencies installed"

# Testing
test:
	@echo "Running pytest suite..."
	pytest tests/ -v --tb=short

test-cov:
	@echo "Running tests with coverage..."
	pytest tests/ --cov=components --cov-report=html --cov-report=term-missing
	@echo "✓ Coverage report generated in htmlcov/index.html"

# Code quality
lint:
	@echo "Running ruff linter..."
	ruff check components/ tests/
	@echo "✓ Linting complete"

format:
	@echo "Formatting code..."
	ruff format components/ tests/
	@echo "✓ Code formatted"

pre-commit:
	@echo "Running pre-commit hooks..."
	pre-commit run --all-files
	@echo "✓ Pre-commit checks complete"

# Build & Index
build-index:
	@echo "Building RAG index from data..."
	@if [ -f "build_index.py" ]; then \
		python build_index.py; \
	else \
		echo "Note: build_index.py not found. Using default data..."; \
		python -c "from components.rag_system import RAGService; rag = RAGService(); print('✓ RAG index loaded')"; \
	fi

rebuild-index:
	@echo "Rebuilding RAG index (forcing refresh)..."
	@rm -rf data/faiss_index* 2>/dev/null || true
	@rm -rf .cache 2>/dev/null || true
	@$(MAKE) build-index

# Docker
docker-build:
	@echo "Building Docker images..."
	docker-compose build
	@echo "✓ Docker images built"

docker-up:
	@echo "Starting Docker containers (development)..."
	docker-compose up -d
	@echo "✓ Containers started"
	@echo "App available at: http://localhost:8501"

docker-prod:
	@echo "Starting Docker containers (production)..."
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
	@echo "✓ Production containers started"
	@echo "App available at: http://localhost:8501"

docker-down:
	@echo "Stopping Docker containers..."
	docker-compose down
	@echo "✓ Containers stopped"

docker-logs:
	@docker-compose logs -f app

docker-clean:
	@echo "Removing Docker containers and volumes..."
	docker-compose down -v
	@echo "✓ Docker cleaned"

# Database & Storage
clean:
	@echo "Cleaning cache and build files..."
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf build/ dist/ *.egg-info/ .coverage htmlcov/ 2>/dev/null || true
	@echo "✓ Cache cleaned"

clean-storage:
	@echo "WARNING: This will delete all stored reports and cache!"
	@read -p "Are you sure? (yes/no) " confirm; \
	if [ "$$confirm" = "yes" ]; then \
		rm -rf data/reports/* data/faiss_index* .cache; \
		echo "✓ Storage cleared"; \
	else \
		echo "Cancelled."; \
	fi

# Utilities
requirements:
	@echo "Exporting requirements.txt..."
	pip freeze > requirements.txt
	@echo "✓ Requirements exported"

health-check:
	@echo "Checking system health..."
	@echo "✓ Python version: $$(python --version)"
	@echo "✓ Streamlit: $$(python -c 'import streamlit; print(streamlit.__version__)')"
	@echo "✓ PyTorch: $$(python -c 'import torch; print(torch.__version__)')"
	@python -c "from components.config import Config; print('✓ Config: Valid' if Config.validate()[0] else '✗ Config: Invalid')"
	@echo "✓ All systems ready!"

# Run specific tests
test-image:
	@echo "Testing image processing..."
	pytest tests/test_image_processing_comprehensive.py -v

test-rag:
	@echo "Testing RAG retrieval..."
	pytest tests/test_rag_retrieval_shape.py -v

test-scoring:
	@echo "Testing scoring service..."
	pytest tests/test_scoring_determinism.py -v

test-agents:
	@echo "Testing agents..."
	pytest tests/test_agent_contract.py -v

test-orchestration:
	@echo "Testing orchestration..."
	pytest tests/test_orchestration_resilience.py -v

# Development shortcuts
dev-quick:
	@$(MAKE) clean
	@$(MAKE) format
	@$(MAKE) dev

dev-test:
	@$(MAKE) test-cov
	@echo ""
	@echo "Opening coverage report..."
	@python -m webbrowser htmlcov/index.html

# Docker shortcuts
docker-dev:
	@$(MAKE) docker-build
	@$(MAKE) docker-up

docker-restart:
	@$(MAKE) docker-down
	@$(MAKE) docker-up

# Help for specific topics
help-docker:
	@echo "Docker Commands"
	@echo "==============="
	@echo "make docker-build     - Build images"
	@echo "make docker-up        - Start containers"
	@echo "make docker-down      - Stop containers"
	@echo "make docker-logs      - View logs"
	@echo "make docker-clean     - Remove everything"
	@echo "make docker-prod      - Production mode"

help-test:
	@echo "Testing Commands"
	@echo "================"
	@echo "make test             - Run all tests"
	@echo "make test-cov         - Run with coverage"
	@echo "make test-image       - Test image processing"
	@echo "make test-rag         - Test RAG retrieval"
	@echo "make test-scoring     - Test scoring"
	@echo "make test-agents      - Test agents"
	@echo "make test-orchestration - Test orchestration"

help-quality:
	@echo "Code Quality Commands"
	@echo "====================="
	@echo "make lint             - Run linter"
	@echo "make format           - Format code"
	@echo "make pre-commit       - Run pre-commit hooks"
	@echo "make health-check     - Verify system"
