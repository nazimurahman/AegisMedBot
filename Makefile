.PHONY: help install dev test lint format docker-up docker-down deploy help

help:
	@echo "Available commands:"
	@echo "  install      Install dependencies"
	@echo "  dev          Run development server"
	@echo "  test         Run tests"
	@echo "  lint         Run linters"
	@echo "  format       Format code"
	@echo "  docker-up    Start Docker services"
	@echo "  docker-down  Stop Docker services"
	@echo "  deploy       Deploy to Kubernetes"

install:
	poetry install

dev:
	poetry run uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

test:
	poetry run pytest tests/ -v --cov=./ --cov-report=term-missing

lint:
	poetry run flake8 .
	poetry run mypy .

format:
	poetry run black .
	poetry run isort .

docker-up:
	docker-compose -f docker/docker-compose.yml up -d

docker-down:
	docker-compose -f docker/docker-compose.yml down

docker-logs:
	docker-compose -f docker/docker-compose.yml logs -f

deploy:
	kubectl apply -f kubernetes/namespaces/
	kubectl apply -f kubernetes/configmaps/
	kubectl apply -f kubernetes/secrets/
	kubectl apply -f kubernetes/deployments/
	kubectl apply -f kubernetes/services/
	kubectl apply -f kubernetes/ingress/
	kubectl apply -f kubernetes/hpa/

deploy-staging:
	kubectl apply -f kubernetes/overlays/staging/

deploy-production:
	kubectl apply -f kubernetes/overlays/production/

seed-database:
	poetry run python scripts/setup/load_sample_data.py

init-db:
	poetry run alembic upgrade head

create-migration:
	poetry run alembic revision --autogenerate -m "$(message)"

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "*.egg" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +