# GDPR Completeness Checker - Docker Management

.PHONY: help build up down logs shell test pull-model clean

# Default target
help:
	@echo "GDPR Completeness Checker - Docker Commands"
	@echo "==========================================="
	@echo ""
	@echo "Setup Commands:"
	@echo "  build         Build the Docker images"
	@echo "  up            Start all services"
	@echo "  down          Stop all services"
	@echo "  clean         Stop services and remove volumes"
	@echo ""
	@echo "Management Commands:"
	@echo "  logs          Show logs from all services"
	@echo "  shell         Open interactive shell in container"
	@echo "  test          Validate Docker setup"
	@echo ""
	@echo "Model Commands:"
	@echo "  pull-model    Pull qwen2.5:7b model"
	@echo "  pull-large    Pull qwen2.5:14b model"
	@echo ""
	@echo "Analysis Commands:"
	@echo "  rcv-demo      Run RCV demo with 5 segments"
	@echo "  pairwise-demo Run Pairwise demo with 5 segments"
	@echo "  direct-demo   Run Direct demo with 5 segments"
	@echo ""
	@echo "Examples:"
	@echo "  make up && make pull-model && make rcv-demo"

# Build Docker images
build:
	docker compose build

# Start services
up:
	docker compose up -d
	@echo "Services started. Use 'make logs' to monitor startup."

# Stop services
down:
	docker compose down

# Show logs
logs:
	docker compose logs -f

# Open shell in container
shell:
	docker compose exec gdpr-checker /app/docker-entrypoint.sh shell

# Test Docker setup
test:
	@bash test-docker-setup.sh

# Pull default model
pull-model:
	docker compose exec gdpr-checker /app/docker-entrypoint.sh pull-model qwen2.5:7b

# Pull larger model
pull-large:
	docker compose exec gdpr-checker /app/docker-entrypoint.sh pull-model qwen2.5:32b

# Demo commands
rcv-demo:
	docker compose exec gdpr-checker /app/docker-entrypoint.sh rcv qwen2.5:7b --max_segments 5

pairwise-demo:
	docker compose exec gdpr-checker /app/docker-entrypoint.sh pairwise qwen2.5:7b --max_segments 5

direct-demo:
	docker compose exec gdpr-checker /app/docker-entrypoint.sh direct qwen2.5:7b --max_segments 5

# Clean up everything
clean:
	docker compose down -v
	docker system prune -f

# Development helpers
dev-build:
	docker compose build --no-cache

dev-logs:
	docker compose logs -f gdpr-checker

dev-restart:
	docker compose restart gdpr-checker

