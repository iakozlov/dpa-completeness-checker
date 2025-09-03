#!/bin/bash
# test-docker-setup.sh
# Test script to validate Docker setup

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

info() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

log "Starting Docker setup validation..."

# Test 1: Check if docker and docker compose are available
info "Test 1: Checking Docker availability"
if command -v docker &> /dev/null; then
    log "Docker is installed"
    docker --version
else
    error "Docker is not installed or not in PATH"
    exit 1
fi

if docker compose version &> /dev/null; then
    log "Docker Compose is available"
    docker compose version
else
    error "Docker Compose is not available"
    exit 1
fi

# Test 2: Check if services can be built
info "Test 2: Building Docker images"
if docker compose build --quiet; then
    log "Docker images built successfully"
else
    error "Failed to build Docker images"
    exit 1
fi

# Test 3: Check if services can start
info "Test 3: Starting services"
if docker compose up -d; then
    log "Services started successfully"
    sleep 5  # Give services time to start
else
    error "Failed to start services"
    exit 1
fi

# Test 4: Check if Ollama service is responding
info "Test 4: Checking Ollama service"
max_attempts=30
attempt=1

while [ $attempt -le $max_attempts ]; do
    if docker compose exec -T ollama curl -f -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        log "Ollama service is responding"
        break
    fi
    
    if [ $attempt -eq $max_attempts ]; then
        error "Ollama service failed to start after $max_attempts attempts"
        docker compose logs ollama
        exit 1
    fi
    
    info "Attempt $attempt/$max_attempts: Waiting for Ollama to start..."
    sleep 5
    ((attempt++))
done

# Test 5: Check if gdpr-checker container is running
info "Test 5: Checking gdpr-checker service"
if docker compose exec -T gdpr-checker python --version > /dev/null 2>&1; then
    log "gdpr-checker service is responding"
    docker compose exec -T gdpr-checker python --version
else
    error "gdpr-checker service is not responding"
    docker compose logs gdpr-checker
    exit 1
fi

# Test 6: Check if required files are mounted
info "Test 6: Checking file mounts"
required_files=(
    "/app/data/test_set.csv"
    "/app/data/requirements/requirements_deontic_manual.json"
    "/app/run_dpa_completeness_rcv.sh"
    "/app/run_dpa_completeness_pairwise.sh"
    "/app/run_dpa_completeness_direct.sh"
)

for file in "${required_files[@]}"; do
    if docker compose exec -T gdpr-checker test -f "$file"; then
        log "File exists: $file"
    else
        error "Missing file: $file"
        exit 1
    fi
done

# Test 7: Check if Python dependencies are installed
info "Test 7: Checking Python dependencies"
required_modules=("pandas" "requests" "json")
for module in "${required_modules[@]}"; do
    if docker compose exec -T gdpr-checker python -c "import $module" 2>/dev/null; then
        log "Python module available: $module"
    else
        error "Missing Python module: $module"
        exit 1
    fi
done

# Test 8: Check if deolingo is installed
info "Test 8: Checking deolingo installation"
if docker compose exec -T gdpr-checker which deolingo > /dev/null 2>&1; then
    log "deolingo is installed"
    docker compose exec -T gdpr-checker deolingo --version || log "deolingo version command not supported"
else
    warn "deolingo is not installed - this will be needed for solver steps"
fi

# Cleanup
info "Cleaning up test services"
docker compose down

log "Docker setup validation completed successfully!"
log "All tests passed. Your Docker environment is ready."
