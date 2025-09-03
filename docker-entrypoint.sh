#!/bin/bash
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

# Function to wait for Ollama to be ready
wait_for_ollama() {
    log "Waiting for Ollama server to be ready..."
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f -s "${OLLAMA_HOST}/api/tags" > /dev/null 2>&1; then
            log "Ollama server is ready!"
            return 0
        fi
        
        log "Attempt $attempt/$max_attempts: Ollama not ready, waiting 5 seconds..."
        sleep 5
        ((attempt++))
    done
    
    error "Ollama server failed to start after $max_attempts attempts"
    return 1
}

# Function to pull a model if it doesn't exist
pull_model_if_needed() {
    local model=$1
    log "Checking if model $model is available..."
    
    if curl -f -s "${OLLAMA_HOST}/api/tags" | grep -q "\"name\":\"$model\"" 2>/dev/null; then
        log "Model $model is already available"
    else
        log "Pulling model $model..."
        curl -X POST "${OLLAMA_HOST}/api/pull" \
            -H "Content-Type: application/json" \
            -d "{\"name\":\"$model\"}"
        log "Model $model pulled successfully"
    fi
}

# Function to show usage
show_usage() {
    echo -e "${BLUE}GDPR Completeness Checker - Docker Container${NC}"
    echo ""
    echo "Available commands:"
    echo "  rcv <model> [options]     - Run RCV approach"
    echo "  pairwise <model> [options] - Run Pairwise approach"  
    echo "  direct <model> [options]   - Run Direct approach"
    echo "  pull-model <model>         - Pull an Ollama model"
    echo "  shell                      - Start interactive shell"
    echo ""
    echo "Examples:"
    echo "  docker-compose exec gdpr-checker /app/docker-entrypoint.sh rcv qwen2.5:7b --max_segments 10"
    echo "  docker-compose exec gdpr-checker /app/docker-entrypoint.sh pairwise qwen2.5:7b --max_segments 5"
    echo "  docker-compose exec gdpr-checker /app/docker-entrypoint.sh pull-model qwen2.5:7b"
    echo ""
}

# Main execution
case "${1:-help}" in
    "rcv")
        wait_for_ollama
        model=${2:-qwen2.5:7b}
        pull_model_if_needed "$model"
        shift 2
        log "Running RCV approach with model $model"
        export OLLAMA_HOST="${OLLAMA_HOST:-http://ollama:11434}"
        exec bash run_dpa_completeness_rcv.sh --model "$model" "$@"
        ;;
    
    "pairwise")
        wait_for_ollama
        model=${2:-qwen2.5:7b}
        pull_model_if_needed "$model"
        shift 2
        log "Running Pairwise approach with model $model"
        export OLLAMA_HOST="${OLLAMA_HOST:-http://ollama:11434}"
        exec bash run_dpa_completeness_pairwise.sh --model "$model" "$@"
        ;;
    
    "direct")
        wait_for_ollama
        model=${2:-qwen2.5:7b}
        pull_model_if_needed "$model"
        shift 2
        log "Running Direct approach with model $model"
        export OLLAMA_HOST="${OLLAMA_HOST:-http://ollama:11434}"
        exec bash run_dpa_completeness_direct.sh --model "$model" "$@"
        ;;
    
    "pull-model")
        wait_for_ollama
        model=${2:-qwen2.5:7b}
        pull_model_if_needed "$model"
        ;;
    
    "shell")
        wait_for_ollama
        log "Starting interactive shell..."
        exec /bin/bash
        ;;
    
    "help"|*)
        show_usage
        ;;
esac
