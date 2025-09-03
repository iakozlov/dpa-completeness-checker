# GDPR Completeness Checker

A system for evaluating the completeness of Data Processing Agreements (DPAs) against GDPR requirements using Large Language Models and Symbolic Solver. This project implements three different approaches for automated GDPR compliance checking.

## Overview

This project provides automated analysis of Data Processing Agreements (DPAs) to determine their completeness against GDPR requirements. The system combines semantic understanding through Large Language Models with formal logical reasoning using Deontic Logic and symbolic solver (deolingo).

### Available Approaches

1. **Pre-Classification**: A three-stage pipeline using LLM classification, fact extraction, and logical verification
2. **Pairwise**: Direct comparison between DPA segments and individual requirements
3. **Direct**: End-to-end LLM-based completeness evaluation

## Quick Start

### Option 1: Docker Setup (Recommended)

The easiest way to run the system is using Docker:

```bash
# Start services
docker compose up -d

# Pull a language model (first time only)
docker compose exec gdpr-checker /app/docker-entrypoint.sh pull-model qwen2.5:7b

# Run RCV approach demo
docker compose exec gdpr-checker /app/docker-entrypoint.sh rcv qwen2.5:7b --max_segments 10

# Or use the Makefile for convenience
make up && make pull-model && make rcv-demo
```

### Option 2: Local Setup

#### Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai/) server running locally
- Deolingo solver (included in `deolingo/` directory)

#### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd gdpr-completeness-checker
   ```

2. **Create virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Deolingo solver**:
   ```bash
   pip install -e ./deolingo/
   ```

5. **Start Ollama server** (in separate terminal):
   ```bash
   ollama serve
   ```

6. **Pull a language model**:
   ```bash
   ollama pull qwen2.5:7b  # Recommended for testing
   # or
   ollama pull qwen2.5:32b  # Better performance, requires more RAM
   ```

## Running Experiments

### Pre-Classification Approach

```bash
# Basic run with default settings
./run_dpa_completeness_rcv.sh

# Custom run with specific parameters
./run_dpa_completeness_rcv.sh \
  --model qwen2.5:14b \
  --max_segments 50 \
  --target_dpa "Online 124" \
  --req_ids "5,6,7,8" \
  --output_dir results/my_rcv_experiment
```

### Pairwise Approach

Direct pairwise comparison between segments and requirements:

```bash
# Basic run
./run_dpa_completeness_pairwise.sh

# Custom run
./run_dpa_completeness_pairwise.sh \
  --model qwen2.5:7b \
  --max_segments 20 \
  --target_dpa "Online 132" \
  --req_ids "5,6,7,8,9,10,11,12" \
  --output_dir results/my_pairwise_experiment
```

### Direct Approach

End-to-end LLM evaluation:

```bash
# Basic run
./run_dpa_completeness_direct.sh

# Custom run  
./run_dpa_completeness_direct.sh \
  --model qwen2.5:32b \
  --max_segments 100 \
  --target_dpa "Online 54" \
  --req_ids "all" \
  --output_dir results/my_direct_experiment
```

### Command Line Parameters

All scripts support these common parameters:

- `--model`: Ollama model name (default: qwen2.5:7b)
- `--max_segments`: Maximum segments to process (default: 0 = all)
- `--target_dpa`: Target DPA name (default: "Online 124")  
- `--req_ids`: Comma-separated requirement IDs or "all" (default: "5,6,7,8,9,10,11,12")
- `--output_dir`: Output directory (default: results/[approach]_approach)
- `--verbose`: Enable verbose output

## Docker Usage

```bash
make up && make pull-model && make rcv-demo

# Full workflow
docker compose up -d
docker compose exec gdpr-checker /app/docker-entrypoint.sh pull-model qwen2.5:14b
docker compose exec gdpr-checker /app/docker-entrypoint.sh rcv qwen2.5:14b --max_segments 50

# Interactive development
make shell
make logs 
```

## Development

### Adding New Models

1. Add model configuration to `config/ollama_config.json`
2. Pull model: `ollama pull <model-name>`
3. Test with small dataset first

### Custom Requirements

1. Add requirements to `data/requirements/` directory
2. Update requirement IDs in scripts
3. Modify evaluation metrics as needed
```
