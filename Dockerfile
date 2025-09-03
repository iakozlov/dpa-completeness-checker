# Use Python 3.12 slim image as base
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Install deolingo from the local directory
RUN pip install -e ./deolingo/

# Create necessary directories
RUN mkdir -p results config data

# Set environment variables
ENV PYTHONPATH=/app
ENV OLLAMA_HOST=http://ollama:11434

# Create a non-root user
RUN useradd -m -u 1000 gdpr && chown -R gdpr:gdpr /app
USER gdpr

# Expose port for potential web interface (future use)
EXPOSE 8080

# Default command - can be overridden
CMD ["python", "--version"]
