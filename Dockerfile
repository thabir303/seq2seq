# Seq2Seq Text-to-Python Code Generation
# Multi-stage Docker build for efficient image size

FROM python:3.9-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK punkt data manually for faster and more reliable installation
RUN mkdir -p /root/nltk_data/tokenizers && \
    wget -q -O /root/nltk_data/tokenizers/punkt.zip https://github.com/nltk/nltk_data/raw/gh-pages/packages/tokenizers/punkt.zip && \
    unzip -q /root/nltk_data/tokenizers/punkt.zip -d /root/nltk_data/tokenizers/ && \
    rm /root/nltk_data/tokenizers/punkt.zip

# Set NLTK data path
ENV NLTK_DATA=/root/nltk_data

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p checkpoints results visualizations

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=""

# Default command - train all models, evaluate, and visualize
CMD ["sh", "-c", "\
    echo '=== Starting Seq2Seq Training Pipeline ===' && \
    python train.py --model all --epochs 15 --resume && \
    echo '\n=== Evaluating Models ===' && \
    python evaluate.py --model all && \
    echo '\n=== Generating Visualizations ===' && \
    python visualize_attention.py --num_examples 5 && \
    echo '\n=== Generating Report ===' && \
    python generate_report.py && \
    echo '\n✓ All tasks completed! Check results/ and visualizations/ folders'"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; print('OK')" || exit 1
