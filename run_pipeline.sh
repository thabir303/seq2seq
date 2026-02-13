#!/bin/bash
# Quick Start Script for Seq2Seq Project
# Run with: bash run_pipeline.sh

set -e  # Exit on error

echo "================================"
echo "Seq2Seq Training Pipeline"
echo "================================"
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Docker is available
if command -v docker &> /dev/null && command -v docker-compose &> /dev/null; then
    echo -e "${GREEN}✓${NC} Docker detected"
    echo ""
    echo "Choose execution method:"
    echo "  1) Docker (Recommended - Runs everything automatically)"
    echo "  2) Local Python (Manual - Requires Python 3.9+)"
    echo ""
    read -p "Enter choice [1-2]: " choice
    
    if [ "$choice" = "1" ]; then
        echo ""
        echo -e "${YELLOW}Building Docker image...${NC}"
        docker-compose build
        
        echo ""
        echo -e "${YELLOW}Running complete pipeline...${NC}"
        docker-compose up
        
        echo ""
        echo -e "${GREEN}✅ Pipeline completed!${NC}"
        echo ""
        echo "📁 Check these folders:"
        echo "  - results/*.json (evaluation metrics)"
        echo "  - visualizations/*.png (plots and heatmaps)"
        echo "  - checkpoints/*.pt (model weights)"
        exit 0
    fi
else
    echo -e "${YELLOW}⚠${NC} Docker not found, using local Python"
    choice="2"
fi

if [ "$choice" = "2" ]; then
    # Check Python
    if ! command -v python &> /dev/null; then
        echo -e "${RED}❌ Python not found${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✓${NC} Using Python: $(python --version)"
    
    # Install dependencies
    echo ""
    echo -e "${YELLOW}Installing dependencies...${NC}"
    pip install -q -r requirements.txt
    python -c "import nltk; nltk.download('punkt', quiet=True)"
    
    # Clean old checkpoints
    echo ""
    echo -e "${YELLOW}Cleaning old checkpoints...${NC}"
    rm -f checkpoints/*.pt checkpoints/*.pkl
    
    # Train models
    echo ""
    echo -e "${YELLOW}Training all models (this may take 2-4 hours)...${NC}"
    python train.py --model all --epochs 15 --resume
    
    # Evaluate
    echo ""
    echo -e "${YELLOW}Evaluating models...${NC}"
    python evaluate.py --model all
    
    # Visualize
    echo ""
    echo -e "${YELLOW}Generating attention visualizations...${NC}"
    python visualize_attention.py --num_examples 5
    
    # Generate report data
    echo ""
    echo -e "${YELLOW}Generating report data...${NC}"
    python generate_report.py
    
    echo ""
    echo -e "${GREEN}✅ All tasks completed!${NC}"
    echo ""
    echo "📁 Results saved to:"
    echo "  - results/*.json"
    echo "  - visualizations/*.png"
    echo "  - checkpoints/*.pt"
    echo ""
    echo "📝 Next step: Edit REPORT_TEMPLATE.md and convert to PDF"
fi
