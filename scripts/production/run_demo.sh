#!/bin/bash
# Quick demo script for privacy-preserving FL fraud detection

set -e

echo "=================================="
echo "Privacy-Preserving FL Fraud Detection Demo"
echo "=================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo -e "${YELLOW}Checking Python version...${NC}"
python3 --version

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate

# Install dependencies
echo -e "${YELLOW}Installing dependencies...${NC}"
pip install -q -r requirements.txt

# Create necessary directories
echo -e "${YELLOW}Creating directories...${NC}"
mkdir -p data/processed data/raw models logs mlruns

# Run quick training demo
echo ""
echo -e "${GREEN}Running training demo (10 rounds, 3 clients)...${NC}"
echo ""

python scripts/run_training.py \
    preset=privacy_medium \
    fl.n_rounds=10 \
    data.n_clients=3 \
    data.batch_size=16 \
    fl.local_epochs=2 \
    mlops.mlflow_enabled=false \
    logging.level=INFO

echo ""
echo -e "${GREEN}Demo completed!${NC}"
echo ""
echo "Next steps:"
echo "  - Check models/ for saved models"
echo "  - Check logs/ for training logs"
echo "  - Run 'python scripts/run_training.py' for full training"
echo ""
