
# Seq2Seq Text-to-Python Code Generation

Implements three Seq2Seq models for generating Python code from natural language:
1. Vanilla RNN (baseline)
2. LSTM 
3. LSTM with Attention

## Setup
## Clone Repository

```bash
git clone https://github.com/thabir303/seq2seq.git
cd seq2seq
```

### Install Dependencies
```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt')"
```

### Using Docker (Recommended)
```bash
docker-compose up
```

## Run

### Train Models
```bash
# Train all models
python train.py --model all --epochs 15 --resume

# Train specific model
python train.py --model lstm_attention --epochs 15
```

### Evaluate
```bash
python evaluate.py --model all
```

### Generate Visualizations
```bash
python visualize_attention.py --num_examples 5
```

### Generate Code (Interactive)
```bash
python generate_code.py
```

## Results

Results saved to:
- `checkpoints/` - Model weights
- `results/` - Evaluation metrics (JSON)
- `visualizations/` - Attention heatmaps (PNG)
