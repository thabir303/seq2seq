# Text-to-Python Code Generation using Seq2Seq Models

This project implements three different Sequence-to-Sequence (Seq2Seq) neural network architectures for generating Python code from natural language docstrings.

## 🎯 Objective

Explore how different recurrent neural network architectures perform on a **text-to-code generation task**, where natural language function descriptions (docstrings) are translated into Python source code.

## 📋 Models Implemented

### 1. Vanilla RNN Seq2Seq (Baseline)
- **Encoder**: Simple RNN
- **Decoder**: Simple RNN
- Fixed-length context vector
- No attention mechanism
- **Goal**: Establish baseline performance

### 2. LSTM Seq2Seq
- **Encoder**: LSTM
- **Decoder**: LSTM
- Fixed-length context vector
- **Goal**: Improve long-range dependency modeling

### 3. LSTM with Attention
- **Encoder**: Bidirectional LSTM
- **Decoder**: LSTM with Bahdanau (Additive) Attention
- Dynamic context vector computed at each step
- **Goal**: Remove fixed-context bottleneck, enable interpretability

## 📁 Project Structure

```
seq2seq/
├── config.py                   # Configuration and hyperparameters
├── train.py                    # Training script
├── evaluate.py                 # Evaluation script
├── visualize_attention.py      # Attention visualization
├── requirements.txt            # Dependencies
├── data/
│   ├── __init__.py
│   ├── dataset.py              # Dataset loading and preprocessing
│   └── vocabulary.py           # Vocabulary building
├── models/
│   ├── __init__.py
│   ├── encoder.py              # Encoder implementations
│   ├── decoder.py              # Decoder implementations
│   ├── attention.py            # Bahdanau attention
│   ├── vanilla_rnn.py          # Model 1: Vanilla RNN Seq2Seq
│   ├── lstm_seq2seq.py         # Model 2: LSTM Seq2Seq
│   └── lstm_attention.py       # Model 3: LSTM with Attention
├── utils/
│   ├── __init__.py
│   ├── metrics.py              # Evaluation metrics
│   ├── visualization.py        # Plotting functions
│   └── helpers.py              # Utility functions
├── checkpoints/                # Saved model checkpoints
├── results/                    # Evaluation results
└── visualizations/             # Generated plots and heatmaps
```

## 🔧 Installation

```bash
# Clone or navigate to the project
cd seq2seq

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (for BLEU score)
python -c "import nltk; nltk.download('punkt')"
```

## 📊 Dataset

The project uses the **CodeSearchNet Python** dataset from Hugging Face:
- **Source**: [Nan-Do/code-search-net-python](https://huggingface.co/datasets/Nan-Do/code-search-net-python)
- **Size**: ~250,000+ pairs (using 10,000 for training)
- **Content**: English docstrings paired with Python functions

### Data Configuration
| Parameter | Value |
|-----------|-------|
| Training examples | 10,000 |
| Validation examples | 1,000 |
| Test examples | 1,000 |
| Max docstring length | 50 tokens |
| Max code length | 80 tokens |

## 🚀 Usage

### Training

```bash
# Train all three models
python train.py --model all --epochs 20

# Train a specific model
python train.py --model vanilla_rnn --epochs 20
python train.py --model lstm --epochs 20
python train.py --model lstm_attention --epochs 20

# Custom configuration
python train.py --model all --epochs 30 --batch_size 32 --lr 0.001 --train_size 10000
```

### Evaluation

```bash
# Evaluate all models
python evaluate.py --model all

# Evaluate a specific model
python evaluate.py --model lstm_attention
```

### Attention Visualization

```bash
# Generate attention heatmaps for 5 examples
python visualize_attention.py --num_examples 5
```

## ⚙️ Training Configuration

| Hyperparameter | Value |
|---------------|-------|
| Embedding Dimension | 256 |
| Hidden Dimension | 256 |
| Number of Layers | 1 |
| Dropout | 0.3 |
| Batch Size | 64 |
| Learning Rate | 0.001 |
| Optimizer | Adam |
| Loss Function | Cross-Entropy |
| Teacher Forcing Ratio | 0.5 |
| Gradient Clipping | 1.0 |

## 📈 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Token-level Accuracy** | Percentage of correctly predicted tokens |
| **BLEU Score** | N-gram overlap between generated and reference code |
| **Exact Match Accuracy** | Percentage of completely correct outputs |
| **Syntax Accuracy** | Valid Python syntax (using AST) |

## 📊 Expected Results

After training, you should see output similar to:

| Model | BLEU Score | Token Accuracy | Exact Match |
|-------|------------|----------------|-------------|
| Vanilla RNN | ~0.15 | ~0.40 | ~0.02 |
| LSTM | ~0.20 | ~0.45 | ~0.03 |
| LSTM + Attention | ~0.25 | ~0.50 | ~0.05 |

*Note: Results may vary based on training time and random initialization.*

## 🔍 Attention Analysis

The attention visualization shows:
1. **Alignment heatmaps** between docstring tokens and generated code tokens
2. **Semantic focus**: Does "maximum" attend to `max()`?
3. **Keyword attention patterns**

Example questions to analyze:
- Does the word "maximum" attend strongly to `max()` function?
- Do parameter names in docstring align with function parameters?
- How does attention distribution change for longer sequences?

## 📝 Deliverables

- [x] Source code for all three models
- [x] Training script with checkpointing
- [x] Evaluation script with metrics
- [x] Attention visualization script
- [x] README with instructions
- [ ] **Trained model checkpoints** (generated after training)
- [ ] **Report (PDF)** (use REPORT_TEMPLATE.md and fill after training)

## 🔄 Running the Full Pipeline

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train all models (recommended: use Google Colab GPU)
#    See train_colab.ipynb for Colab setup
python train.py --model all --epochs 15

# 3. Evaluate all models
python evaluate.py --model all

# 4. Generate attention visualizations
python visualize_attention.py --num_examples 5

# 5. Generate report summary
python generate_report.py

# 6. Complete the PDF report
#    - Open REPORT_TEMPLATE.md
#    - Fill in [TODO] sections with results
#    - Convert to PDF using pandoc or online tools
```

## 🎓 Key Learning Outcomes

1. **Vanilla RNN Limitations**: Observe performance degradation for longer docstrings
2. **LSTM Improvements**: Better handling of long-range dependencies
3. **Attention Benefits**: 
   - Overcomes fixed-context bottleneck
   - Provides interpretability through attention weights
   - Better code generation quality

## 📚 References

- [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)
- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) (Bahdanau Attention)
- [CodeSearchNet Challenge](https://github.com/github/CodeSearchNet)

## 📄 License

This project is for educational purposes as part of a Machine Learning assignment.
