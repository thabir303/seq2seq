"""
Configuration file for Seq2Seq Text-to-Python Code Generation.
Contains all hyperparameters and settings for training and evaluation.
"""

import torch
import os

# ============================================
# Device Configuration
# ============================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================
# Paths
# ============================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
VISUALIZATION_DIR = os.path.join(BASE_DIR, 'visualizations')

# Create directories if they don't exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

# ============================================
# Dataset Configuration
# ============================================
DATASET_NAME = "Nan-Do/code-search-net-python"
TRAIN_SIZE = 20000   # More data = better generalization
VAL_SIZE = 2000
TEST_SIZE = 1000

# ============================================
# Tokenization Configuration
# ============================================
MAX_DOCSTRING_LENGTH = 60  # Slightly longer docstrings
MAX_CODE_LENGTH = 120      # More room for code output

# Special tokens
PAD_TOKEN = '<PAD>'
SOS_TOKEN = '<SOS>'
EOS_TOKEN = '<EOS>'
UNK_TOKEN = '<UNK>'

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3

# ============================================
# Model Configuration  
# ============================================
EMBEDDING_DIM = 256        # Standard size for good accuracy
HIDDEN_DIM = 512           # Larger hidden dim for better representation
NUM_LAYERS = 2             # 2 layers for deeper learning
DROPOUT = 0.3              # Dropout rate

# ============================================
# Training Configuration
# ============================================
BATCH_SIZE = 64             # Good for GPU memory
LEARNING_RATE = 0.001
NUM_EPOCHS = 25             # More epochs for better convergence
TEACHER_FORCING_RATIO = 0.75  # Higher TF ratio helps learn faster
CLIP_GRAD = 1.0             # Gradient clipping value

# ============================================
# Model Types
# ============================================
MODEL_VANILLA_RNN = 'vanilla_rnn'
MODEL_LSTM = 'lstm'
MODEL_LSTM_ATTENTION = 'lstm_attention'

ALL_MODELS = [MODEL_VANILLA_RNN, MODEL_LSTM, MODEL_LSTM_ATTENTION]

# ============================================
# Logging Configuration
# ============================================
LOG_INTERVAL = 100  # Log every N batches
SAVE_INTERVAL = 5   # Save checkpoint every N epochs
