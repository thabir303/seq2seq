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
TRAIN_SIZE = 10000  # Use 5,000-10,000 training examples
VAL_SIZE = 1000
TEST_SIZE = 1000

# ============================================
# Tokenization Configuration
# ============================================
MAX_DOCSTRING_LENGTH = 50  # Maximum docstring tokens
MAX_CODE_LENGTH = 80       # Maximum code tokens

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
EMBEDDING_DIM = 256        # Embedding dimension (128-256 as per assignment)
HIDDEN_DIM = 256           # Hidden dimension (256 as per assignment)
NUM_LAYERS = 1             # Number of RNN/LSTM layers
DROPOUT = 0.3              # Dropout rate

# ============================================
# Training Configuration
# ============================================
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 20
TEACHER_FORCING_RATIO = 0.5  # Probability of using teacher forcing
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
