"""
Interactive Code Generation Script

Usage:
    python generate_code.py
    
Then enter your docstring and get Python code output.
"""

import os
import torch
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import DEVICE, PAD_IDX, SOS_IDX, EOS_IDX, MAX_CODE_LENGTH, CHECKPOINT_DIR
from data import Vocabulary
from models.lstm_attention import create_attention_model
from utils.helpers import load_checkpoint, tokens_to_string


def load_model_and_vocab():
    """Load the best model (LSTM with Attention) and vocabularies."""
    
    # Check if vocabularies exist
    src_vocab_path = os.path.join(CHECKPOINT_DIR, 'src_vocab.pkl')
    tgt_vocab_path = os.path.join(CHECKPOINT_DIR, 'tgt_vocab.pkl')
    checkpoint_path = os.path.join(CHECKPOINT_DIR, 'lstm_attention_best.pt')
    
    missing_files = []
    if not os.path.exists(src_vocab_path):
        missing_files.append('src_vocab.pkl')
    if not os.path.exists(tgt_vocab_path):
        missing_files.append('tgt_vocab.pkl')
    if not os.path.exists(checkpoint_path):
        missing_files.append('lstm_attention_best.pt')
    
    if missing_files:
        print(f"\n❌ Error: Required files not found in {CHECKPOINT_DIR}/")
        print(f"   Missing: {', '.join(missing_files)}")
        print(f"\n📝 You need to train the models first!")
        print(f"\n🚀 Run one of these commands:")
        print(f"   1. Train all models (6-12 hours on CPU):")
        print(f"      python train.py --model all --epochs 15 --resume")
        print(f"\n   2. Train only LSTM+Attention (2-4 hours on CPU):")
        print(f"      python train.py --model lstm_attention --epochs 15")
        print(f"\n   3. Quick test (2 epochs, 5 minutes):")
        print(f"      python train.py --model lstm_attention --epochs 2 --train_size 500")
        print(f"\n   4. Use Docker (automatic):")
        print(f"      docker-compose up")
        print(f"\n   5. Or train in Google Colab (GPU, faster):")
        print(f"      See seq2seq_assignment.ipynb")
        sys.exit(1)
    
    # Load vocabularies
    src_vocab = Vocabulary.load(src_vocab_path)
    tgt_vocab = Vocabulary.load(tgt_vocab_path)
    
    # Create model
    model = create_attention_model(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        device=DEVICE
    )
    
    # Load checkpoint
    load_checkpoint(model, checkpoint_path, device=DEVICE)
    model.eval()
    
    return model, src_vocab, tgt_vocab


def generate_code(model, src_vocab, tgt_vocab, docstring: str, max_length=MAX_CODE_LENGTH):
    """
    Generate Python code from a docstring.
    
    Args:
        model: Trained model
        src_vocab: Source vocabulary
        tgt_vocab: Target vocabulary
        docstring: Input natural language description
        max_length: Maximum output length
        
    Returns:
        Generated Python code as string
    """
    
    # Tokenize input
    tokens = docstring.lower().split()
    
    # Convert to indices
    src_indices = [src_vocab.word2idx.get(token, src_vocab.word2idx['<UNK>']) for token in tokens]
    src_tensor = torch.tensor([src_indices], dtype=torch.long).to(DEVICE)
    src_lengths = torch.tensor([len(src_indices)], dtype=torch.long).to(DEVICE)
    
    # Generate
    with torch.no_grad():
        # Encode
        encoder_outputs, hidden = model.encoder(src_tensor, src_lengths)
        
        # Decode
        decoder_input = torch.tensor([[SOS_IDX]], dtype=torch.long).to(DEVICE)
        generated_indices = []
        
        for _ in range(max_length):
            if hasattr(model.decoder, 'attention'):
                # LSTM with Attention
                output, hidden, _ = model.decoder(decoder_input, hidden, encoder_outputs)
            else:
                # LSTM or Vanilla RNN
                output, hidden = model.decoder(decoder_input, hidden)
            
            # Get prediction
            pred_idx = output.argmax(dim=-1).item()
            
            if pred_idx == EOS_IDX:
                break
            
            generated_indices.append(pred_idx)
            decoder_input = torch.tensor([[pred_idx]], dtype=torch.long).to(DEVICE)
    
    # Convert indices to tokens
    generated_tokens = [tgt_vocab.idx2word[idx] for idx in generated_indices]
    
    # Join tokens
    code = ' '.join(generated_tokens)
    
    return code


def main():
    """Interactive code generation."""
    
    print("="*60)
    print("Python Code Generator from Natural Language")
    print("="*60)
    print("\nLoading model...")
    
    try:
        model, src_vocab, tgt_vocab = load_model_and_vocab()
        print("✓ Model loaded successfully!")
        print(f"✓ Device: {DEVICE}")
        print(f"✓ Source vocabulary size: {len(src_vocab)}")
        print(f"✓ Target vocabulary size: {len(tgt_vocab)}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    print("\n" + "="*60)
    print("Enter a function description (or 'quit' to exit)")
    print("="*60)
    
    print("\nExample inputs:")
    print('  "calculate the sum of two numbers"')
    print('  "return the maximum value from a list"')
    print('  "check if a number is even"')
    print()
    
    while True:
        try:
            # Get input
            docstring = input("\n📝 Docstring: ").strip()
            
            if docstring.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            if not docstring:
                print("Please enter a valid description.")
                continue
            
            # Generate code
            print("\n⏳ Generating code...")
            code = generate_code(model, src_vocab, tgt_vocab, docstring)
            
            # Display result
            print("\n" + "="*60)
            print("Generated Python Code:")
            print("="*60)
            print(code)
            print("="*60)
            
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            continue


if __name__ == "__main__":
    main()
