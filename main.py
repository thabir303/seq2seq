#!/usr/bin/env python3
"""
Main entry point for Seq2Seq Text-to-Python Code Generation.

This script provides a unified interface to train, evaluate, and visualize
all three Seq2Seq models for the assignment.

Usage:
    python main.py train --model all
    python main.py evaluate --model all
    python main.py visualize --num_examples 5
    python main.py demo
"""

import os
import sys
import argparse


def run_training(args):
    """Run training script."""
    cmd = f"python train.py --model {args.model} --epochs {args.epochs}"
    if args.batch_size:
        cmd += f" --batch_size {args.batch_size}"
    if args.lr:
        cmd += f" --lr {args.lr}"
    if args.train_size:
        cmd += f" --train_size {args.train_size}"
    
    print(f"Running: {cmd}")
    os.system(cmd)


def run_evaluation(args):
    """Run evaluation script."""
    cmd = f"python evaluate.py --model {args.model}"
    if args.batch_size:
        cmd += f" --batch_size {args.batch_size}"
    
    print(f"Running: {cmd}")
    os.system(cmd)


def run_visualization(args):
    """Run attention visualization script."""
    cmd = f"python visualize_attention.py --num_examples {args.num_examples}"
    
    print(f"Running: {cmd}")
    os.system(cmd)


def run_demo(args):
    """Run a quick demo with small data."""
    print("="*60)
    print("QUICK DEMO - Training with minimal data")
    print("="*60)
    
    # Quick training
    cmd = "python train.py --model all --epochs 2 --train_size 500 --batch_size 32"
    print(f"\nStep 1: Training (minimal)\nRunning: {cmd}")
    os.system(cmd)
    
    # Quick evaluation
    cmd = "python evaluate.py --model all"
    print(f"\nStep 2: Evaluation\nRunning: {cmd}")
    os.system(cmd)
    
    # Attention visualization
    cmd = "python visualize_attention.py --num_examples 3"
    print(f"\nStep 3: Attention Visualization\nRunning: {cmd}")
    os.system(cmd)
    
    print("\n" + "="*60)
    print("DEMO COMPLETE!")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description='Seq2Seq Text-to-Python Code Generation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    Train all models:
        python main.py train --model all --epochs 20
    
    Evaluate all models:
        python main.py evaluate --model all
    
    Visualize attention:
        python main.py visualize --num_examples 5
    
    Run quick demo:
        python main.py demo
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train models')
    train_parser.add_argument('--model', type=str, default='all',
                              choices=['vanilla_rnn', 'lstm', 'lstm_attention', 'all'])
    train_parser.add_argument('--epochs', type=int, default=20)
    train_parser.add_argument('--batch_size', type=int, default=None)
    train_parser.add_argument('--lr', type=float, default=None)
    train_parser.add_argument('--train_size', type=int, default=None)
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate models')
    eval_parser.add_argument('--model', type=str, default='all',
                             choices=['vanilla_rnn', 'lstm', 'lstm_attention', 'all'])
    eval_parser.add_argument('--batch_size', type=int, default=None)
    
    # Visualize command
    vis_parser = subparsers.add_parser('visualize', help='Visualize attention')
    vis_parser.add_argument('--num_examples', type=int, default=5)
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run quick demo')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        run_training(args)
    elif args.command == 'evaluate':
        run_evaluation(args)
    elif args.command == 'visualize':
        run_visualization(args)
    elif args.command == 'demo':
        run_demo(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
