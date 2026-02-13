"""
Script to help generate the final PDF report after training and evaluation.
This script collects all results and creates a markdown report with plots.
"""

import json
import os
from datetime import datetime

def generate_report():
    """Generate complete report with results."""
    
    print("="*60)
    print("Report Generation Helper")
    print("="*60)
    
    # Check if results exist
    results_dir = "results"
    viz_dir = "visualizations"
    
    if not os.path.exists(f"{results_dir}/all_models_evaluation.json"):
        print("\n❌ Error: Evaluation results not found!")
        print("Please run: python evaluate.py --model all")
        return
    
    # Load results
    with open(f"{results_dir}/all_models_evaluation.json", 'r') as f:
        results = json.load(f)
    
    print("\n📊 Model Performance Summary:")
    print("-" * 60)
    print(f"{'Model':<20} {'BLEU':<10} {'Token Acc':<12} {'Exact Match'}")
    print("-" * 60)
    
    for model_name, metrics in results['models'].items():
        print(f"{model_name:<20} {metrics['bleu_score']:<10.4f} "
              f"{metrics['token_accuracy']:<12.4f} {metrics['exact_match_accuracy']:.4f}")
    
    print("\n📁 Available Visualizations:")
    print("-" * 60)
    
    # List all generated plots
    if os.path.exists(viz_dir):
        for file in sorted(os.listdir(viz_dir)):
            if file.endswith('.png'):
                print(f"  ✓ {file}")
    
    print("\n📝 Next Steps to Complete Report:")
    print("-" * 60)
    print("1. Open REPORT_TEMPLATE.md")
    print("2. Fill in [TODO] sections with above results")
    print("3. Insert visualization images:")
    
    if os.path.exists(viz_dir):
        print("\n   In the markdown, use:")
        for file in sorted(os.listdir(viz_dir)):
            if file.endswith('.png'):
                print(f"   ![{file}](visualizations/{file})")
    
    print("\n4. Convert markdown to PDF:")
    print("   Option A: Use Pandoc:")
    print("     pandoc REPORT_TEMPLATE.md -o REPORT.pdf")
    print("\n   Option B: Use online converter:")
    print("     https://www.markdowntopdf.com/")
    print("\n   Option C: Copy to Google Docs and export as PDF")
    
    print("\n5. Add example predictions from results:")
    with open(f"{results_dir}/vanilla_rnn_evaluation.json", 'r') as f:
        sample = json.load(f)
        if 'sample_predictions' in sample:
            print(f"\n   Sample predictions available: {len(sample['sample_predictions'])} examples")
    
    print("\n" + "="*60)
    print("Report generation complete!")
    print("="*60)

if __name__ == "__main__":
    generate_report()
