"""
Evaluation metrics for Seq2Seq code generation.

Metrics:
    - Token-level Accuracy: Percentage of correctly predicted tokens
    - BLEU Score: N-gram overlap between generated and reference code
    - Exact Match Accuracy: Percentage of exactly correct outputs

These metrics are used during evaluation on the test set as per assignment requirements.
"""

import torch
from typing import List, Tuple, Dict, Optional
from collections import Counter
import math
import ast


def calculate_bleu(references: List[List[str]], 
                   hypotheses: List[List[str]],
                   max_n: int = 4,
                   weights: Optional[Tuple[float, ...]] = None) -> float:
    """
    Calculate corpus-level BLEU score.
    
    BLEU measures N-gram overlap between generated (hypothesis) and 
    reference sequences. Higher is better (max 1.0).
    
    Args:
        references: List of reference token sequences
        hypotheses: List of hypothesis (generated) token sequences
        max_n: Maximum n-gram order (default: 4 for BLEU-4)
        weights: Weights for each n-gram order (default: uniform)
        
    Returns:
        BLEU score (0.0 to 1.0)
    """
    if weights is None:
        weights = tuple([1.0 / max_n] * max_n)
    
    # Collect statistics
    matches = [0] * max_n
    totals = [0] * max_n
    ref_length = 0
    hyp_length = 0
    
    for ref, hyp in zip(references, hypotheses):
        ref_length += len(ref)
        hyp_length += len(hyp)
        
        # Count n-gram matches for each order
        for n in range(1, max_n + 1):
            if len(hyp) >= n:
                # Get n-grams
                ref_ngrams = Counter(tuple(ref[i:i+n]) for i in range(len(ref) - n + 1))
                hyp_ngrams = Counter(tuple(hyp[i:i+n]) for i in range(len(hyp) - n + 1))
                
                # Count matches (clipped by reference count)
                for ngram, count in hyp_ngrams.items():
                    matches[n-1] += min(count, ref_ngrams.get(ngram, 0))
                
                totals[n-1] += len(hyp) - n + 1
    
    # Calculate precision for each n-gram order
    precisions = []
    for i in range(max_n):
        if totals[i] > 0:
            precisions.append(matches[i] / totals[i])
        else:
            precisions.append(0.0)
    
    # Calculate brevity penalty
    if hyp_length == 0:
        return 0.0
    
    if hyp_length < ref_length:
        bp = math.exp(1 - ref_length / hyp_length)
    else:
        bp = 1.0
    
    # Calculate weighted geometric mean of precisions
    log_precision_sum = 0.0
    for i, (precision, weight) in enumerate(zip(precisions, weights)):
        if precision > 0:
            log_precision_sum += weight * math.log(precision)
        else:
            # Smoothing: add small epsilon
            log_precision_sum += weight * math.log(1e-10)
    
    bleu = bp * math.exp(log_precision_sum)
    return bleu


def calculate_accuracy(predictions: List[List[int]], 
                       targets: List[List[int]],
                       pad_idx: int = 0,
                       ignore_special: bool = True) -> float:
    """
    Calculate token-level accuracy.
    
    Measures percentage of correctly predicted tokens (excluding padding).
    
    Args:
        predictions: List of predicted token index sequences
        targets: List of target token index sequences
        pad_idx: Padding token index to ignore
        ignore_special: Whether to ignore special tokens (SOS, EOS)
        
    Returns:
        Token accuracy (0.0 to 1.0)
    """
    correct = 0
    total = 0
    
    # Special token indices (if ignoring)
    special_indices = {0, 1, 2, 3} if ignore_special else {pad_idx}
    
    for pred, tgt in zip(predictions, targets):
        min_len = min(len(pred), len(tgt))
        
        for i in range(min_len):
            if tgt[i] not in special_indices:
                total += 1
                if pred[i] == tgt[i]:
                    correct += 1
        
        # Count extra target tokens as incorrect
        for i in range(min_len, len(tgt)):
            if tgt[i] not in special_indices:
                total += 1
    
    if total == 0:
        return 0.0
    
    return correct / total


def calculate_exact_match(references: List[str], 
                          hypotheses: List[str]) -> float:
    """
    Calculate exact match accuracy.
    
    Measures percentage of outputs that exactly match the reference.
    Especially meaningful for small code snippets.
    
    Args:
        references: List of reference strings
        hypotheses: List of hypothesis (generated) strings
        
    Returns:
        Exact match accuracy (0.0 to 1.0)
    """
    if len(references) == 0:
        return 0.0
    
    exact_matches = sum(1 for ref, hyp in zip(references, hypotheses) if ref.strip() == hyp.strip())
    return exact_matches / len(references)


def calculate_syntax_accuracy(generated_codes: List[str]) -> Tuple[float, Dict[str, int]]:
    """
    Calculate syntax accuracy using Python AST.
    
    Attempts to parse generated code and counts:
        - Syntax errors
        - Indentation errors
        - Other errors
    
    Args:
        generated_codes: List of generated code strings
        
    Returns:
        Tuple of (syntax accuracy, error type counts)
    """
    valid_count = 0
    error_types = {
        'syntax_error': 0,
        'indentation_error': 0,
        'other_error': 0
    }
    
    for code in generated_codes:
        try:
            # Try to parse the code
            ast.parse(code)
            valid_count += 1
        except IndentationError:
            error_types['indentation_error'] += 1
        except SyntaxError:
            error_types['syntax_error'] += 1
        except Exception:
            error_types['other_error'] += 1
    
    if len(generated_codes) == 0:
        return 0.0, error_types
    
    accuracy = valid_count / len(generated_codes)
    return accuracy, error_types


def analyze_errors(references: List[List[str]], 
                   hypotheses: List[List[str]]) -> Dict[str, float]:
    """
    Analyze different types of errors in generated code.
    
    Error types analyzed:
        - Missing tokens
        - Extra tokens
        - Wrong tokens
        - Order errors
    
    Args:
        references: List of reference token sequences
        hypotheses: List of hypothesis token sequences
        
    Returns:
        Dictionary with error statistics
    """
    total_missing = 0
    total_extra = 0
    total_wrong = 0
    total_ref_tokens = 0
    
    for ref, hyp in zip(references, hypotheses):
        ref_counter = Counter(ref)
        hyp_counter = Counter(hyp)
        
        total_ref_tokens += len(ref)
        
        # Missing: in reference but not in hypothesis
        for token, count in ref_counter.items():
            diff = count - hyp_counter.get(token, 0)
            if diff > 0:
                total_missing += diff
        
        # Extra: in hypothesis but not in reference
        for token, count in hyp_counter.items():
            diff = count - ref_counter.get(token, 0)
            if diff > 0:
                total_extra += diff
        
        # Wrong: tokens at same position but different
        min_len = min(len(ref), len(hyp))
        wrong = sum(1 for i in range(min_len) if ref[i] != hyp[i])
        total_wrong += wrong
    
    if total_ref_tokens == 0:
        return {
            'missing_rate': 0.0,
            'extra_rate': 0.0,
            'wrong_rate': 0.0
        }
    
    return {
        'missing_rate': total_missing / total_ref_tokens,
        'extra_rate': total_extra / total_ref_tokens,
        'wrong_rate': total_wrong / total_ref_tokens
    }


def calculate_metrics_by_length(references: List[List[str]],
                                hypotheses: List[List[str]],
                                lengths: List[int],
                                bins: List[int] = [10, 20, 30, 40, 50]) -> Dict[str, Dict[str, float]]:
    """
    Calculate metrics grouped by docstring length.
    
    Useful for analyzing performance vs. input length.
    
    Args:
        references: Reference token sequences
        hypotheses: Generated token sequences
        lengths: Docstring lengths for each example
        bins: Length bin boundaries
        
    Returns:
        Dictionary mapping length bins to metrics
    """
    # Group by length bins
    binned = {f"0-{bins[0]}": ([], [])}
    for i in range(len(bins) - 1):
        binned[f"{bins[i]+1}-{bins[i+1]}"] = ([], [])
    binned[f"{bins[-1]+1}+"] = ([], [])
    
    for ref, hyp, length in zip(references, hypotheses, lengths):
        # Find appropriate bin
        if length <= bins[0]:
            bin_key = f"0-{bins[0]}"
        elif length > bins[-1]:
            bin_key = f"{bins[-1]+1}+"
        else:
            for i in range(len(bins) - 1):
                if bins[i] < length <= bins[i+1]:
                    bin_key = f"{bins[i]+1}-{bins[i+1]}"
                    break
        
        binned[bin_key][0].append(ref)
        binned[bin_key][1].append(hyp)
    
    # Calculate metrics for each bin
    results = {}
    for bin_key, (refs, hyps) in binned.items():
        if len(refs) > 0:
            results[bin_key] = {
                'bleu': calculate_bleu(refs, hyps),
                'count': len(refs)
            }
    
    return results


if __name__ == "__main__":
    # Test metrics
    references = [
        ["def", "max_value", "(", "nums", ")", ":", "return", "max", "(", "nums", ")"],
        ["def", "add", "(", "a", ",", "b", ")", ":", "return", "a", "+", "b"]
    ]
    
    hypotheses = [
        ["def", "max_value", "(", "nums", ")", ":", "return", "max", "(", "nums", ")"],
        ["def", "add", "(", "a", ",", "b", ")", ":", "return", "a", "-", "b"]
    ]
    
    print(f"BLEU Score: {calculate_bleu(references, hypotheses):.4f}")
    
    # Test exact match
    ref_strings = [" ".join(r) for r in references]
    hyp_strings = [" ".join(h) for h in hypotheses]
    print(f"Exact Match: {calculate_exact_match(ref_strings, hyp_strings):.4f}")
    
    # Test syntax accuracy
    codes = [
        "def foo(): return 1",
        "def bar( return 2",  # Syntax error
        "def baz():\nreturn 3"  # Indentation error
    ]
    acc, errors = calculate_syntax_accuracy(codes)
    print(f"Syntax Accuracy: {acc:.4f}")
    print(f"Error Types: {errors}")
