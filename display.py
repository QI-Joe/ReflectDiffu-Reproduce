#!/usr/bin/env python3
"""
Display evaluation results for ReflectDiffu model.

This script loads a trained model and evaluates it on both:
1. Relevance metrics: BLEU-1/2/3/4 and BARTScore  
2. Informativeness metrics: Perplexity (PPL) and Distinct-1/2

Usage:
    python display.py --model_path path/to/model.pt --data_path dataset/test.pkl --max_samples 100
"""

import argparse
import torch
import time
from pathlib import Path
import logging

# Import evaluation modules
from evaluation.relevance_evaluation import ReflectDiffuEvaluator as RelevanceEvaluator
from evaluation.informativeness import InformativenessEvaluator, print_informativeness_results

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_model(model_path: str, device: torch.device):
    """
    Load ReflectDiffu model from checkpoint file.
    
    Args:
        model_path: Path to saved model (.pt file)
        device: Torch device for model loading
        
    Returns:
        Loaded model in evaluation mode
    """
    logger.info(f"Loading model from {model_path}")
    model = torch.load(model_path, map_location=device)
    model.eval()
    logger.info("Model loaded successfully")
    return model


def evaluate_relevance(model, data_path: str, max_samples: int, device: torch.device):
    """
    Evaluate relevance metrics (BLEU and BARTScore).
    
    Args:
        model: Loaded ReflectDiffu model
        data_path: Path to test data
        max_samples: Maximum number of samples to evaluate
        device: Torch device
        
    Returns:
        Relevance evaluation results
    """
    logger.info("Starting relevance evaluation (BLEU + BARTScore)...")
    
    # Create relevance evaluator
    relevance_evaluator = RelevanceEvaluator(
        encoder=model.encoder,
        it_module=model.it_module,
        decoder=model.decoder_with_loss,
        device=device
    )
    
    # Prepare evaluation data
    eval_samples = relevance_evaluator.prepare_evaluation_data()
    
    
    # Run evaluation
    start_time = time.time()
    results = relevance_evaluator.evaluate(
        eval_samples=eval_samples,
        max_len=32,
        use_pointer=False,
        temperature=1.0,
        top_k=0,
        top_p=1.0
    )
    eval_time = time.time() - start_time
    
    logger.info(f"Relevance evaluation completed in {eval_time:.2f}s")
    return results


def evaluate_informativeness(model, data_path: str, max_samples: int, device: torch.device):
    """
    Evaluate informativeness metrics (Perplexity and Distinct-n).
    
    Args:
        model: Loaded ReflectDiffu model
        data_path: Path to test data
        max_samples: Maximum number of samples to evaluate
        device: Torch device
        
    Returns:
        Informativeness evaluation results
    """
    logger.info("Starting informativeness evaluation (PPL + Distinct-n)...")
    
    # Create informativeness evaluator
    informativeness_evaluator = InformativenessEvaluator(
        encoder=model.encoder,
        it_module=model.it_module,
        decoder=model.decoder_with_loss,
        device=device
    )
    
    # Run evaluation
    start_time = time.time()
    results = informativeness_evaluator.evaluate_from_file(
        data_path=data_path,
        max_samples=max_samples
    )
    eval_time = time.time() - start_time
    
    logger.info(f"Informativeness evaluation completed in {eval_time:.2f}s")
    return results


def print_relevance_results(results):
    """Print relevance evaluation results in a formatted way."""
    print(f"\n{'='*60}")
    print(f"{'RELEVANCE EVALUATION RESULTS':^60}")
    print(f"{'='*60}")
    print(f"Samples:      {results.num_samples}")
    print(f"Gen Time:     {results.generation_time:.2f}s")
    print(f"{'='*60}")
    print(f"BLEU Metrics:")
    print(f"  BLEU-1:     {results.bleu_1:.2f}%")
    print(f"  BLEU-2:     {results.bleu_2:.2f}%")
    print(f"  BLEU-3:     {results.bleu_3:.2f}%")
    print(f"  BLEU-4:     {results.bleu_4:.2f}%")
    print(f"  BLEU:       {results.bleu_overall:.2f}")
    print(f"  BP:         {results.brevity_penalty:.3f}")
    print(f"{'='*60}")
    print(f"BARTScore:    {results.bart_score:.4f}")
    print(f"{'='*60}")
    print(f"Length Stats:")
    print(f"  Avg Hyp:    {results.avg_hyp_length:.1f} words")
    print(f"  Avg Ref:    {results.avg_ref_length:.1f} words")
    print(f"{'='*60}")


def print_summary(relevance_results, informativeness_results):
    """Print a summary of both evaluations."""
    print(f"\n{'='*60}")
    print(f"{'EVALUATION SUMMARY':^60}")
    print(f"{'='*60}")
    print(f"Dataset Size:     {relevance_results.num_samples} samples")
    print(f"{'='*60}")
    print(f"Relevance Metrics:")
    print(f"  BLEU-4:         {relevance_results.bleu_4:.2f}%")
    print(f"  BLEU Overall:   {relevance_results.bleu_overall:.2f}")
    print(f"  BARTScore:      {relevance_results.bart_score:.4f}")
    print(f"{'='*60}")
    print(f"Informativeness Metrics:")
    print(f"  Perplexity:     {informativeness_results.perplexity:.2f}")
    print(f"  Distinct-1:     {informativeness_results.distinct_1:.3f}")
    print(f"  Distinct-2:     {informativeness_results.distinct_2:.3f}")
    print(f"{'='*60}")
    print(f"Performance:")
    print(f"  Total Time:     {relevance_results.generation_time:.2f}s")
    print(f"  Avg Length:     {relevance_results.avg_hyp_length:.1f} words")
    print(f"{'='*60}")

import os
import json
import time
from typing import Any, Dict, Optional

def _to_serializable(results_obj: Any) -> Dict[str, Any]:
    """
    尝试将任意结果对象转为可 JSON 序列化的字典。
    优先使用 vars()；若失败则回退到 getattr 方式。
    """
    try:
        data = dict(vars(results_obj))
    except TypeError:
        # 对于不支持 vars 的对象，尝试用常见字段名收集
        fields = [
            "num_samples", "generation_time",
            "bleu_1", "bleu_2", "bleu_3", "bleu_4", "bleu_overall", "brevity_penalty",
            "bart_score",
            "avg_hyp_length", "avg_ref_length",
            "perplexity", "distinct_1", "distinct_2",
        ]
        data = {}
        for f in fields:
            if hasattr(results_obj, f):
                data[f] = getattr(results_obj, f)
    # 过滤不可序列化的值（如 Tensor），尽量转基本类型
    def to_base(v):
        if hasattr(v, "item") and callable(v.item):
            try:
                return v.item()
            except Exception:
                pass
        if isinstance(v, (list, tuple)):
            return [to_base(x) for x in v]
        if isinstance(v, dict):
            return {k: to_base(x) for k, x in v.items()}
        return v
    return {k: to_base(v) for k, v in data.items()}


def save_eval_results(
    relevance_results: Any,
    informativeness_results: Any,
    save_dir: str,
    run_name: Optional[str] = None,
) -> str:
    """
    保存评估结果到指定目录的 JSON 文件，返回文件路径。
    """
    os.makedirs(save_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    run_tag = run_name or "run"
    fname = f"eval_{run_tag}_{ts}.json"
    fpath = os.path.join(save_dir, fname)

    payload = {
        "meta": {
            "run_name": run_name,
            "timestamp": ts,
        },
        "relevance": _to_serializable(relevance_results),
        "informativeness": _to_serializable(informativeness_results),
    }
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return fpath



def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate ReflectDiffu model on relevance and informativeness metrics")
    parser.add_argument("--model_path", type=str, default=r"output/best_model/best_model_09-27-14-06_22.68.pt", help="Path to saved model (.pt file)")
    parser.add_argument("--data_path", type=str, default="dataset/available_dataset/test.pkl", help="Path to test data")
    parser.add_argument("--max_samples", type=int, default=100, help="Maximum number of samples to evaluate")
    parser.add_argument("--quiet", action="store_true", default=True, help="Reduce logging output")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    logger.info(f"Using device: {device}")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Max samples: {args.max_samples}")
    
    # Verify paths exist
    if not Path(args.model_path).exists():
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    if not Path(args.data_path).exists():
        raise FileNotFoundError(f"Data file not found: {args.data_path}")
    
    # Load model
    model = load_model(args.model_path, device)
    
    print(f"\n🚀 Starting evaluation with {args.max_samples} samples...")
    print(f"📁 Model: {args.model_path}")
    print(f"📊 Data: {args.data_path}")
    print(f"💻 Device: {device}")
    
    # Evaluate relevance metrics
    print(f"\n📈 Evaluating relevance metrics...")
    relevance_results = evaluate_relevance(model, args.data_path, args.max_samples, device)
    
    # Evaluate informativeness metrics  
    print(f"\n📊 Evaluating informativeness metrics...")
    informativeness_results = evaluate_informativeness(model, args.data_path, args.max_samples, device)
    
    # Display results
    print_relevance_results(relevance_results)
    print_informativeness_results(informativeness_results, "INFORMATIVENESS EVALUATION RESULTS")
    print_summary(relevance_results, informativeness_results)
    
    save_dir = "output/eval_logs"
    json_path = save_eval_results(
    relevance_results=relevance_results,
    informativeness_results=informativeness_results,
    save_dir=save_dir,
    run_name="reflectdiffu_val"  # 可自定义
)
    
    print(f"\n✅ Evaluation completed successfully!")
        


if __name__ == "__main__":
    main()