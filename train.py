import argparse
import os
import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional, List, Dict, Any
import logging
from model_intergration import ReflectDiffu
from evaluation import EvaluationConfig, TrainingEvaluator, EVALUATION_AVAILABLE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_optimizer(model: ReflectDiffu, lr: float = 1e-4, weight_decay: float = 1e-5):
    """Create optimizer for the ReflectDiffu model."""
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def train_step(model: ReflectDiffu, optimizer, delta=1.0, zeta=1.0, eta=1.0, verbose=True):
    """
    Perform a single training step.
    
    Args:
        model: ReflectDiffu model
        optimizer: Optimizer
        delta, zeta, eta: Loss weights
        verbose: Whether to print loss details
        
    Returns:
        Dict with loss values
    """
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model.forward()
    
    # Compute joint loss
    joint_loss = model.compute_joint_loss(outputs, delta, zeta, eta)
    
    # Backward pass
    joint_loss.backward()
    optimizer.step()
    
    # Prepare results
    results = {
        'joint_loss': float(joint_loss),
        'Lem': float(outputs['Lem']),
        'Ltwice': float(outputs['Ltwice']),
        'Lres': float(outputs['Lres'])
    }
    
    if verbose:
        print(f"Joint Loss: {results['joint_loss']:.4f} = "
              f"{delta}*{results['Lem']:.4f} + "
              f"{zeta}*{results['Ltwice']:.4f} + "
              f"{eta}*{results['Lres']:.4f}")
    
    return results


def train_epoch(model: ReflectDiffu, optimizer, num_steps: int = 10, 
                delta=1.0, zeta=1.0, eta=1.0, verbose=True):
    """
    Train for one epoch (multiple steps on the same data).
    
    Args:
        model: ReflectDiffu model
        optimizer: Optimizer
        num_steps: Number of training steps
        delta, zeta, eta: Loss weights
        verbose: Whether to print progress
        
    Returns:
        List of loss dictionaries for each step
    """
    model.train()
    epoch_losses = []
    
    if verbose:
        print(f"\\n=== Training Epoch ({num_steps} steps) ===")
    
    for step in range(num_steps):
        step_losses = train_step(model, optimizer, delta, zeta, eta, verbose=False)
        epoch_losses.append(step_losses)
        
        if verbose and (step + 1) % max(1, num_steps // 5) == 0:
            print(f"Step {step+1}/{num_steps}: Joint Loss = {step_losses['joint_loss']:.4f}")
    
    # Compute average losses
    avg_losses = {
        key: sum(losses[key] for losses in epoch_losses) / len(epoch_losses)
        for key in epoch_losses[0].keys()
    }
    
    if verbose:
        print(f"\\nEpoch Average Losses:")
        print(f"  Joint Loss: {avg_losses['joint_loss']:.4f}")
        print(f"  Lem: {avg_losses['Lem']:.4f}")
        print(f"  Ltwice: {avg_losses['Ltwice']:.4f}")
        print(f"  Lres: {avg_losses['Lres']:.4f}")
    
    return epoch_losses


def train_with_evaluation(args, eval_config: Optional[EvaluationConfig] = None):
    """集成评估的训练函数 - 按照设计文档实现"""
    device = torch.device('cuda:0' if args.cuda and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. 模型初始化 (保持不变)
    model = ReflectDiffu(
        vocab_size=1000,  # Will be updated based on data
        model_dim=128,
        device=device,
        use_era=args.use_era,
        coverage_weight=0.0
    )
    
    # Initialize all components
    model.initialize_all(
        data_path=args.ec_data,
        batch_size=args.batch_size,
        max_length=64
    )
    
    # Create optimizer
    optimizer = create_optimizer(model, lr=1e-4)
    
    # 2. 评估器初始化 (新增)
    training_evaluator = None
    if eval_config and EVALUATION_AVAILABLE:
        try:
            training_evaluator = TrainingEvaluator(model, eval_config)
            training_evaluator.initialize()
            print("✅ Training evaluator initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize evaluator: {e}")
            training_evaluator = None
    elif eval_config and not EVALUATION_AVAILABLE:
        logger.warning("Evaluation requested but evaluation module not available")
    
    # 3. 训练循环 (轻微修改)
    print("\\n🚀 Starting Training with Evaluation...")
    num_epochs = 3
    steps_per_epoch = 5
    total_steps = 0
    
    for epoch in range(num_epochs):
        print(f"\\n{'='*50}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'='*50}")
        
        # 训练一个epoch
        epoch_losses = train_epoch(
            model=model,
            optimizer=optimizer,
            num_steps=steps_per_epoch,
            delta=args.delta,
            zeta=args.zeta,
            eta=args.eta,
            verbose=True
        )
        
        total_steps += steps_per_epoch
        print(f"[Reflect-Diffu] | Completed Epoch {epoch + 1}. For loss {epoch_losses[-1]}")
        
        # 检查是否需要评估
        if training_evaluator and training_evaluator.should_evaluate(epoch + 1, total_steps):
            print("\\n🔍 Running evaluation...")
            model.eval()
            with torch.no_grad():
                eval_results = training_evaluator.evaluate()
                if eval_results:
                    training_evaluator.log_results(eval_results, epoch + 1, total_steps)
            model.train()
    
    print("\\n✅ Training completed!")
    
    # 4. 最终评估
    if training_evaluator and eval_config.eval_at_end:
        print("\\n🔍 Running final evaluation...")
        model.eval()
        with torch.no_grad():
            final_results = training_evaluator.evaluate()
            if final_results:
                training_evaluator.log_results(final_results, num_epochs, total_steps)
    else:
        # 传统的最终测试
        print("\\n=== Final Model Test ===")
        model.eval()
        with torch.no_grad():
            final_outputs = model.forward()
            final_joint_loss = model.compute_joint_loss(
                final_outputs, args.delta, args.zeta, args.eta
            )
            print(f"Final Joint Loss: {float(final_joint_loss):.4f}")
    
    # 输出评估器状态摘要
    if training_evaluator:
        status = training_evaluator.get_status()
        print(f"\\n📊 Evaluation Summary:")
        print(f"  Best BLEU-4: {status['best_bleu4']:.2f}%")
        print(f"  Best BARTScore: {status['best_bart_score']:.4f}")
        print(f"  Total Samples: {status['num_eval_samples']}")
    
    return model


def train(args):
    """原始训练函数 (向后兼容)"""
    return train_with_evaluation(args, eval_config=None)
    

def main():
    ap = argparse.ArgumentParser(description="ReflectDiffu Training with Integrated Evaluation")
    
    # Original training arguments
    ap.add_argument('--ec_data', type=str, default=r"dataset/emotion_labels.pkl", 
                   help='Path to emotion contagion pickle file or directory')
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--cuda', action='store_true', default=True)
    ap.add_argument('--use_era', action='store_true', default=False, 
                   help='Try to use ERA model for h_tilde')
    ap.add_argument('--delta', type=float, default=1.0, help='Weight for emotion loss')
    ap.add_argument('--zeta', type=float, default=1.0, help='Weight for intent twice loss')
    ap.add_argument('--eta', type=float, default=1.0, help='Weight for response loss')
    
    # Evaluation configuration arguments
    ap.add_argument('--enable_eval', action='store_true', default=False,
                   help='Enable evaluation during training')
    ap.add_argument('--eval_every_epochs', type=int, default=1,
                   help='Evaluate every N epochs')
    ap.add_argument('--eval_every_steps', type=int, default=None,
                   help='Evaluate every N steps (overrides eval_every_epochs)')
    ap.add_argument('--eval_data_path', type=str, default=None,
                   help='Path to evaluation data (uses training data if not specified)')
    ap.add_argument('--max_eval_samples', type=int, default=100,
                   help='Maximum number of evaluation samples')
    ap.add_argument('--max_gen_length', type=int, default=32,
                   help='Maximum generation length for evaluation')
    ap.add_argument('--eval_temperature', type=float, default=1.0,
                   help='Temperature for evaluation generation')
    ap.add_argument('--eval_use_pointer', action='store_true', default=False,
                   help='Use pointer-generator during evaluation')
    ap.add_argument('--skip_bart_score', action='store_true', default=False,
                   help='Skip BARTScore computation (faster evaluation)')
    ap.add_argument('--eval_results_dir', type=str, default="evaluation_results",
                   help='Directory to save evaluation results')
    ap.add_argument('--eval_log_examples', action='store_true', default=True,
                   help='Log generation examples during evaluation')
    ap.add_argument('--eval_num_examples', type=int, default=5,
                   help='Number of examples to log')
    
    args = ap.parse_args()
    
    # Create evaluation configuration if enabled
    eval_config = None
    if args.enable_eval:
        eval_config = EvaluationConfig(
            eval_every_epochs=args.eval_every_epochs,
            eval_every_steps=args.eval_every_steps,
            eval_at_end=True,
            eval_data_path=args.eval_data_path,
            max_eval_samples=args.max_eval_samples,
            max_gen_length=args.max_gen_length,
            use_pointer=args.eval_use_pointer,
            temperature=args.eval_temperature,
            top_k=0,
            top_p=1.0,
            compute_bleu=True,
            compute_bart_score=not args.skip_bart_score,
            bart_checkpoint="facebook/bart-large-cnn",
            save_results=True,
            results_dir=args.eval_results_dir,
            log_examples=args.eval_log_examples,
            num_examples=args.eval_num_examples,
            eval_batch_size=1,
            disable_tqdm=False
        )
        
        print("🔍 Evaluation enabled with configuration:")
        print(f"  Eval every epochs: {eval_config.eval_every_epochs}")
        print(f"  Eval every steps: {eval_config.eval_every_steps}")
        print(f"  Max eval samples: {eval_config.max_eval_samples}")
        print(f"  Max gen length: {eval_config.max_gen_length}")
        print(f"  Use pointer: {eval_config.use_pointer}")
        print(f"  Compute BARTScore: {eval_config.compute_bart_score}")
        print(f"  Results dir: {eval_config.results_dir}")
    
    # Run training with or without evaluation
    if eval_config:
        train_with_evaluation(args, eval_config)
    else:
        train(args)



if __name__ == '__main__':
    main()