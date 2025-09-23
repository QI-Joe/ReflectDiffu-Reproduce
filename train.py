import argparse
import os
import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional, List, Dict, Any
import logging
from model_intergration import ReflectDiffu
from evaluation import EvaluationConfig, TrainingEvaluator, EVALUATION_AVAILABLE
from early_stopping import EarlyStopping, create_early_stopping
from torch.utils.tensorboard import SummaryWriter
from lr_scheduler import create_scheduler
from config_loader import load_config, save_effective, as_dotdict


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_optimizer(model: ReflectDiffu, lr: float = 1e-4, weight_decay: float = 1e-5):
    """Create optimizer for the ReflectDiffu model."""
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def train_step(model: ReflectDiffu, optimizer, batch_data, delta=1.0, zeta=1.0, eta=1.0, verbose=True):
    """
    Perform a single training step on one batch.
    
    Args:
        model: ReflectDiffu model
        optimizer: Optimizer
        batch_data: Single batch data dictionary
        delta, zeta, eta: Loss weights
        verbose: Whether to print loss details
        
    Returns:
        Dict with loss values
    """
    optimizer.zero_grad()
    
    # Forward pass with specific batch
    outputs = model.forward(batch_data)
    
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


def train_epoch(model: ReflectDiffu, optimizer, batches, scheduler, num_epochs: int = 1, 
                delta=1.0, zeta=1.0, eta=1.0, verbose=True, writer: SummaryWriter=None, start_step:int=0,):
    """
    Train for one epoch using multiple batches.
    
    Args:
        model: ReflectDiffu model
        optimizer: Optimizer
        batches: List of batch data dictionaries
        num_epochs: Number of times to iterate over all batches
        delta, zeta, eta: Loss weights
        verbose: Whether to print progress
        
    Returns:
        List of loss dictionaries for each step
    """
    model.train()
    epoch_losses = []
    
    total_steps = len(batches) * num_epochs
    if verbose:
        print(f"\\n=== Training Epoch ({len(batches)} batches × {num_epochs} epochs = {total_steps} steps) ===")
    
    step = 0
    for batch_idx, batch_data in enumerate(batches):
        if len(batch_data["user"]) != len(batch_data["response"]):
            lens = min(len(batch_data["user"]), len(batch_data["response"]))
            batch_data["user"], batch_data["response"] = batch_data["user"][:lens], batch_data["response"][:lens]
            batch_data["p_intent"] = batch_data["p_intent"][:lens]
            
            
        step_losses = train_step(model, optimizer, batch_data, delta, zeta, eta, verbose=False)
        epoch_losses.append(step_losses)
        
        
        if writer is not None:
            global_step = start_step + step
            writer.add_scalar('train/joint_loss', step_losses['joint_loss'], global_step)
            writer.add_scalar('train/Lem', step_losses['Lem'], global_step)
            writer.add_scalar('train/Ltwice', step_losses['Ltwice'], global_step)
            writer.add_scalar('train/Lres', step_losses['Lres'], global_step)
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('train/lr', current_lr, global_step)
        # Step scheduler after logging so LR logged corresponds to current step
        scheduler.step()
        step += 1
        
        if verbose and (step % max(1, total_steps // 10) == 0):
            print(f"Step {step}/{total_steps} (Batch {batch_idx+1}): Joint Loss = {step_losses['joint_loss']:.4f}")
    
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


def train_with_evaluation(args, eval_config: Optional[EvaluationConfig] = None, unified_cfg: Optional[Dict[str, Any]] = None):
    """集成评估的训练函数 - 按照设计文档实现"""
    device = torch.device('cuda:0' if args.cuda and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. 模型初始化 (保持不变)
    # Resolve config model params
    model_dim = unified_cfg.get('model', {}).get('model_dim', 128) if unified_cfg else 128
    coverage_weight = unified_cfg.get('model', {}).get('coverage_weight', 0.0) if unified_cfg else 0.0
    model = ReflectDiffu(
        vocab_size=1000,
        model_dim=model_dim,
        device=device,
        use_era=args.use_era if not unified_cfg else unified_cfg.get('model', {}).get('era', {}).get('enabled', False),
        coverage_weight=coverage_weight,
        unified_config=unified_cfg
    )
    
    # Initialize all components and get batches
    data_path = args.ec_data if not unified_cfg else unified_cfg['data']['ec_data_path']
    batch_size = args.batch_size if not unified_cfg else unified_cfg['data']['batch_size']
    max_input_length = 64 if not unified_cfg else unified_cfg['data']['max_input_length']
    batches = model.initialize_all(
        data_path=data_path,
        batch_size=batch_size,
        max_length=max_input_length
    )
    
    # Create optimizer
    base_lr = args.lr if not unified_cfg else unified_cfg['optimizer']['lr']
    weight_decay = args.weight_decay if not unified_cfg else unified_cfg['optimizer']['weight_decay']
    optimizer = create_optimizer(model, lr=base_lr, weight_decay=weight_decay)
    # Prepare LR scheduler (needs total steps)
    steps_per_epoch = len(batches)
    num_epochs = args.num_epochs if not unified_cfg else unified_cfg['run']['num_epochs']
    total_training_steps = steps_per_epoch * num_epochs
    if unified_cfg:
        sched_cfg = unified_cfg['optimizer']['scheduler']
        sched_type = sched_cfg.get('type', 'none')
        warmup_steps = sched_cfg.get('warmup_steps', 0)
        if warmup_steps == 0:
            warmup_ratio = sched_cfg.get('warmup_ratio', 0.0)
            warmup_steps = int(total_training_steps * warmup_ratio)
    else:
        sched_type = args.lr_scheduler
        if args.warmup_steps > 0:
            warmup_steps = args.warmup_steps
        else:
            warmup_steps = int(total_training_steps * args.warmup_ratio)
    scheduler = create_scheduler(
        optimizer,
        schedule_type=sched_type,
        warmup_steps=warmup_steps,
        total_steps=total_training_steps
    ) if sched_type != 'none' else None
    if scheduler:
        print(f"LR Scheduler: {sched_type} | warmup_steps={warmup_steps} | total_steps={total_training_steps}")
    
    # 2. 评估器初始化 (新增)
    training_evaluator = TrainingEvaluator(model, eval_config)
    training_evaluator.initialize()
    print("✅ Training evaluator initialized")
    
    early_stopping = create_early_stopping(
        patience=getattr(args, 'early_stopping_patience', 5),
        min_delta=getattr(args, 'early_stopping_min_delta', 0.001),
        mode=getattr(args, 'early_stopping_mode', 'auto'),
        loss_weight=getattr(args, 'early_stopping_loss_weight', 0.7),
        bleu1_weight=getattr(args, 'early_stopping_bleu1_weight', 0.3),
        restore_best_weights=True,
        verbose=True
    )
    print("🛑 Early stopping enabled")
    
    # 3. 训练循环 (修改为使用batches)
    print("\\n🚀 Starting Training with Evaluation...")
    num_epochs = num_epochs  # already resolved above
    total_steps = 0
    
    writer = SummaryWriter(log_dir=os.path.join('logs', 'tensorboard'))
    global_step = 0
    for epoch in range(num_epochs):
        print(f"\\n{'='*50}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'='*50}")
        
        # 训练一个epoch，遍历所有batches
        epoch_losses = train_epoch(
            model=model,
            optimizer=optimizer,
            batches=batches,
            scheduler=scheduler,
            num_epochs=1,
            delta=(args.delta if not unified_cfg else unified_cfg['loss_weights']['delta']),
            zeta=(args.zeta if not unified_cfg else unified_cfg['loss_weights']['zeta']),
            eta=(args.eta if not unified_cfg else unified_cfg['loss_weights']['eta']),
            verbose=True,
            writer=writer,
            start_step=global_step
        )
        total_steps += len(batches)
        global_step += len(batches)
        current_loss = epoch_losses[-1]['joint_loss']
        print(f"[Reflect-Diffu] | Completed Epoch {epoch + 1}. For loss {epoch_losses[-1]['joint_loss']:.4f}, total steps {total_steps}.")
        
        eval_results = None
        if training_evaluator and training_evaluator.should_evaluate(epoch + 1, total_steps):
            print("\\n🔍 Running evaluation...")
            model.eval()
            with torch.no_grad():
                eval_pair = training_evaluator.evaluate()
                if eval_pair:
                    relevance_res, info_res = eval_pair
                    training_evaluator.log_results(relevance_res, info_res, epoch + 1, total_steps)
                    writer.add_scalar('eval/bleu4', getattr(relevance_res, 'bleu_4', 0.0), global_step)
                    if info_res is not None:
                        writer.add_scalar('eval/perplexity', getattr(info_res, 'perplexity', 0.0), global_step)
            model.train()
        
            bleu1_score = getattr(relevance_res, 'bleu_1', 0.0) if eval_pair else 0.0
            
            # 检查是否应该早停
            should_stop = early_stopping(
                loss=current_loss,
                bleu1=bleu1_score,
                model=model,
                epoch=epoch + 1
            )
            
            if should_stop:
                print(f"\\n🛑 Early stopping triggered! Training stopped at epoch {epoch + 1}")
                
                # 显示早停总结
                summary = early_stopping.get_summary()
                print(f"\\n📋 Early Stopping Summary:")
                print(f"  Best loss: {summary['best_loss']:.4f} at epoch {summary['best_epoch']}")
                print(f"  Best BLEU-1: {summary['best_bleu1']:.2f}%")
                print(f"  Training stopped after {summary['total_epochs']} epochs")
                break
    
    # 4. 最终评估
    if training_evaluator and eval_config.eval_at_end:
        print("\\n🔍 Running final evaluation...")
        model.eval()
        with torch.no_grad():
            final_pair = training_evaluator.evaluate()
            if final_pair:
                final_rel, final_info = final_pair
                training_evaluator.log_results(final_rel, final_info, num_epochs, total_steps)
    else:
        # 传统的最终测试 - 使用第一个batch
        print("\\n=== Final Model Test ===")
        model.eval()
        with torch.no_grad():
            final_outputs = model.forward(batches[0] if batches else None)
            final_joint_loss = model.compute_joint_loss(
                final_outputs,
                (args.delta if not unified_cfg else unified_cfg['loss_weights']['delta']),
                (args.zeta if not unified_cfg else unified_cfg['loss_weights']['zeta']),
                (args.eta if not unified_cfg else unified_cfg['loss_weights']['eta'])
            )
            print(f"Final Joint Loss: {float(final_joint_loss):.4f}")
    
    # 输出评估器状态摘要
    if training_evaluator:
        status = training_evaluator.get_status()
        print(f"\\n📊 Evaluation Summary:")
        print(f"  Best BLEU-4: {status['best_bleu4']:.2f}%")
        print(f"  Best BARTScore: {status['best_bart_score']:.4f}")
        print(f"  Total Samples: {status['num_eval_samples']}")
        
    summary = early_stopping.get_summary()
    print(f"\\n🛑 Final Early Stopping Summary:")
    print(f"  Mode: {summary['mode']}")
    print(f"  Best loss: {summary['best_loss']:.4f}")
    print(f"  Best BLEU-1: {summary['best_bleu1']:.2f}%")
    print(f"  Best epoch: {summary['best_epoch']}")
    print(f"  Total epochs trained: {summary['total_epochs']}")
    print(f"  Early stopped: {summary['should_stop']}")
    
    writer.close()
    return model


def train(args, eval_config):
    """原始训练函数 (向后兼容)"""
    return train_with_evaluation(args, eval_config=eval_config)
    

def main():
    ap = argparse.ArgumentParser(description="ReflectDiffu Training with Integrated Evaluation")
    ap.add_argument('--config', type=str, default=None, help='Path to unified JSON config (overrides most CLI args)')
    
    # Original training arguments
    ap.add_argument('--ec_data', type=str, default=r"dataset/emotion_labels_user_response.pkl", 
                   help='Path to emotion contagion pickle file or directory')
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--cuda', action='store_true', default=True)
    ap.add_argument('--use_era', action='store_true', default=False, 
                   help='Try to use ERA model for h_tilde')
    ap.add_argument('--delta', type=float, default=1.0, help='Weight for emotion loss')
    ap.add_argument('--zeta', type=float, default=1.0, help='Weight for intent twice loss')
    ap.add_argument('--eta', type=float, default=1.0, help='Weight for response loss')
    # Optimization hyperparameters
    ap.add_argument('--lr', type=float, default=1e-4, help='Base learning rate')
    ap.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for AdamW')
    ap.add_argument('--num_epochs', type=int, default=1000, help='Total training epochs')
    ap.add_argument('--lr_scheduler', type=str, default='none', choices=['none','linear','cosine'], help='Learning rate schedule type')
    ap.add_argument('--warmup_steps', type=int, default=0, help='Number of warmup steps (if >0 overrides warmup_ratio)')
    ap.add_argument('--warmup_ratio', type=float, default=0.06, help='Warmup ratio (fraction of total steps) if warmup_steps==0')
    
    # Evaluation configuration arguments
    ap.add_argument('--enable_eval', action='store_true', default=True,
                   help='Enable evaluation during training')
    ap.add_argument('--eval_every_epochs', type=int, default=1,
                   help='Evaluate every N epochs')
    ap.add_argument('--eval_every_steps', type=int, default=None,
                   help='Evaluate every N steps (overrides eval_every_epochs)')
    ap.add_argument('--eval_data_path', type=str, default="dataset/emotion_labels_test.pkl",
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
    ap.add_argument('--eval_results_dir', type=str, default="output",
                   help='Directory to save evaluation results')
    ap.add_argument('--eval_log_examples', action='store_true', default=True,
                   help='Log generation examples during evaluation')
    ap.add_argument('--eval_num_examples', type=int, default=5,
                   help='Number of examples to log')
    
    # Early Stopping arguments
    ap.add_argument('--enable_early_stopping', action='store_true', default=False,
                   help='Enable early stopping based on loss and BLEU-1')
    ap.add_argument('--early_stopping_patience', type=int, default=3,
                   help='Number of epochs to wait before early stopping')
    ap.add_argument('--early_stopping_min_delta', type=float, default=0.001,
                   help='Minimum improvement to reset patience')
    ap.add_argument('--early_stopping_mode', type=str, default='auto', 
                   choices=['auto', 'loss', 'bleu1'],
                   help='Early stopping mode: auto (combined), loss only, or bleu1 only')
    ap.add_argument('--early_stopping_loss_weight', type=float, default=0.7,
                   help='Weight for loss in combined early stopping mode')
    ap.add_argument('--early_stopping_bleu1_weight', type=float, default=0.3,
                   help='Weight for BLEU-1 in combined early stopping mode')
    
    args = ap.parse_args()
    unified_cfg = None
    if args.config:
        unified_cfg = load_config(args.config)
        print(f"Loaded unified config from {args.config}")
        # Derive EvaluationConfig from unified config if enabled
        if unified_cfg.get('evaluation', {}).get('enabled', True):
            eval_section = unified_cfg['evaluation']
            gen = eval_section['generation']
            metrics = eval_section['metrics']
            eval_config = EvaluationConfig(
                eval_every_epochs=eval_section.get('every_epochs', 1),
                eval_every_steps=eval_section.get('every_steps'),
                eval_at_end=True,
                eval_data_path=unified_cfg['data']['ec_data_path'],
                max_eval_samples=eval_section.get('max_eval_samples', 100),
                max_gen_length=gen.get('max_len', 32),
                use_pointer=gen.get('use_pointer', False),
                temperature=gen.get('temperature', 1.0),
                top_k=gen.get('top_k', 0),
                top_p=gen.get('top_p', 1.0),
                compute_bleu=metrics.get('bleu', True),
                compute_bart_score=metrics.get('bart_score', True),
                compute_informativeness=metrics.get('informativeness', True),
                bart_checkpoint="facebook/bart-large-cnn",
                save_results=True,
                results_dir=eval_section.get('results_dir', 'output'),
                log_examples=eval_section.get('examples', {}).get('log', True),
                num_examples=eval_section.get('examples', {}).get('num', 5),
                eval_batch_size=1,
                disable_tqdm=False
            )
        else:
            eval_config = None
        # Save effective config (currently identical) to output for record
        save_effective(unified_cfg, os.path.join('output', 'effective_config.json'))
        return train_with_evaluation(args, eval_config=eval_config, unified_cfg=unified_cfg)
    
    # Fallback: Create evaluation configuration from CLI if unified config not provided
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
    train(args, eval_config)



if __name__ == '__main__':
    main()