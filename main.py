#!/usr/bin/env python3
"""
Main Training Script for ERA System

Orchestrates the entire pipeline:
- Data loading and processing
- Model initialization
- Training with monitoring
- Evaluation and result reporting

Usage:
    python main.py --config configs/nuner_baseline.json
    python main.py --bert_model roberta-base --batch_size 32 --num_epochs 5
    python main.py --help

Features:
- Command line argument parsing
- Configuration management
- Automatic data preprocessing with caching
- Model training with checkpointing
- Comprehensive evaluation
- Result reporting and visualization
"""

import os
import sys
import argparse
import logging
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import ERA modules
from src.era.config import ERAConfig, ERAConfigPresets
from src.era.data_processor import EmpathyDataProcessor
from src.era.era_model import create_era_model
from src.era.trainer import ERATrainer
from src.era.evaluator import ERAEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('era_training.log')
    ]
)
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Train ERA (Emotion Reason Annotator) model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train with default NuNER configuration
    python main.py
    
    # Train with custom configuration file
    python main.py --config configs/custom.json
    
    # Train with custom parameters
    python main.py --bert_model roberta-base --batch_size 32 --num_epochs 5
    
    # Debug mode with small dataset
    python main.py --debug --max_train_samples 100
    
    # Resume training from checkpoint
    python main.py --resume_from checkpoints/checkpoint-epoch-3
    
    # Evaluate only (no training)
    python main.py --eval_only --model_path checkpoints/best_model
        """
    )
    
    # Configuration
    parser.add_argument(
        "--config", 
        type=str, 
        help="Path to JSON configuration file"
    )
    parser.add_argument(
        "--preset", 
        type=str, 
        choices=["nuner_baseline", "fast_debug", "small_gpu", "io_scheme"],
        default="nuner_baseline",
        help="Use predefined configuration preset"
    )
    
    # Data arguments
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="./dataset",
        help="Directory containing EmpatheticDialogues data"
    )
    parser.add_argument(
        "--models_dir", 
        type=str, 
        default="./models",
        help="Directory containing downloaded models"
    )
    parser.add_argument(
        "--cache_dir", 
        type=str, 
        default="./cache",
        help="Directory for caching processed data"
    )
    
    # Model arguments
    parser.add_argument(
        "--bert_model", 
        type=str,
        help="BERT model name or path (e.g., roberta-base)"
    )
    parser.add_argument(
        "--use_crf", 
        action="store_true",
        help="Use CRF layer (default: True for preset configs)"
    )
    parser.add_argument(
        "--no_crf", 
        action="store_true",
        help="Disable CRF layer"
    )
    parser.add_argument(
        "--tagging_scheme", 
        type=str, 
        choices=["BIO", "IO"],
        help="Tagging scheme to use"
    )
    parser.add_argument(
        "--frozen_layers", 
        type=int,
        help="Number of BERT layers to freeze"
    )
    
    # Training arguments
    parser.add_argument(
        "--batch_size", 
        type=int,
        help="Training batch size"
    )
    parser.add_argument(
        "--learning_rate", 
        type=float,
        help="Learning rate"
    )
    parser.add_argument(
        "--num_epochs", 
        type=int,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--warmup_ratio", 
        type=float,
        help="Warmup ratio for learning rate scheduler"
    )
    parser.add_argument(
        "--weight_decay", 
        type=float,
        help="Weight decay for optimizer"
    )
    
    # Output arguments
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./checkpoints",
        help="Directory to save model checkpoints"
    )
    parser.add_argument(
        "--logging_dir", 
        type=str, 
        default="./logs",
        help="Directory for logging (TensorBoard)"
    )
    parser.add_argument(
        "--run_name", 
        type=str,
        help="Name for this training run"
    )
    
    # Execution mode arguments
    parser.add_argument(
        "--eval_only", 
        action="store_true",
        help="Only evaluate, do not train"
    )
    parser.add_argument(
        "--model_path", 
        type=str,
        help="Path to model for evaluation (when using --eval_only)"
    )
    parser.add_argument(
        "--resume_from", 
        type=str,
        help="Resume training from checkpoint directory"
    )
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug mode with smaller dataset"
    )
    
    # Data processing arguments
    parser.add_argument(
        "--max_train_samples", 
        type=int,
        help="Maximum number of training samples (for debugging)"
    )
    parser.add_argument(
        "--max_eval_samples", 
        type=int,
        help="Maximum number of evaluation samples (for debugging)"
    )
    parser.add_argument(
        "--force_reprocess", 
        action="store_true",
        help="Force reprocessing of data (ignore cache)"
    )
    
    # Hardware arguments
    parser.add_argument(
        "--device", 
        type=str, 
        default="auto",
        help="Device to use (auto, cpu, cuda, cuda:0, etc.)"
    )
    parser.add_argument(
        "--fp16", 
        action="store_true",
        help="Use mixed precision training"
    )
    
    return parser.parse_args()


def create_config_from_args(args: argparse.Namespace) -> ERAConfig:
    """
    Create ERA configuration from command line arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        ERA configuration object
    """
    # Start with preset configuration
    if args.config:
        # Load from JSON file
        config = ERAConfig.from_json(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    else:
        # Use preset
        preset_map = {
            "nuner_baseline": ERAConfigPresets.nuner_baseline,
            "fast_debug": ERAConfigPresets.fast_debug,
            "small_gpu": ERAConfigPresets.small_gpu,
            "io_scheme": ERAConfigPresets.io_scheme
        }
        config = preset_map[args.preset]()
        logger.info(f"Using preset configuration: {args.preset}")
    
    # Override with command line arguments
    override_dict = {}
    
    # Data arguments
    if args.data_dir:
        override_dict["data_dir"] = args.data_dir
    if args.cache_dir:
        override_dict["cache_dir"] = args.cache_dir
    
    # Model arguments
    if args.bert_model:
        override_dict["bert_model"] = args.bert_model
    if args.use_crf:
        override_dict["use_crf"] = True
    if args.no_crf:
        override_dict["use_crf"] = False
    if args.tagging_scheme:
        override_dict["tagging_scheme"] = args.tagging_scheme
    if args.frozen_layers is not None:
        override_dict["frozen_layers"] = args.frozen_layers
    
    # Training arguments
    if args.batch_size:
        override_dict["batch_size"] = args.batch_size
    if args.learning_rate:
        override_dict["learning_rate"] = args.learning_rate
        override_dict["encoder_lr"] = args.learning_rate
        override_dict["head_lr"] = args.learning_rate * 1.5  # Slightly higher for head
    if args.num_epochs:
        override_dict["num_epochs"] = args.num_epochs
    if args.warmup_ratio:
        override_dict["warmup_ratio"] = args.warmup_ratio
    if args.weight_decay:
        override_dict["weight_decay"] = args.weight_decay
    
    # Output arguments
    if args.output_dir:
        override_dict["output_dir"] = args.output_dir
    if args.logging_dir:
        override_dict["logging_dir"] = args.logging_dir
    if args.run_name:
        override_dict["run_name"] = args.run_name
    
    # Debug mode
    if args.debug:
        override_dict["debug_mode"] = True
        override_dict["batch_size"] = 4
        override_dict["num_epochs"] = 2
        override_dict["logging_steps"] = 5
        override_dict["eval_steps"] = 10
        if args.max_train_samples is None:
            override_dict["max_train_samples"] = 100
        if args.max_eval_samples is None:
            override_dict["max_eval_samples"] = 50
    
    # Hardware arguments
    if args.device:
        override_dict["device"] = args.device
    if args.fp16:
        override_dict["fp16"] = args.fp16
    
    # Data processing arguments
    if args.max_train_samples:
        override_dict["max_train_samples"] = args.max_train_samples
    if args.max_eval_samples:
        override_dict["max_eval_samples"] = args.max_eval_samples
    
    # Update configuration
    config.update_from_dict(override_dict)
    
    return config


def setup_data_processor(config: ERAConfig, args: argparse.Namespace) -> EmpathyDataProcessor:
    """
    Setup data processor with configuration.
    
    Args:
        config: ERA configuration
        args: Command line arguments
        
    Returns:
        Configured data processor
    """
    # Determine LLM model path
    llm_model_path = None
    if hasattr(config, 'llm_model_path') and config.llm_model_path:
        llm_model_path = config.llm_model_path
    elif args.models_dir:
        # Try to find ChatGLM model in models directory
        models_dir = Path(args.models_dir)
        for model_name in ["chatglm4", "chatglm3"]:
            model_path = models_dir / model_name
            if model_path.exists():
                llm_model_path = str(model_path)
                logger.info(f"Found LLM model: {llm_model_path}")
                break
    
    # Clear cache if requested
    if args.force_reprocess:
        cache_dir = Path(config.cache_dir)
        if cache_dir.exists():
            import shutil
            shutil.rmtree(cache_dir)
            logger.info("Cleared data cache")
    
    # Create data processor
    processor = EmpathyDataProcessor(
        data_dir=config.data_dir,
        llm_model_path=llm_model_path,
        tokenizer_model=config.bert_model,
        max_length=config.max_length,
        tagging_scheme=config.tagging_scheme,
        cache_dir=config.cache_dir
    )
    
    return processor


def main():
    """Main training function."""
    logger.info("Starting ERA training pipeline...")
    
    # Parse arguments
    args = parse_arguments()
    
    # Create configuration
    config = create_config_from_args(args)
    
    # Print configuration
    config.print_config()
    
    # Save configuration
    config_file = Path(config.output_dir) / "config.json"
    config.to_json(str(config_file))
    logger.info(f"Configuration saved to: {config_file}")
    
    try:
        # Setup data processor
        logger.info("Setting up data processor...")
        data_processor = setup_data_processor(config, args)
        
        if not args.eval_only:
            # Process data and create dataloaders
            logger.info("Processing data...")
            train_loader, valid_loader, test_loader = data_processor.process_full_pipeline()
            
            logger.info(f"Data processing completed:")
            logger.info(f"  Train batches: {len(train_loader)}")
            logger.info(f"  Valid batches: {len(valid_loader)}")
            logger.info(f"  Test batches: {len(test_loader)}")
            
            # Create model
            logger.info("Creating ERA model...")
            model = create_era_model(
                bert_model=config.bert_model,
                num_labels=config.num_labels,
                use_crf=config.use_crf,
                frozen_layers=config.frozen_layers,
                dropout_rate=config.dropout_rate,
                device=config.device
            )
            
            # Create evaluator
            evaluator = ERAEvaluator(config)
            
            # Create trainer
            logger.info("Setting up trainer...")
            trainer = ERATrainer(
                model=model,
                config=config,
                train_dataloader=train_loader,
                eval_dataloader=valid_loader,
                test_dataloader=test_loader,
                evaluator=evaluator
            )
            
            # Resume training if requested
            if args.resume_from:
                trainer.resume_training(args.resume_from)
            
            # Start training
            logger.info("Starting training...")
            start_time = time.time()
            
            training_results = trainer.train()
            
            end_time = time.time()
            training_time = end_time - start_time
            
            # Log training results
            logger.info("Training completed!")
            logger.info(f"  Training time: {training_time:.2f} seconds")
            logger.info(f"  Best epoch: {training_results['best_epoch']}")
            logger.info(f"  Best metric: {training_results['best_metric']:.4f}")
            
            # Cleanup
            trainer.cleanup()
            
        else:
            # Evaluation only mode
            if not args.model_path:
                raise ValueError("--model_path is required when using --eval_only")
            
            logger.info(f"Loading model from {args.model_path} for evaluation...")
            
            # Load model
            from src.era.era_model import ERA_BERT_CRF
            model = ERA_BERT_CRF.from_pretrained(args.model_path)
            
            # Process test data only
            dialogues = data_processor.load_empathetic_dialogues()
            annotated_dialogues = data_processor.annotate_with_llm(dialogues)
            tokenized_samples = data_processor.tokenize_and_align_labels(annotated_dialogues)
            
            # Create test dataloader
            test_loader = data_processor.create_dataloader(tokenized_samples, shuffle=False)
            
            # Create evaluator and evaluate
            evaluator = ERAEvaluator(config)
            
            # Get predictions
            model.eval()
            all_predictions = []
            all_labels = []
            
            import torch
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            
            with torch.no_grad():
                for batch in test_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = model(**batch)
                    predictions = outputs["predictions"].cpu().numpy()
                    labels = batch["labels"].cpu().numpy()
                    
                    all_predictions.extend(predictions)
                    all_labels.extend(labels)
            
            # Compute and print metrics
            evaluator.print_evaluation_report(all_predictions, all_labels)
            
            # Save results
            results_file = Path(config.output_dir) / "evaluation_results.json"
            evaluator.save_evaluation_results(all_predictions, all_labels, str(results_file))
            
        logger.info("ERA pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in ERA pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()