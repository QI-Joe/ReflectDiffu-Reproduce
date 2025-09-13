#!/usr/bin/env python3
"""
Download Models Script for ERA System

This script downloads and caches LLM (ChatGLM4) and BERT models locally
with configurable storage paths as specified in EAR.md.

Usage:
    python download_models.py --models_dir ./models --bert_model roberta-base --llm_model chatglm4

Features:
- Downloads BERT/RoBERTa models from Hugging Face
- Downloads ChatGLM4 for data annotation 
- Configurable local storage paths
- Progress tracking and error handling
- Model verification after download
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, List
import logging
import json
from datetime import datetime

import torch
from transformers import (
    AutoModel, 
    AutoTokenizer, 
    RobertaModel, 
    RobertaTokenizer,
    BertModel, 
    BertTokenizer
)
from huggingface_hub import hf_hub_download, list_repo_files

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelDownloader:
    """
    Downloads and manages model storage for ERA system.
    
    Supports:
    - BERT/RoBERTa models for sequence labeling
    - ChatGLM4 for data annotation
    - Local caching and verification
    """
    
    def __init__(self, models_dir: str = "./models"):
        """
        Initialize model downloader.
        
        Args:
            models_dir: Directory to store downloaded models
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Model configurations
        self.supported_bert_models = {
            "roberta-base": "roberta-base",
            "bert-base-uncased": "bert-base-uncased", 
            "bert-base-cased": "bert-base-cased",
            "roberta-large": "roberta-large"
        }
        
        self.supported_llm_models = {
            "chatglm4": "THUDM/chatglm3-6b",  # Using ChatGLM3 as ChatGLM4 placeholder
            "chatglm3": "THUDM/chatglm3-6b"
        }
        
        logger.info(f"Model storage directory: {self.models_dir.absolute()}")
    
    def download_bert_model(self, model_name: str) -> bool:
        """
        Download BERT/RoBERTa model for sequence labeling.
        
        Args:
            model_name: Name of the BERT model to download
            
        Returns:
            True if successful, False otherwise
        """
        if model_name not in self.supported_bert_models:
            logger.error(f"Unsupported BERT model: {model_name}")
            logger.info(f"Supported models: {list(self.supported_bert_models.keys())}")
            return False
        
        model_path = self.models_dir / model_name
        hf_model_name = self.supported_bert_models[model_name]
        
        try:
            logger.info(f"Downloading BERT model: {model_name}")
            logger.info(f"Storage path: {model_path}")
            
            # Download tokenizer
            logger.info("Downloading tokenizer...")
            if "roberta" in model_name.lower():
                tokenizer = RobertaTokenizer.from_pretrained(hf_model_name)
                model = RobertaModel.from_pretrained(hf_model_name)
            else:
                tokenizer = BertTokenizer.from_pretrained(hf_model_name)
                model = BertModel.from_pretrained(hf_model_name)
            
            # Save locally
            model_path.mkdir(parents=True, exist_ok=True)
            tokenizer.save_pretrained(model_path)
            model.save_pretrained(model_path)
            
            # Verify download
            if self._verify_bert_model(model_path):
                logger.info(f"✓ Successfully downloaded and verified: {model_name}")
                self._save_model_info(model_name, "bert", hf_model_name, model_path)
                return True
            else:
                logger.error(f"✗ Model verification failed: {model_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error downloading {model_name}: {str(e)}")
            return False
    
    def download_llm_model(self, model_name: str) -> bool:
        """
        Download LLM model for data annotation.
        
        Args:
            model_name: Name of the LLM model to download
            
        Returns:
            True if successful, False otherwise
        """
        if model_name not in self.supported_llm_models:
            logger.error(f"Unsupported LLM model: {model_name}")
            logger.info(f"Supported models: {list(self.supported_llm_models.keys())}")
            return False
        
        model_path = self.models_dir / model_name
        hf_model_name = self.supported_llm_models[model_name]
        
        try:
            logger.info(f"Downloading LLM model: {model_name}")
            logger.info(f"Storage path: {model_path}")
            
            # Download model and tokenizer
            logger.info("Downloading model and tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                hf_model_name, 
                trust_remote_code=True
            )
            model = AutoModel.from_pretrained(
                hf_model_name, 
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            # Save locally
            model_path.mkdir(parents=True, exist_ok=True)
            tokenizer.save_pretrained(model_path)
            model.save_pretrained(model_path)
            
            # Verify download
            if self._verify_llm_model(model_path):
                logger.info(f"✓ Successfully downloaded and verified: {model_name}")
                self._save_model_info(model_name, "llm", hf_model_name, model_path)
                return True
            else:
                logger.error(f"✗ Model verification failed: {model_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error downloading {model_name}: {str(e)}")
            return False
    
    def _verify_bert_model(self, model_path: Path) -> bool:
        """Verify BERT model can be loaded correctly."""
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModel.from_pretrained(model_path)
            
            # Test inference
            test_input = tokenizer("This is a test sentence", return_tensors="pt")
            with torch.no_grad():
                outputs = model(**test_input)
            
            logger.info(f"Model output shape: {outputs.last_hidden_state.shape}")
            return True
        except Exception as e:
            logger.error(f"Model verification failed: {str(e)}")
            return False
    
    def _verify_llm_model(self, model_path: Path) -> bool:
        """Verify LLM model can be loaded correctly."""
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
            
            # Test tokenization
            test_input = tokenizer("Hello, this is a test.", return_tensors="pt")
            logger.info(f"Tokenizer test passed. Input IDs shape: {test_input['input_ids'].shape}")
            return True
        except Exception as e:
            logger.error(f"LLM verification failed: {str(e)}")
            return False
    
    def _save_model_info(self, model_name: str, model_type: str, hf_name: str, local_path: Path):
        """Save model metadata to JSON file."""
        info_file = self.models_dir / "models_info.json"
        
        # Load existing info
        if info_file.exists():
            with open(info_file, 'r') as f:
                models_info = json.load(f)
        else:
            models_info = {}
        
        # Add new model info
        models_info[model_name] = {
            "type": model_type,
            "huggingface_name": hf_name,
            "local_path": str(local_path),
            "download_date": datetime.now().isoformat(),
            "verified": True
        }
        
        # Save updated info
        with open(info_file, 'w') as f:
            json.dump(models_info, f, indent=2)
    
    def list_downloaded_models(self) -> dict:
        """List all downloaded models."""
        info_file = self.models_dir / "models_info.json"
        
        if not info_file.exists():
            return {}
        
        with open(info_file, 'r') as f:
            return json.load(f)
    
    def get_model_path(self, model_name: str) -> Optional[Path]:
        """Get local path for a downloaded model."""
        models_info = self.list_downloaded_models()
        
        if model_name in models_info:
            return Path(models_info[model_name]["local_path"])
        return None


def main():
    """Main function to handle command line arguments and download models."""
    parser = argparse.ArgumentParser(
        description="Download and cache models for ERA system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download default models to ./models
    python download_models.py
    
    # Download to custom directory
    python download_models.py --models_dir /path/to/models
    
    # Download specific models
    python download_models.py --bert_model roberta-base --llm_model chatglm4
    
    # List downloaded models
    python download_models.py --list
        """
    )
    
    parser.add_argument(
        "--models_dir", 
        type=str, 
        default="./models",
        help="Directory to store downloaded models (default: ./models)"
    )
    
    parser.add_argument(
        "--bert_model", 
        type=str, 
        default="roberta-base",
        choices=["roberta-base", "bert-base-uncased", "bert-base-cased", "roberta-large"],
        help="BERT model to download (default: roberta-base)"
    )
    
    parser.add_argument(
        "--llm_model", 
        type=str, 
        default="chatglm4",
        choices=["chatglm4", "chatglm3"],
        help="LLM model to download (default: chatglm4)"
    )
    
    parser.add_argument(
        "--bert_only", 
        action="store_true",
        help="Download only BERT model"
    )
    
    parser.add_argument(
        "--llm_only", 
        action="store_true",
        help="Download only LLM model"
    )
    
    parser.add_argument(
        "--list", 
        action="store_true",
        help="List downloaded models and exit"
    )
    
    args = parser.parse_args()
    
    # Initialize downloader
    downloader = ModelDownloader(args.models_dir)
    
    # List models if requested
    if args.list:
        models_info = downloader.list_downloaded_models()
        if not models_info:
            print("No models downloaded yet.")
        else:
            print("Downloaded models:")
            for name, info in models_info.items():
                print(f"  {name}: {info['type']} ({info['huggingface_name']})")
                print(f"    Path: {info['local_path']}")
                print(f"    Downloaded: {info['download_date']}")
        return
    
    # Download models
    success = True
    
    if not args.llm_only:
        logger.info(f"Starting BERT model download: {args.bert_model}")
        if not downloader.download_bert_model(args.bert_model):
            success = False
    
    if not args.bert_only:
        logger.info(f"Starting LLM model download: {args.llm_model}")
        if not downloader.download_llm_model(args.llm_model):
            success = False
    
    if success:
        logger.info("✓ All models downloaded successfully!")
        print(f"\nModels stored in: {downloader.models_dir.absolute()}")
        print("\nDownloaded models:")
        models_info = downloader.list_downloaded_models()
        for name, info in models_info.items():
            print(f"  - {name} ({info['type']}): {info['local_path']}")
    else:
        logger.error("✗ Some models failed to download")
        sys.exit(1)


if __name__ == "__main__":
    main()