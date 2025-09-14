"""
Data Processor for Emotion-Contagion Encoder

Simple data processing for the specified format:
[
    {
        user: boolean,
        origin_prompt: str
        sentence1: [
            [token1, token2, ...tokenn],
            [label1, label2, ...labeln]
        ]
    },
    ...
]

Handles token-label alignment and creates attention masks.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional, Union
import json
import logging

logger = logging.getLogger(__name__)


class EmotionContagionDataProcessor:
    """
    Simple data processor for emotion-contagion encoder.
    
    Handles the specified data format and creates aligned tokens/labels
    with proper attention masks for the encoder.
    """
    
    def __init__(self, max_length: int = 512):
        """
        Initialize data processor.
        
        Args:
            max_length: Maximum sequence length for padding/truncation
        """
        self.max_length = max_length
        self.label_to_id = {"noem": 0, "em": 1}  # Reason label mapping
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}
        
    def load_data(self, data_path: str) -> List[Dict]:
        """
        Load data from JSON file.
        
        Args:
            data_path: Path to JSON file containing the data
            
        Returns:
            List of data samples
        """
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Loaded {len(data)} samples from {data_path}")
        return data
    
    def process_sample(self, sample: Dict) -> Dict:
        """
        Process a single sample to extract tokens, labels, and create masks.
        
        Args:
            sample: Raw data sample with the specified format
            
        Returns:
            Processed sample with aligned tokens/labels and attention mask
        """
        # Extract tokens and labels from sentence1
        tokens = sample["sentence1"][0]  # [token1, token2, ..., tokenn]
        labels = sample["sentence1"][1]  # [label1, label2, ..., labeln]
        
        # Validate alignment
        if len(tokens) != len(labels):
            raise ValueError(
                f"Token-label mismatch: {len(tokens)} tokens vs {len(labels)} labels"
            )
        
        # Convert labels to IDs
        label_ids = [self.label_to_id.get(label, 0) for label in labels]
        
        # Truncate if necessary
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
            label_ids = label_ids[:self.max_length]
            labels = labels[:self.max_length]
        
        # Create attention mask (1 for valid tokens, 0 for padding)
        seq_len = len(tokens)
        attention_mask = [1] * seq_len + [0] * (self.max_length - seq_len)
        
        # Pad tokens and labels to max_length
        padded_tokens = tokens + ["[PAD]"] * (self.max_length - seq_len)
        padded_label_ids = label_ids + [0] * (self.max_length - seq_len)  # 0 = noem for padding
        padded_labels = labels + ["noem"] * (self.max_length - seq_len)
        
        return {
            "tokens": padded_tokens,
            "label_ids": padded_label_ids,
            "labels": padded_labels,
            "attention_mask": attention_mask,
            "seq_len": seq_len,
            "user": sample.get("user", False),
            "origin_prompt": sample.get("origin_prompt", "")
        }
    
    def process_batch(self, samples: List[Dict]) -> List[Dict]:
        """
        Process a batch of samples.
        
        Args:
            samples: List of raw data samples
            
        Returns:
            List of processed samples
        """
        processed_samples = []
        
        for sample in samples:
            try:
                processed = self.process_sample(sample)
                processed_samples.append(processed)
            except Exception as e:
                logger.warning(f"Failed to process sample: {e}")
                continue
                
        return processed_samples
    
    def create_dataset(self, data: List[Dict]) -> "EmotionContagionDataset":
        """
        Create a PyTorch dataset from processed data.
        
        Args:
            data: List of raw data samples
            
        Returns:
            EmotionContagionDataset instance
        """
        processed_data = self.process_batch(data)
        return EmotionContagionDataset(processed_data)
    
    def create_dataloader(
        self, 
        dataset: "EmotionContagionDataset", 
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0
    ) -> DataLoader:
        """
        Create a PyTorch DataLoader.
        
        Args:
            dataset: EmotionContagionDataset instance
            batch_size: Batch size for training
            shuffle: Whether to shuffle the data
            num_workers: Number of worker processes
            
        Returns:
            PyTorch DataLoader
        """
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate function for DataLoader to create batched tensors.
        
        Args:
            batch: List of processed samples
            
        Returns:
            Batched tensors
        """
        # Stack all sequences
        batch_tokens = [sample["tokens"] for sample in batch]
        batch_label_ids = torch.tensor([sample["label_ids"] for sample in batch], dtype=torch.long)
        batch_attention_mask = torch.tensor([sample["attention_mask"] for sample in batch], dtype=torch.long)
        batch_seq_len = torch.tensor([sample["seq_len"] for sample in batch], dtype=torch.long)
        
        return {
            "tokens": batch_tokens,  # Keep as list of strings for word embedding lookup
            "label_ids": batch_label_ids,  # [B, L]
            "attention_mask": batch_attention_mask,  # [B, L]
            "seq_len": batch_seq_len,  # [B]
        }


class EmotionContagionDataset(Dataset):
    """
    PyTorch Dataset for emotion-contagion encoder.
    """
    
    def __init__(self, processed_data: List[Dict]):
        """
        Initialize dataset.
        
        Args:
            processed_data: List of processed samples from EmotionContagionDataProcessor
        """
        self.data = processed_data
        
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        return self.data[idx]