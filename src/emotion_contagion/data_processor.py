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
import os
import pickle
from src.tokenizer_loader import get_tokenizer
from src.era.data_processor import DialogueSample

logger = logging.getLogger(__name__)

TOKENIZER = get_tokenizer()

class RenameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "data_processor":
            module = "src.era.data_processor"
        return super().find_class(module, name)

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
        self.label_to_id = {"<noem>": 0, "<em>": 1}  # Reason label mapping
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}
        
    def load_data(self, data_path: str) -> List[Dict]:
        """
        Load data from a pickle (.pkl) file, a directory of pickle files, or (fallback) a JSON file.

        Args:
            data_path: Path to a .pkl/.json file or a directory containing .pkl files.

        Returns:
            List of data sample dicts
        """
        data: List[Dict] = []

        # Single file handling
        if data_path.endswith('.pkl'):
            with open(data_path, 'rb') as f:
                loaded = RenameUnpickler(f).load()
            if isinstance(loaded, list):
                data = loaded
            else:
                data = [loaded]
            logger.info(f"Loaded {len(data)} samples from pickle file {data_path}")
            return data

        raise ValueError(
            f"Unsupported data file type for '{data_path}'. Expected directory or .pkl file."
        )
    
    def process_sample(self, sample: DialogueSample) -> Dict:
        """
        Process a single sample to extract tokens, labels, and create masks.
        
        Args:
            sample: Raw data sample with the specified format
            
        Returns:
            Processed sample with aligned tokens/labels and attention mask
        """
        # Expect sample as list of (token,label) tuples
        user_tokens, user_labels = sample.user_tokens, sample.user_labels
        response_tokens, response_labels = sample.response_tokens, sample.response_labels

        def inner_label(word_tokens, word_labels, is_user: bool):

            if len(word_tokens) != len(word_labels):
                raise ValueError(f"Token-label mismatch: {len(word_tokens)} vs {len(word_labels)}")

            # Truncate at word level first (labels follow)
            if len(word_tokens) > self.max_length:
                word_tokens = word_tokens[:self.max_length]
                word_labels = word_labels[:self.max_length]

            # Word-level emotion label ids (no expansion to wordpieces)
            label_ids = [self.label_to_id.get(l, 0) for l in word_labels]

            # Tokenizer over the joined text; relies on HF internal wordpiece + max_length padding
            text = " ".join(word_tokens)
            encoded = TOKENIZER(
                text,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            input_ids = encoded['input_ids'][0].tolist()
            attention_mask = encoded['attention_mask'][0].tolist()

            # Pad label_ids to max_length (labels correspond to first N original words)
            if len(label_ids) < self.max_length:
                label_ids = label_ids + [0] * (self.max_length - len(label_ids))

            seq_len = min(len(word_tokens), self.max_length)

            return {
                'tokens': word_tokens[:seq_len],  # unpadded original words
                'input_ids': input_ids,            # token ids length = max_length
                'label_ids': label_ids,            # word-level labels padded to max_length
                'labels': word_labels[:seq_len],   # truncated labels (no padding strings)
                'attention_mask': attention_mask,  # tokenizer mask length = max_length
                'seq_len': seq_len,
                'user': is_user,
                'origin_prompt': text,
                'matched_emotion': sample.emotion
            }
        return inner_label(user_tokens, user_labels, True), inner_label(response_tokens, response_labels, False)
    
    def process_batch(self, samples: List[Dict], ifeval: bool=False) -> List[Dict]:
        """            
        Returns:
            List of processed samples
        """
        user_batch, response_batch = list(), list()
        
        for idx, sample in enumerate(samples):
            user_process, response_process = self.process_sample(sample)
            user_batch.append(user_process)
            response_batch.append(response_process)

        return user_batch, response_batch
    
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
        batch_tokens = [s['tokens'] for s in batch]
        batch_input_ids = torch.tensor([s['input_ids'] for s in batch], dtype=torch.long)
        batch_label_ids = torch.tensor([s['label_ids'] for s in batch], dtype=torch.long)
        batch_attention_mask = torch.tensor([s['attention_mask'] for s in batch], dtype=torch.long)
        batch_seq_len = torch.tensor([s['seq_len'] for s in batch], dtype=torch.long)

        return {
            'tokens': batch_tokens,
            'input_ids': batch_input_ids,
            'label_ids': batch_label_ids,
            'attention_mask': batch_attention_mask,
            'seq_len': batch_seq_len,
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