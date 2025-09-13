"""
Data Processor for ERA System

Handles EmpatheticDialogues loading, ChatGLM4 annotation, 
token-label alignment, and train/valid/test splitting (8:1:1).

Based on specifications in EAR.md.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import logging
from dataclasses import dataclass
import pickle
import random
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)

@dataclass
class DialogueSample:
    """
    Represents a single dialogue sample with emotion reason annotations.
    """
    conv_id: str
    utterance_id: int
    utterance: str
    speaker_id: int
    emotion: str
    prompt: str
    tokens: List[str]
    labels: List[str]  # <em> or <noem>
    selfeval: list[list[int]]
    

@dataclass
class TokenizedSample:
    """
    Represents a tokenized sample ready for model training.
    """
    input_ids: List[int]
    attention_mask: List[int]
    labels: List[int]  # 0 for <noem>, 1 for <em>, -100 for ignore
    token_type_ids: Optional[List[int]] = None


class EmpathyDataProcessor:
    """
    Processes EmpatheticDialogues dataset for ERA training.
    
    Features:
    - Loads EmpatheticDialogues from CSV files
    - Uses LLM (ChatGLM4) for emotion reason annotation
    - Handles tokenization and label alignment
    - Supports BIO and IO tagging schemes
    - Implements 8:1:1 train/valid/test split
    """
    
    def __init__(
        self,
        data_dir: str,
        llm_model_path: str = None,
        tokenizer_model: str = "bert-base",
        max_length: int = 512,
        tagging_scheme: str = "IO",  # "BIO" or "IO"
        cache_dir: str = "./cache"
    ):
        """
        Initialize data processor.
        
        Args:
            data_dir: Directory containing EmpatheticDialogues CSV files
            llm_model_path: Path to local LLM model for annotation
            tokenizer_model: Model name or path for tokenizer
            max_length: Maximum sequence length
            tagging_scheme: Either "BIO" or "IO" for label encoding
            cache_dir: Directory for caching processed data
        """
        self.data_dir = Path(data_dir)
        self.llm_model_path = llm_model_path
        self.tokenizer_model = tokenizer_model
        self.max_length = max_length
        self.tagging_scheme = tagging_scheme
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
        
        # Label mappings
        if tagging_scheme == "BIO":
            self.label_to_id = {"O": 0, "B-EM": 1, "I-EM": 2}
            self.id_to_label = {0: "O", 1: "B-EM", 2: "I-EM"}
            self.num_labels = 3
        else:  # IO scheme
            self.label_to_id = {"O": 0, "EM": 1}
            self.id_to_label = {0: "O", 1: "EM"}
            self.num_labels = 2
        
        # Initialize LLM for annotation (if provided)
        self.llm_model = None
        self.llm_tokenizer = None
        if llm_model_path and os.path.exists(llm_model_path):
            self._load_llm_model()
    
    def _load_llm_model(self):
        """Load LLM model for data annotation."""
        try:
            logger.info(f"Loading LLM model from: {self.llm_model_path}")
            self.llm_tokenizer = AutoTokenizer.from_pretrained(
                self.llm_model_path, trust_remote_code=True
            )
            self.llm_model = AutoModel.from_pretrained(
                self.llm_model_path, 
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            if torch.cuda.is_available():
                self.llm_model = self.llm_model.cuda()
            logger.info("✓ LLM model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load LLM model: {e}")
            self.llm_model = None
            self.llm_tokenizer = None
    
    def load_empathetic_dialogues(self) -> List[DialogueSample]:
        """
        Load EmpatheticDialogues dataset from CSV files.
        
        Returns:
            List of DialogueSample objects
        """
        # Check cache first
        cache_file = self.cache_dir / "raw_dialogues.pkl"
        if cache_file.exists():
            logger.info("Loading dialogues from cache...")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        logger.info("Loading EmpatheticDialogues dataset...")
        
        dialogues = []
        
        # Load train, valid, test splits
        for split in ['train']: # , 'valid', 'test'
            csv_file = self.data_dir / f"{split}.csv"
            if not csv_file.exists():
                logger.warning(f"File not found: {csv_file}")
                continue
            
            logger.info(f"Loading {split} data...")
            df = pd.read_csv(csv_file)
            
            for idx, row in df.iterrows():
                # Extract dialogue information
                conv_id = str(row.get('conv_id', ''))
                utterance = str(row.get('utterance', ''))
                utterance_idx = int(row.get('utterance_idx', -1))
                speaker_idx = int(row.get('speaker_idx', -1))
                emotion = str(row.get('context', ''))  # emotion context
                prompt = str(row.get('prompt', ''))
                selfeval = str(row.get('selfeval', ''))
                selfeval = [list(map(int, eval.split("|"))) for eval in selfeval.split("_")]
                
                if utterance and len(utterance.strip()) > 0:
                    # Create sample without labels initially
                    sample = DialogueSample(
                        dialogue_id=conv_id,
                        speaker_id=speaker_idx,
                        utterance=utterance,
                        utterance_id=utterance_idx,
                        emotion=emotion,
                        tokens=[],  # Will be filled by tokenization
                        labels=[],  # Will be filled by annotation
                        prompt=prompt,
                        selfeval=selfeval
                    )
                    dialogues.append(sample)
        
        logger.info(f"Loaded {len(dialogues)} dialogue samples")
        
        # Cache raw dialogues
        with open(cache_file, 'wb') as f:
            pickle.dump(dialogues, f)
        
        return dialogues
    
    def annotate_with_llm(self, dialogues: List[DialogueSample]) -> List[DialogueSample]:
        """
        Annotate dialogues with emotion reason labels using LLM.
        
        Args:
            dialogues: List of DialogueSample objects
            
        Returns:
            List of annotated DialogueSample objects
        """
        # Check cache first
        cache_file = self.cache_dir / f"annotated_dialogues_{self.tagging_scheme}.pkl"
        if cache_file.exists():
            logger.info("Loading annotated dialogues from cache...")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        logger.info("Annotating dialogues with LLM...")
        
        annotated_dialogues = []
        
        for i, dialogue in enumerate(dialogues):
            if i % 100 == 0:
                logger.info(f"Annotating dialogue {i+1}/{len(dialogues)}")
            
            try:
                # Get LLM annotation
                tokens, labels = self._get_llm_annotation(dialogue)
                
                # Update dialogue sample
                dialogue.tokens = tokens
                dialogue.labels = labels
                annotated_dialogues.append(dialogue)
                
            except Exception as e:
                logger.warning(f"Failed to annotate dialogue {dialogue.dialogue_id}: {e}")
                continue
        
        # Cache annotated dialogues
        with open(cache_file, 'wb') as f:
            pickle.dump(annotated_dialogues, f)
        
        logger.info(f"Annotated {len(annotated_dialogues)} dialogues")
        return annotated_dialogues
    
    def _get_llm_annotation(self, dialogue: DialogueSample) -> Tuple[List[str], List[str]]:
        """
        Get emotion reason annotation from LLM.
        
        Args:
            dialogue: DialogueSample to annotate
            
        Returns:
            Tuple of (tokens, labels)
        """
        # Create prompt for LLM
        prompt = self._create_annotation_prompt(dialogue)
        
        # Get LLM response
        with torch.no_grad():
            response = self._query_llm(prompt)
        
        # Parse response to extract tokens and labels
        tokens, labels = self._parse_llm_response(response, dialogue.utterance)
        
        return tokens, labels
    
    def _create_annotation_prompt(self, dialogue: DialogueSample) -> str:
        """
        Create prompt for LLM annotation.
        
        Args:
            dialogue: DialogueSample to create prompt for
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""
Please analyze the following dialogue and identify words that represent emotion reasons.
Mark each word with <em> if it's an emotion reason word, or <noem> if it's not.

Dialogue Context:
Emotion: {dialogue.emotion}
Situation: {dialogue.situation}
Speaker: {dialogue.speaker}

Utterance: "{dialogue.utterance}"

Please provide the output in the following format:
word1:<em/noem> word2:<em/noem> ...

Example:
Input: "I am very sad because I lost my dog"
Output: I:<noem> am:<noem> very:<noem> sad:<em> because:<noem> I:<noem> lost:<em> my:<noem> dog:<noem>

Your output:
"""
        return prompt
    
    def _query_llm(self, prompt: str) -> str:
        """
        Query LLM with prompt.
        
        Args:
            prompt: Input prompt
            
        Returns:
            LLM response
        """
        inputs = self.llm_tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.llm_model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.llm_tokenizer.eos_token_id
            )
        
        response = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part (after the prompt)
        response = response[len(prompt):].strip()
        
        return response
    
    def _parse_llm_response(self, response: str, original_text: str) -> Tuple[List[str], List[str]]:
        """
        Parse LLM response to extract tokens and labels.
        
        Args:
            response: LLM response string
            original_text: Original utterance text
            
        Returns:
            Tuple of (tokens, labels)
        """
        tokens = []
        labels = []
        
        try:
            # Parse format: word1:<em/noem> word2:<em/noem> ...
            parts = response.split()
            for part in parts:
                if ':' in part:
                    word, label = part.split(':', 1)
                    tokens.append(word)
                    
                    # Convert to standard label format
                    if '<em>' in label or 'em>' in label:
                        labels.append('<em>')
                    else:
                        labels.append('<noem>')
        except Exception as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            # Fall back to simple tokenization
            tokens = original_text.split()
            labels = ['<noem>'] * len(tokens)
        
        # Ensure tokens and labels have same length
        if len(tokens) != len(labels):
            logger.warning("Token-label length mismatch, using simple tokenization")
            tokens = original_text.split()
            labels = ['<noem>'] * len(tokens)
        
        return tokens, labels
    
    def convert_to_bio_labels(self, tokens: List[str], labels: List[str]) -> List[str]:
        """
        Convert IO labels to BIO format.
        
        Args:
            tokens: List of tokens
            labels: List of IO labels (<em>/<noem>)
            
        Returns:
            List of BIO labels
        """
        bio_labels = []
        in_entity = False
        
        for label in labels:
            if label == '<em>':
                if not in_entity:
                    bio_labels.append('B-EM')
                    in_entity = True
                else:
                    bio_labels.append('I-EM')
            else:  # <noem>
                bio_labels.append('O')
                in_entity = False
        
        return bio_labels
    
    def tokenize_and_align_labels(self, dialogues: List[DialogueSample]) -> List[TokenizedSample]:
        """
        Tokenize samples and align labels with subword tokens.
        
        Args:
            dialogues: List of annotated DialogueSample objects
            
        Returns:
            List of TokenizedSample objects
        """
        logger.info("Tokenizing and aligning labels...")
        
        tokenized_samples = []
        
        for dialogue in dialogues:
            # Convert labels to appropriate scheme
            if self.tagging_scheme == "BIO":
                scheme_labels = self.convert_to_bio_labels(dialogue.tokens, dialogue.labels)
            else:  # IO
                scheme_labels = [label.replace('<em>', 'EM').replace('<noem>', 'O') 
                               for label in dialogue.labels]
            
            # Tokenize with the model tokenizer
            tokenized = self.tokenizer(
                dialogue.utterance,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors=None,
                return_offsets_mapping=True,
                add_special_tokens=True
            )
            
            # Align labels with subword tokens
            aligned_labels = self._align_labels_with_tokens(
                dialogue.utterance,
                dialogue.tokens,
                scheme_labels,
                tokenized['offset_mapping']
            )
            
            # Convert labels to IDs
            label_ids = [self.label_to_id.get(label, -100) for label in aligned_labels]
            
            # Create tokenized sample
            sample = TokenizedSample(
                input_ids=tokenized['input_ids'],
                attention_mask=tokenized['attention_mask'],
                labels=label_ids,
                token_type_ids=tokenized.get('token_type_ids', None)
            )
            
            tokenized_samples.append(sample)
        
        logger.info(f"Tokenized {len(tokenized_samples)} samples")
        return tokenized_samples
    
    def _align_labels_with_tokens(
        self, 
        text: str, 
        word_tokens: List[str], 
        word_labels: List[str],
        offset_mapping: List[Tuple[int, int]]
    ) -> List[str]:
        """
        Align word-level labels with subword tokens.
        
        Args:
            text: Original text
            word_tokens: List of word tokens
            word_labels: List of word-level labels
            offset_mapping: List of (start, end) character offsets for each subword
            
        Returns:
            List of aligned labels for subword tokens
        """
        aligned_labels = []
        
        # Create word-to-character mapping
        word_spans = []
        start_idx = 0
        for word in word_tokens:
            start = text.find(word, start_idx)
            if start != -1:
                end = start + len(word)
                word_spans.append((start, end))
                start_idx = end
            else:
                word_spans.append((start_idx, start_idx))
        
        # Align subword tokens with word labels
        for start, end in offset_mapping:
            if start == 0 and end == 0:  # Special tokens ([CLS], [SEP], [PAD])
                aligned_labels.append(-100)  # Ignore special tokens
            else:
                # Find which word this subword belongs to
                label = 'O'  # Default
                for i, (word_start, word_end) in enumerate(word_spans):
                    if word_start <= start < word_end or word_start < end <= word_end:
                        if i < len(word_labels):
                            label = word_labels[i]
                        break
                
                aligned_labels.append(label)
        
        return aligned_labels
    
    def split_data(
        self, 
        samples: List[TokenizedSample], 
        train_ratio: float = 0.8,
        valid_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_seed: int = 42
    ) -> Tuple[List[TokenizedSample], List[TokenizedSample], List[TokenizedSample]]:
        """
        Split data into train/valid/test sets.
        
        Args:
            samples: List of TokenizedSample objects
            train_ratio: Ratio for training set
            valid_ratio: Ratio for validation set
            test_ratio: Ratio for test set
            random_seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_samples, valid_samples, test_samples)
        """
        assert abs(train_ratio + valid_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"
        
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # First split: train vs (valid + test)
        train_samples, temp_samples = train_test_split(
            samples, 
            test_size=1-train_ratio, 
            random_state=random_seed
        )
        
        # Second split: valid vs test
        valid_size = valid_ratio / (valid_ratio + test_ratio)
        valid_samples, test_samples = train_test_split(
            temp_samples, 
            test_size=1-valid_size, 
            random_state=random_seed
        )
        
        logger.info(f"Data split - Train: {len(train_samples)}, "
                   f"Valid: {len(valid_samples)}, Test: {len(test_samples)}")
        
        return train_samples, valid_samples, test_samples
    
    def create_dataloader(
        self, 
        samples: List[TokenizedSample], 
        batch_size: int = 16,
        shuffle: bool = True
    ) -> DataLoader:
        """
        Create PyTorch DataLoader from tokenized samples.
        
        Args:
            samples: List of TokenizedSample objects
            batch_size: Batch size
            shuffle: Whether to shuffle data
            
        Returns:
            PyTorch DataLoader
        """
        dataset = ERADataset(samples)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self._collate_fn
        )
        return dataloader
    
    def _collate_fn(self, batch):
        """
        Collate function for DataLoader.
        
        Args:
            batch: List of samples from ERADataset
            
        Returns:
            Batched tensors
        """
        input_ids = torch.tensor([sample.input_ids for sample in batch])
        attention_mask = torch.tensor([sample.attention_mask for sample in batch])
        labels = torch.tensor([sample.labels for sample in batch])
        
        batch_dict = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
        
        # Add token_type_ids if available
        if batch[0].token_type_ids is not None:
            token_type_ids = torch.tensor([sample.token_type_ids for sample in batch])
            batch_dict['token_type_ids'] = token_type_ids
        
        return batch_dict
    
    def process_full_pipeline(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Run the complete data processing pipeline.
        
        Returns:
            Tuple of (train_loader, valid_loader, test_loader)
        """
        logger.info("Starting full data processing pipeline...")
        
        # Step 1: Load raw dialogues
        dialogues = self.load_empathetic_dialogues()
        
        # Step 2: Annotate with LLM
        annotated_dialogues = self.annotate_with_llm(dialogues)
        
        # Step 3: Tokenize and align labels
        tokenized_samples = self.tokenize_and_align_labels(annotated_dialogues)
        
        # Step 4: Split data
        train_samples, valid_samples, test_samples = self.split_data(tokenized_samples)
        
        # Step 5: Create dataloaders
        train_loader = self.create_dataloader(train_samples, shuffle=True)
        valid_loader = self.create_dataloader(valid_samples, shuffle=False)
        test_loader = self.create_dataloader(test_samples, shuffle=False)
        
        logger.info("✓ Data processing pipeline completed")
        
        return train_loader, valid_loader, test_loader


class ERADataset(Dataset):
    """
    PyTorch Dataset for ERA training.
    """
    
    def __init__(self, samples: List[TokenizedSample]):
        """
        Initialize dataset.
        
        Args:
            samples: List of TokenizedSample objects
        """
        self.samples = samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> TokenizedSample:
        return self.samples[idx]


# Example usage and testing functions
def test_data_processor():
    """Test function for data processor."""
    processor = EmpathyDataProcessor(
        data_dir="./dataset",
        tokenizer_model="roberta-base",
        tagging_scheme="BIO"
    )
    
    # Test loading dialogues
    dialogues = processor.load_empathetic_dialogues()
    print(f"Loaded {len(dialogues)} dialogues")
    
    if dialogues:
        print(f"Sample dialogue: {dialogues[0].utterance}")
    
    return processor


if __name__ == "__main__":
    # Test the data processor
    test_data_processor()