"""
Data Processor for ERA System

Handles EmpatheticDialogues loading, ChatGLM4 annotation, 
token-label alignment, and train/valid/test splitting (8:1:1).

Uses IO tagging scheme: {O: 0, EM: 1}
Based on specifications in EAR.md.
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from pathlib import Path
import logging
from dataclasses import dataclass
import pickle
import random
from sklearn.model_selection import train_test_split
from openai import OpenAI
import re
import string
from dotenv import load_dotenv

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
EDGE_PUNCT = string.punctuation
load_dotenv()
GROK3_API = os.getenv('GROK3_API')
GROK_URL = os.getenv('GROK_URL')

@dataclass
class DialogueSample:
    """
    Represents a single dialogue sample with emotion reason annotations.
    """
    conv_id: str
    round_talk: int
    speaker_id: List[int]
    emotion: str
    prompt: str
    user_tokens: List[str]
    user_labels: List[str]
    response_tokens: List[str]
    response_labels: List[str]
    selfeval: str
    user_src_text: str
    consistency: bool
    

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
    - Uses IO tagging scheme only: {O: 0, EM: 1}
    - Implements 8:1:1 train/valid/test split
    """
    
    def __init__(
        self,
        data_dir: str,
        if_api: bool = False,
        llm_model_path: str = None,
        tokenizer_model: str = "bert-base",
        max_length: int = 512,
        cache_dir: str = "./cache"
    ):
        """
        Initialize data processor.
        
        Args:
            data_dir: Directory containing EmpatheticDialogues CSV files
            llm_model_path: Path to local LLM model for annotation
            tokenizer_model: Model name or path for tokenizer
            max_length: Maximum sequence length
            cache_dir: Directory for caching processed data
        """
        self.data_dir = Path(data_dir)
        self.llm_model_path = llm_model_path
        self.tokenizer_model = tokenizer_model
        self.max_length = max_length
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model) if not if_api else None
        
        # Label mappings
        self.label_to_id = {"O": 0, "EM": 1}
        self.id_to_label = {0: "O", 1: "EM"}
        self.num_labels = 2
        
        # Initialize LLM for annotation (if provided)
        self.llm_model = None
        self.llm_tokenizer = None
        if not if_api and llm_model_path and os.path.exists(llm_model_path):
            self._load_llm_model()
    
    def _load_API_data(self):
        """Load API LLM model for data annotation."""
        return GROK3_API, GROK_URL

    def _call_API_LLM_model(self, given_prompt: str):
        api, model_url = self._load_API_data()
        client = OpenAI(
            api_key=api,
            base_url=model_url
        )

        response = client.chat.completions.create(
            model="grok-3",
            messages=[
                {"role": "system", "content": """You are a helpful assistant. Please analyze the following dialogue and identify words that represent emotion reasons.
        Mark each word with <em> if it's an emotion reason word, or <noem> if it's not.
        Attention! you should only label for words in "Speaker" and ignore "Emotion" and "Context" 
        Please provide the output in the following format:
        word1:<em/noem> word2:<em/noem> ...
        Example:
        Input: "I am very sad because I lost my dog"
        Output: I:<noem> am:<noem> very:<noem> sad:<em> because:<noem> I:<noem> lost:<em> my:<noem> dog:<noem>"""},
                {"role": "user", "content": str(given_prompt)}
            ],
            max_tokens=1000,
            temperature=0.0
        )
    
        return response.choices[0].message.content
    
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
            
    def model_eject(self, llm_model, llm_tokenizer):
        self.llm_model = llm_model
        self.llm_tokenizer = llm_tokenizer
    
    def pattern_parser(self, df: pd.DataFrame, if_api: bool, task, threshold):
        dialogues, counter, prev = [], 400, 400
        
        for conv_id, group in df.groupby('conv_id'):
            group_length = len(group)
            load_length = (group_length // 2) * 2
            
            if group_length<2: continue
            
            user_id, response_id, group_dialogues = int(group.iloc[0]['speaker_idx']), int(group.iloc[1]['speaker_idx']), list()
            counter += 1
            for idx in range(0, load_length, 2):
                user_row, response_row = group.iloc[idx], group.iloc[idx+1]
                # Extract dialogue information
                conv_id = str(user_row.get('conv_id', ''))
                round_talk = int(group_length//2)
                emotion = str(user_row.get('context', ''))  # emotion context
                prompt = str(user_row.get('prompt', ''))
                selfeval = str(user_row.get('selfeval', ''))
                # selfeval = [list(map(int, eval.split("|"))) for eval in selfeval.split("_")]
                
                user_utterance = str(user_row.get('utterance', ''))
                user_speaker_idx = int(user_row.get('speaker_idx', -1))
                assert user_speaker_idx == user_id, "User speaker_idx mismatch"
                if not if_api:
                    user_token, user_label, consistency_check = self.fast_annotate_llm(user_utterance, user_speaker_idx == response_id)
                else:
                    user_token, user_label = self.fast_annotate_api_llm(user_utterance, user_speaker_idx == response_id)
        
                    
                response_utterance = str(response_row.get('utterance', ''))
                response_speaker_idx = int(response_row.get('speaker_idx', -1))
                assert response_speaker_idx == response_id, "Response speaker_idx mismatch"
                response_token, response_label, _ = self.fast_annotate_llm(response_utterance, response_speaker_idx == response_id)
                # Create sample without labels initially
                sample = DialogueSample(
                    conv_id=conv_id,
                    speaker_id=[user_id, response_id],
                    round_talk=round_talk,
                    emotion=emotion,
                    user_tokens=user_token,
                    user_labels=user_label,
                    response_tokens=response_token,
                    response_labels=response_label,
                    prompt=prompt,
                    selfeval=selfeval,
                    user_src_text=user_utterance,
                    consistency=consistency_check > threshold
                )
                group_dialogues.append(sample)
            dialogues.extend(group_dialogues)
            
            if (counter+1)%100 == 0:
                with open(rf"../../dataset/local_llm/annotated_{task}_{prev}_{counter}.pkl", "wb") as f:
                    pickle.dump(dialogues, f)
                print(f"saved {(counter+1)//100}th round data with name annotated_{prev}_{counter}.pkl")
                prev, dialogues = counter+1, []

        return dialogues
    
    # More robust data cleaning function
    def clean_empathetic_dialogues_data(self, df):
        print("Cleaning empathetic dialogues data...")
        
        # Check for various corruption patterns
        initial_count = len(df)
        
        # 1. Remove rows with extremely long conv_id (likely corrupted)
        conv_id_lengths = df['conv_id'].str.len()
        max_normal_length = 50  # Normal conv_ids are like "hit:123_conv:456"
        corrupted_conv_id = df[conv_id_lengths > max_normal_length]
        
        conv_id_pattern = r'^hit:\d+_conv:\d+$'
        valid_conv_id_mask = df['conv_id'].str.match(conv_id_pattern, na=False)
        
        # 3. Remove rows where essential fields are NaN
        essential_fields = ['conv_id', 'utterance']
        valid_essential_mask = df[essential_fields].notna().all(axis=1)
        
        # Combine all cleaning criteria
        clean_mask = (conv_id_lengths <= max_normal_length) & valid_conv_id_mask & valid_essential_mask
        
        cleaned_df = df[clean_mask].copy()
        
        print(f"Original rows: {initial_count}")
        print(f"Rows with long conv_id: {len(corrupted_conv_id)}")
        print(f"Rows with invalid conv_id format: {(~valid_conv_id_mask).sum()}")
        print(f"Rows with missing essential fields: {(~valid_essential_mask).sum()}")
        print(f"Final cleaned rows: {len(cleaned_df)}")
        print(f"Removed {initial_count - len(cleaned_df)} corrupted rows total")
        
        return cleaned_df

    
    def load_empathetic_dialogues(self, given_df: pd.DataFrame, filename: str, if_api: bool=False) -> List[DialogueSample]:
        """
        Load EmpatheticDialogues dataset from CSV files.
        
        Returns:
            List of DialogueSample objects
        """
        
        dialogues = []
        
        logger.info(f"Loading {filename} data...")
        df = given_df
        if filename=="test":
            df = self.clean_empathetic_dialogues_data(df)
        dialogues = self.pattern_parser(df, if_api, filename, 0.6)
        
        logger.info(f"Loaded {len(dialogues)} dialogue samples")
        
        return dialogues
    
    def jaccard_sim(self, s1, s2):
        clean = lambda s: set([
            w for w in re.sub(r"[^\w\s]", "", s.lower()).split()
            if w not in ENGLISH_STOP_WORDS
        ])
        set1, set2 = clean(s1), clean(s2)
        return len(set1 & set2) / max(1, len(set1 | set2))
        
    def fast_annotate_llm(self, dialogue: str, is_response: bool):
        if is_response:
            tokens = [tok for tok in dialogue.split()]
            labels = ["<noem>" for _ in tokens]
            return tokens, labels, -1
        preprocessed = [self.normalize_text(tok) for tok in dialogue.split()]
        length = len(preprocessed)
        dialogue = ' '.join(preprocessed)
        prompt = self._create_annotation_prompt(dialogue)
        response = self._query_llm(prompt)
        tokens, labels = self._parse_llm_response(response, dialogue)
        
        simlarity = self.jaccard_sim(' '.join(tokens[1:1+length]), dialogue)
        return tokens[1:], labels[1:], simlarity
    
    def fast_annotate_api_llm(self, dialogue: str, is_response: bool):
        if is_response:
            tokens = dialogue.split()
            labels = ["<noem>" for _ in tokens]
            return tokens, labels
        response = self._call_API_LLM_model(dialogue)
        tokens, labels = self._parse_llm_response(response, dialogue)
        return tokens, labels

    def normalize_text(self, s: str) -> str:
        # Light normalization; adjust as needed
        s = s.lower()
        # Replace placeholder tokens
        s = s.replace("_comma_", "")
        # Remove repeated spaces and strip
        s = re.sub(r"\s+", " ", s).strip(EDGE_PUNCT)
        return s
    
    def _create_annotation_prompt(self, dialogue: str) -> str:
        """
        Create prompt for LLM annotation.
        
        Args:
            dialogue: DialogueSample to create prompt for
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""
You are an expert linguistic annotator.

Your task:
- For each word spoken by Speaker, decide if it expresses an emotional state or emotional reason.
- If yes → label with <em>
- If not → label with <noem>
- Ignore punctuation completely.
- Important: Give only the labels in the format word:<em/noem> separated by spaces.
- Do not explain your reasoning in the output.

Process:
1. Read the dialogue carefully.
2. Think step by step (internally, without writing it out) about each word's meaning.
3. Then output the final result strictly in the required format.

Dialogue:
Speaker: <Start> {dialogue} <End>

Example:
"I am very sad because I lost my dog"
I:<noem> am:<noem> very:<noem> sad:<em> because:<noem> I:<noem> lost:<em> my:<noem> dog:<noem>

Now your output:
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
        
        parts = response.split()
        for part in parts:
            if ':' in part:
                word, label = part.split(':', 1)
                tokens.append(word)
                labels.append(label)    
            
        # Ensure tokens and labels have same length
        if len(tokens) != len(labels):
            logger.warning("Token-label length mismatch, using simple tokenization")
            tokens = original_text.split()
            labels = ['<noem>'] * len(tokens)
        
        return tokens, labels
    

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
        tokenizer_model="google-bert/bert-base-uncased"
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