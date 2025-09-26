import os
from typing import List, Dict, Optional
import torch
from dataclasses import dataclass
from src.emotion_contagion.data_processor import EmotionContagionDataProcessor
from portable_inference import build_random_intent, get_agent, get_intent_distribution
from src.tokenizer_loader import get_tokenizer
import logging

logger = logging.getLogger(__name__)


TOKENIZER = get_tokenizer()
@dataclass
class EvaluationSample:
    """Single evaluation sample containing user input and target response for generation."""
    user_tokens: List[str]  # User input tokens (unpadded, truncated to max_length)
    user_input_ids: List[int]  # Tokenizer-encoded input ids (length = max_length)
    user_label_ids: List[int]  # User input emotion labels (padded to max_length)
    user_attention_mask: List[int]  # Attention mask from tokenizer (length = max_length)
    response_tokens: List[str]  # Target response tokens (raw tokens, no manual padding)
    response_input_ids: Optional[List[int]] = None  # Tokenizer ids for reference response (no extra special tokens, truncated)
    h_tilde: Optional[torch.Tensor] = None
    p_intent: Optional[List[float]] = None


def prepare_evaluation_data(test_data_path: str = "dataset/available_dataset/test.pkl") -> List[EvaluationSample]:
        """
        Prepare evaluation data from conversational format.
            
        Returns:
            List of EvaluationSample objects
        """
        # Process data in pairs: user=even indices, response=odd indices
        eval_samples = []
        dp = EmotionContagionDataProcessor(max_length=64)
        raw_data = dp.load_data(test_data_path)
        # self.batch_data = dp.process_batch(raw_data, ifeval=True)
        
        
        # Load intent predictions
        model_path = os.path.join(os.path.dirname(__file__), 'pre-trained', 'model', 'model')
        agent = get_agent()
        
        intent_idx = 0
        
        # Process conversations: each conversation is a list of (token, label) tuples
        # We expect alternating user-response pairs within each conversation
        for conv in raw_data:

            user_tokens = conv.user_tokens
            user_labels_raw = conv.user_labels 
            response_tokens = conv.response_tokens
            user_label_ids = [1 if l == '<em>' else 0 for l in user_labels_raw]

            if not user_tokens or not response_tokens:
                continue

            max_len = dp.max_length
            # Truncate tokens & labels together
            if len(user_tokens) > max_len:
                user_tokens = user_tokens[:max_len]
                user_label_ids = user_label_ids[:max_len]

            # Encode with tokenizer (let tokenizer handle padding to max_length)
            user_text = " ".join(user_tokens)
            encoded = TOKENIZER(
                user_text,
                padding='max_length',
                truncation=True,
                max_length=max_len,
                return_tensors='pt'
            )
            user_input_ids = encoded['input_ids'][0].tolist()
            user_attention_mask = encoded['attention_mask'][0].tolist()

            # Pad emotion labels to max_len (independent of wordpiece expansion)
            if len(user_label_ids) < max_len:
                user_label_ids = user_label_ids + [0] * (max_len - len(user_label_ids))
            else:
                user_label_ids = user_label_ids[:max_len]

            # Intent prediction (random placeholder)
            p_intent = get_intent_distribution(agent, [{"origin_prompt": user_text}])[0]
            intent_idx += 1

            # Encode reference response for consistent decoding metrics (avoid adding BOS/EOS twice)
            response_text = " ".join(response_tokens)
            # We don't force max_length padding for reference; just truncate to max_len to be comparable
            resp_encoded = TOKENIZER(
                response_text,
                padding=False,
                truncation=True,
                max_length=max_len,
                add_special_tokens=False,
                return_tensors='pt'
            )
            response_input_ids = resp_encoded['input_ids'][0].tolist()

            eval_samples.append(
                EvaluationSample(
                    user_tokens=user_tokens, 
                    user_input_ids=user_input_ids,
                    user_label_ids=user_label_ids,
                    user_attention_mask=user_attention_mask,
                    response_tokens=response_tokens,
                    response_input_ids=response_input_ids,
                    h_tilde=None,
                    p_intent=p_intent
                )
            )
        
        logger.info(f"Prepared {len(eval_samples)} evaluation samples")
        return eval_samples