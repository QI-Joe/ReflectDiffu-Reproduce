import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass
from pathlib import Path

from torcheval.metrics import Perplexity
TORCHEVAL_AVAILABLE = True

# Import shared components from relevance evaluation
from src.emotion_contagion.data_processor import EmotionContagionDataProcessor
from portable_inference import build_random_intent
from src.tokenizer_loader import get_tokenizer

TOKENIZER = get_tokenizer()

logger = logging.getLogger(__name__)


@dataclass
class InformativenessResults:
    """Container for informativeness evaluation results."""
    perplexity: float
    distinct_1: float
    distinct_2: float
    num_samples: int
    avg_response_length: float
    total_tokens: int
    valid_responses: int


class TextProcessor:
    """Handles text processing and vocabulary operations (shared from relevance_evaluation)."""
    
    def __init__(self, word_embedding):
        self.word_embedding = word_embedding
        self.special_tokens = ["<bos>", "<eos>", "[PAD]", "[UNK]"]
        self._ensure_special_tokens()
        
    def _ensure_special_tokens(self):
        """Ensure special tokens exist in vocabulary."""
        current_vocab = set(self.word_embedding.word_to_idx.keys())
        new_tokens = [tok for tok in self.special_tokens if tok not in current_vocab]
        
        if new_tokens:
            all_tokens = list(current_vocab) + new_tokens
            self.word_embedding.word_to_idx = {word: idx for idx, word in enumerate(sorted(all_tokens))}
            self.word_embedding.idx_to_word = {idx: word for word, idx in self.word_embedding.word_to_idx.items()}
            
            old_vocab_size = self.word_embedding.vocab_size
            new_vocab_size = len(self.word_embedding.word_to_idx)
            if new_vocab_size > old_vocab_size:
                old_weight = self.word_embedding.embedding.weight.data
                device = old_weight.device
                self.word_embedding.embedding = torch.nn.Embedding(new_vocab_size, self.word_embedding.embedding_dim).to(device)
                self.word_embedding.embedding.weight.data[:old_vocab_size] = old_weight
                self.word_embedding.vocab_size = new_vocab_size
    
    @property
    def bos_id(self) -> int:
        return self.word_embedding.word_to_idx.get("<bos>", 0)
    
    @property
    def eos_id(self) -> int:
        return self.word_embedding.word_to_idx.get("<eos>", 0)
    
    @property
    def pad_id(self) -> int:
        return self.word_embedding.word_to_idx.get("[PAD]", 0)
    
    @property
    def unk_id(self) -> int:
        return self.word_embedding.word_to_idx.get("[UNK]", 0)
    
    def tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """Convert tokens to IDs using vocabulary."""
        return [self.word_embedding.word_to_idx.get(t, self.unk_id) for t in tokens]
    
    def ids_to_text(self, ids: List[int]) -> str:
        """Convert IDs to readable text, removing special tokens."""
        if not hasattr(self.word_embedding, 'idx_to_word') or not self.word_embedding.idx_to_word:
            self.word_embedding.idx_to_word = {idx: word for word, idx in self.word_embedding.word_to_idx.items()}
        
        tokens = []
        for i in ids:
            if i in self.word_embedding.idx_to_word:
                token = self.word_embedding.idx_to_word[i]
                if token not in self.special_tokens:
                    tokens.append(token)
        return " ".join(tokens)


class DistinctNCalculator:
    """Calculates Distinct-n metrics to measure lexical diversity."""
    
    @staticmethod
    def compute_distinct_n(responses: List[str], n: int = 1) -> float:
        """Compute distinct-n metric for generated responses."""
        if not responses:
            return 0.0
        
        all_ngrams = []
        for response in responses:
            if not response.strip():
                continue
            tokens = response.strip().lower().split()
            if len(tokens) >= n:
                ngrams = [' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
                all_ngrams.extend(ngrams)
        
        if not all_ngrams:
            return 0.0
        
        unique_ngrams = len(set(all_ngrams))
        total_ngrams = len(all_ngrams)
        return unique_ngrams / total_ngrams


class InformativenessEvaluator:
    """
    Main evaluation class for informativeness metrics using torcheval.Perplexity.
    
    This class provides evaluation for:
    - Perplexity using torcheval.Perplexity metric
    - Distinct-1 and Distinct-2 metrics for lexical diversity
    """
    
    def __init__(self, encoder, it_module, decoder, device: torch.device):
        """
        Initialize evaluator with model components.
        
        Args:
            encoder: Emotion Contagion Encoder
            it_module: Intent Twice Module  
            decoder: Response Decoder (with or without loss wrapper)
            device: Torch device
        """
        self.device = device
        self.encoder = encoder
        self.it_module = it_module
        self.decoder = decoder
        
        # Initialize text processor
        self.text_processor = TextProcessor(encoder.word_embedding)
        
        # Initialize torcheval perplexity metric if available
        if TORCHEVAL_AVAILABLE:
            self.perplexity_metric = Perplexity(device=device)
        else:
            self.perplexity_metric = None
            logger.warning("torcheval not available, perplexity will be computed manually")
        
        # Initialize semantic scorer for inference
        from src.emotion_contagion.foundation_emb import IntentSemanticScorer
        self.semantic_scorer = IntentSemanticScorer(d_in=128).to(device)
        
        logger.info(f"InformativenessEvaluator initialized on device: {device}")
    
    def prepare_evaluation_data(self, data_path: str = "dataset/emotion_labels_test.pkl", max_samples: int = None):
        """
        Prepare evaluation data from test file following relevance_evaluation pattern.
        
        Args:
            data_path: Path to test data file
            max_samples: Maximum number of samples to process
            
        Returns:
            List of processed evaluation samples
        """
        dp = EmotionContagionDataProcessor(max_length=64)
        raw_data = dp.load_data(data_path)
        
        # Load intent predictions
        model_path = Path(__file__).parent.parent / 'pre-trained' / 'model' / 'model'
        # agent = build_agent(str(model_path))
        
        eval_samples = []
        
        # Process conversations following relevance_evaluation pattern
        for conv_idx, conv in enumerate(raw_data):
            if not isinstance(conv, (list, tuple)) or len(conv) != 2:
                continue
            user, response = conv
            min_len = min(len(user), len(response))

            for i in range(min_len):
                try:
                    user_item = user[i]
                    response_item = response[i]
                    user_tokens, user_labels = zip(*user_item)
                    response_tokens, _ = zip(*response_item)
                except Exception:
                    continue

                user_tokens = list(user_tokens)
                response_tokens = list(response_tokens)

                if not user_tokens or not response_tokens:
                    continue

                max_len = dp.max_length
                user_label_ids = [1 if label == '<em>' else 0 for label in user_labels]

                if len(user_tokens) > max_len:
                    user_tokens = user_tokens[:max_len]
                    user_label_ids = user_label_ids[:max_len]

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

                if len(user_label_ids) < max_len:
                    user_label_ids = user_label_ids + [0] * (max_len - len(user_label_ids))
                else:
                    user_label_ids = user_label_ids[:max_len]

                response_text = " ".join(response_tokens)
                resp_encoded = TOKENIZER(
                    response_text,
                    padding=False,
                    truncation=True,
                    max_length=max_len,
                    add_special_tokens=False,
                    return_tensors='pt'
                )
                response_input_ids = resp_encoded['input_ids'][0].tolist()

                p_intent = build_random_intent([user_text], self.device)

                eval_samples.append({
                    'user_tokens': user_tokens,
                    'user_input_ids': user_input_ids,
                    'user_label_ids': user_label_ids,
                    'user_attention_mask': user_attention_mask,
                    'response_tokens': response_tokens,
                    'response_input_ids': response_input_ids,
                    'p_intent': p_intent
                })

                if max_samples and len(eval_samples) >= max_samples:
                    break
            if max_samples and len(eval_samples) >= max_samples:
                break
        
        logger.info(f"Prepared {len(eval_samples)} evaluation samples")
        return eval_samples
    
    @torch.no_grad()
    def compute_perplexity_for_response(self, sample: dict, response_text: str) -> Optional[float]:
        """
        Compute perplexity for a single response using the model decoder.
        
        Args:
            sample: Evaluation sample with user context
            response_text: Generated response text
            
        Returns:
            Perplexity score or None if computation fails
        """
        if not response_text.strip():
            return None
        
        
        # 1. Encode user input using tokenizer ids
        enc_out = self.encoder.forward(
            tokens=torch.tensor([sample['user_input_ids']], dtype=torch.long, device=self.device),
            label_ids=torch.tensor([sample['user_label_ids']], dtype=torch.long, device=self.device),
            attention_mask=torch.tensor([sample['user_attention_mask']], dtype=torch.long, device=self.device),
            h_tilde=None
        )
        
        Q = enc_out["Q"]
        
        # 2. Intent Twice Integration
        P_semantic, _ = self.semantic_scorer(Q)
        enc_out["P_semantic"] = P_semantic
        
        it_out = self.it_module.forward(
            encoder_out=enc_out,
            intent_out={"p_intent": sample['p_intent']}
        )
        
        Emofused = it_out["Emofused"]
        if Emofused.dim() == 2:
            Emofused = Emofused.unsqueeze(1)
        
        # 3. Prepare response tokens for perplexity computation
        # Tokenize response text with central tokenizer (no extra specials)
        resp_enc = TOKENIZER(
            response_text,
            add_special_tokens=False,
            truncation=True,
            max_length=64
        )
        token_ids = resp_enc['input_ids']

        bos_id = TOKENIZER.cls_token_id if TOKENIZER.cls_token_id is not None else self.text_processor.bos_id
        eos_id = TOKENIZER.sep_token_id if TOKENIZER.sep_token_id is not None else self.text_processor.eos_id

        input_ids = [bos_id] + token_ids
        target_ids = token_ids + [eos_id]
        
        if len(input_ids) != len(target_ids):
            return None
        
        # Convert to tensors
        input_tensor = torch.tensor([input_ids], device=self.device, dtype=torch.long)
        target_tensor = torch.tensor(target_ids, device=self.device, dtype=torch.long)
        
        # 4. Forward pass through decoder
        src_ids = torch.zeros((1, Emofused.size(1)), device=self.device, dtype=torch.long)
        
        decoder_out = self.decoder.decoder(
            trg_input_ids=input_tensor,
            emofused=Emofused,
            src_token_ids=src_ids,
            tgt_key_padding_mask=None,
            emofused_key_padding_mask=None,
            extended_vocab_size=None
        )
        
        # Get logits and compute perplexity
        logits = decoder_out["Pw"]  # [1, seq_len, vocab_size]
        
        self.perplexity_metric.reset()
        self.perplexity_metric.update(logits, target_tensor.unsqueeze(0))
        return self.perplexity_metric.compute().item()
                
    
    def evaluate_responses(self, responses: List[str], eval_samples: List[dict] = None) -> InformativenessResults:
        """
        Evaluate informativeness metrics for generated responses.
        
        Args:
            responses: List of generated response strings
            eval_samples: Optional evaluation samples for perplexity computation
            
        Returns:
            InformativenessResults with all metrics
        """
        if not responses:
            return InformativenessResults(
                perplexity=float('inf'),
                distinct_1=0.0,
                distinct_2=0.0,
                num_samples=0,
                avg_response_length=0.0,
                total_tokens=0,
                valid_responses=0
            )
        
        # Filter valid responses
        valid_responses = [r for r in responses if r.strip()]
        
        # Compute distinct-n metrics
        distinct_1 = DistinctNCalculator.compute_distinct_n(valid_responses, n=1)
        distinct_2 = DistinctNCalculator.compute_distinct_n(valid_responses, n=2)
        
        # Compute perplexity if evaluation samples are provided
        perplexities = []
        total_tokens = 0
        
        if eval_samples and len(eval_samples) == len(responses):
            for sample, response in zip(eval_samples, responses):
                if response.strip():
                    ppl = self.compute_perplexity_for_response(sample, response)
                    if ppl is not None and not torch.isnan(torch.tensor(ppl)) and not torch.isinf(torch.tensor(ppl)):
                        perplexities.append(ppl)
                        total_tokens += len(response.split())
        
        # Calculate average perplexity
        if perplexities:
            avg_perplexity = sum(perplexities) / len(perplexities)
        else:
            avg_perplexity = float('inf')
            logger.warning("No valid perplexity scores computed")
        
        # Compute statistics
        response_lengths = [len(r.split()) for r in valid_responses]
        avg_length = sum(response_lengths) / len(response_lengths) if response_lengths else 0.0
        
        return InformativenessResults(
            perplexity=avg_perplexity,
            distinct_1=distinct_1,
            distinct_2=distinct_2,
            num_samples=len(valid_responses),
            avg_response_length=avg_length,
            total_tokens=total_tokens,
            valid_responses=len(perplexities)
        )
    
    def evaluate_from_file(self, data_path: str = "dataset/emotion_labels_test.pkl", max_samples: int = None) -> InformativenessResults:
        """
        Convenience method to evaluate informativeness from data file.
        
        Args:
            data_path: Path to test data file
            max_samples: Maximum number of samples to evaluate
            
        Returns:
            InformativenessResults object
        """
        eval_samples = self.prepare_evaluation_data(data_path=data_path, max_samples=max_samples)
        
        # Extract response texts for evaluation
        responses = [' '.join(sample['response_tokens']) for sample in eval_samples]
        
        return self.evaluate_responses(responses, eval_samples)


def print_informativeness_results(results: InformativenessResults, title: str = "Informativeness Results"):
    """Pretty print informativeness results."""
    print(f"\n{'='*50}")
    print(f"{title:^50}")
    print(f"{'='*50}")
    print(f"Samples:          {results.num_samples}")
    print(f"Valid Responses:  {results.valid_responses}")
    print(f"Total Tokens:     {results.total_tokens}")
    print(f"Avg Length:       {results.avg_response_length:.1f} words")
    print(f"{'='*50}")
    print(f"Perplexity (PPL): {results.perplexity:.2f}")
    print(f"Distinct-1:       {results.distinct_1:.3f}")
    print(f"Distinct-2:       {results.distinct_2:.3f}")
    print(f"{'='*50}")


def load_model_and_evaluate(model_path: str, data_path: str = "dataset/emotion_labels_test.pkl", 
                          max_samples: int = None, device: torch.device = None) -> InformativenessResults:
    """
    Convenience function to load model and evaluate informativeness.
    
    Args:
        model_path: Path to saved model (.pt file)
        data_path: Path to test data file
        max_samples: Maximum number of samples to evaluate
        device: Torch device
        
    Returns:
        InformativenessResults object
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = torch.load(model_path, map_location=device)
    model.eval()
    
    # Create evaluator
    decoder = getattr(model, 'decoder_with_loss', model.decoder)
    evaluator = InformativenessEvaluator(
        encoder=model.encoder,
        it_module=model.it_module,
        decoder=decoder,
        device=device
    )
    
    # Evaluate
    return evaluator.evaluate_from_file(data_path=data_path, max_samples=max_samples)


# Example usage
if __name__ == "__main__":
    # Example evaluation without model loading
    sample_responses = [
        "I understand how you feel.",
        "That sounds really difficult to deal with.", 
        "I'm here to listen if you need to talk.",
        "Have you considered talking to someone about this?",
        "Your feelings are completely valid."
    ]
    
    # Compute only distinct-n metrics (no model needed)
    print("Computing distinct-n metrics for sample responses:")
    d1 = DistinctNCalculator.compute_distinct_n(sample_responses, n=1)
    d2 = DistinctNCalculator.compute_distinct_n(sample_responses, n=2)
    print(f"Distinct-1: {d1:.3f}")
    print(f"Distinct-2: {d2:.3f}")
    
    # For full evaluation with perplexity, use:
    results = load_model_and_evaluate("path/to/model.pt", max_samples=100)
    print_informativeness_results(results)
