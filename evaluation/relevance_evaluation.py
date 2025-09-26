import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Any
import logging
from dataclasses import dataclass
import json
import time
import os
from pathlib import Path
import pickle
from src.emotion_contagion.encoder import EmotionContagionEncoder
from src.intent_twice.intent_twice_integration import IntentTwiceModule
from src.intent_twice.EMU import EMUConfig, EMU
from src.intent_twice.response_decoder import PaperCompliantDecoderWithLoss
from src.tokenizer_loader import get_tokenizer
from evaluation.informativeness import InformativenessEvaluator
from evaluation.utils import prepare_evaluation_data, EvaluationSample

from sacrebleu.metrics import BLEU
SACREBLEU_AVAILABLE = True
EVALUATION_AVAILABLE = True
TOKENIZER = get_tokenizer()

try:
    from bart_score import BARTScorer
    BARTSCORE_AVAILABLE = True
except ImportError:
    BARTSCORE_AVAILABLE = False
    print("Warning: bart-score not available. Install with: pip install bart-score")

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResults:
    """Container for evaluation results."""
    bleu_1: float
    bleu_2: float
    bleu_3: float
    bleu_4: float
    bleu_overall: float
    bart_score: float
    brevity_penalty: float
    num_samples: int
    avg_hyp_length: float
    avg_ref_length: float
    generation_time: float


class TextProcessor:
    """Handles text processing and vocabulary operations."""
    
    def __init__(self, word_embedding):
        self.word_embedding = word_embedding
        self.special_tokens = ["<bos>", "<eos>", "[PAD]", "[UNK]"]
        self._ensure_special_tokens()
        
    def _ensure_special_tokens(self):
        """Ensure special tokens exist in vocabulary."""
        current_vocab = set(self.word_embedding.word_to_idx.keys())
        new_tokens = [tok for tok in self.special_tokens if tok not in current_vocab]
        
        if new_tokens:
            # Extend vocabulary
            all_tokens = list(current_vocab) + new_tokens
            self.word_embedding.word_to_idx = {word: idx for idx, word in enumerate(sorted(all_tokens))}
            self.word_embedding.idx_to_word = {idx: word for word, idx in self.word_embedding.word_to_idx.items()}
            
            # Extend embedding layer
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
        return self.word_embedding.word_to_idx["<bos>"]
    
    @property
    def eos_id(self) -> int:
        return self.word_embedding.word_to_idx["<eos>"]
    
    @property
    def pad_id(self) -> int:
        return self.word_embedding.word_to_idx["[PAD]"]
    
    @property
    def unk_id(self) -> int:
        return self.word_embedding.word_to_idx.get("[UNK]", 0)
    
    def tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """Convert tokens to IDs using vocabulary."""
        return [self.word_embedding.word_to_idx.get(t, self.unk_id) for t in tokens]
    
    def ids_to_tokens(self, ids: List[int]) -> List[str]:
        """Convert IDs back to tokens."""
        if not hasattr(self.word_embedding, 'idx_to_word') or not self.word_embedding.idx_to_word:
            # Build reverse mapping if not available
            self.word_embedding.idx_to_word = {idx: word for word, idx in self.word_embedding.word_to_idx.items()}
        
        tokens = []
        for i in ids:
            if i in self.word_embedding.idx_to_word:
                tokens.append(self.word_embedding.idx_to_word[i])
            else:
                tokens.append("[UNK]")
        return tokens
    
    def ids_to_text(self, ids: List[int]) -> str:
        """Convert IDs to readable text, removing special tokens."""
        tokens = self.ids_to_tokens(ids)
        # Remove special tokens
        tokens = [t for t in tokens if t not in self.special_tokens]
        return " ".join(tokens)


class InferenceEngine:
    """Handles autoregressive inference for response generation."""

    def __init__(self, encoder: EmotionContagionEncoder, it_module: IntentTwiceModule, decoder: PaperCompliantDecoderWithLoss, \
        text_processor: TextProcessor, device: torch.device):
        self.encoder = encoder
        self.it_module = it_module
        self.decoder = decoder
        self.text_processor = text_processor
        self.device = device
        
        # Import here to avoid circular imports
        from src.emotion_contagion.foundation_emb import IntentSemanticScorer
        self.semantic_scorer = IntentSemanticScorer(d_in=128).to(device)  # Adjust d_in based on Q dimension
    
    @torch.no_grad()
    def generate_single_response(
        self, 
        sample: EvaluationSample,
        max_len: int = 32,
        use_pointer: bool = False,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0
    ) -> List[int]:
        """
        Generate response for a single sample using autoregressive decoding.
        
        Args:
            sample: Input evaluation sample
            max_len: Maximum generation length
            use_pointer: Whether to use pointer-generator mechanism
            temperature: Sampling temperature (1.0 = no scaling)
            top_k: Top-k sampling (0 = disabled)
            top_p: Top-p (nucleus) sampling (1.0 = disabled)
            
        Returns:
            Generated token IDs (without <bos> and <eos>)
        """
        # 1. Emotion Contagion Encoder
        tokens = torch.tensor([sample.user_input_ids], dtype=torch.long, device=self.device)
        enc_out = self.encoder.forward(
            tokens=tokens,
            label_ids=torch.tensor([sample.user_label_ids], dtype=torch.long, device=self.device),
            attention_mask=torch.tensor([sample.user_attention_mask], dtype=torch.long, device=self.device),
            h_tilde=sample.h_tilde
        )
        
        Q = enc_out["Q"]  # [1, D] or [1, Lq, D]
        
        # 2. Intent Twice Integration
        P_semantic, _ = self.semantic_scorer(Q)
        enc_out["P_semantic"] = P_semantic
        
        it_out = self.it_module.forward(
            encoder_out=enc_out, 
            intent_out={"p_intent": sample.p_intent}
        )
        
        Emofused = it_out["Emofused"]  # [1, D] or [1, L_emof, D]
        if Emofused.dim() == 2:
            Emofused = Emofused.unsqueeze(1)  # [1, 1, D]
        
        # 3. Prepare source token IDs for pointer-generator
        if use_pointer:
            src_ids = torch.tensor(
                [self.text_processor.tokens_to_ids(sample.user_tokens)], 
                device=self.device, dtype=torch.long
            )
            L_emof = Emofused.size(1)
            # Align length
            if src_ids.size(1) < L_emof:
                src_ids = F.pad(src_ids, (0, L_emof - src_ids.size(1)), value=self.text_processor.pad_id)
            else:
                src_ids = src_ids[:, :L_emof]
        else:
            src_ids = torch.zeros((1, Emofused.size(1)), device=self.device, dtype=torch.long)
        
        # 4. Autoregressive generation
        # Use central TOKENIZER BOS/EOS for generation (preferred). Falls back to legacy TextProcessor IDs
        # if tokenizer lacks cls/sep definitions. This keeps backward compatibility while unifying paths.
        bos_id = TOKENIZER.cls_token_id if TOKENIZER.cls_token_id is not None else self.text_processor.bos_id
        eos_id = TOKENIZER.sep_token_id if TOKENIZER.sep_token_id is not None else self.text_processor.eos_id
        ys = torch.full((1, 1), bos_id, device=self.device, dtype=torch.long)
        finished = torch.zeros(1, dtype=torch.bool, device=self.device)

        for step in range(max_len):
            out = self.decoder.decoder(
                trg_input_ids=ys,
                emofused=Emofused,
                src_token_ids=src_ids,
                tgt_key_padding_mask=None,
                emofused_key_padding_mask=None,
                extended_vocab_size=None
            )

            # Get probability distribution for last position
            Pw_last = out["Pw"][:, -1, :]  # [1, V]

            # Apply temperature scaling
            if temperature != 1.0:
                Pw_last = Pw_last / temperature

            # Apply top-k filtering
            if top_k > 0:
                top_k_vals, top_k_indices = torch.topk(Pw_last, top_k, dim=-1)
                mask = torch.full_like(Pw_last, float('-inf'))
                mask.scatter_(1, top_k_indices, top_k_vals)
                Pw_last = mask

            # Apply top-p filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(Pw_last, descending=True, dim=-1)
                cumulative_probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                Pw_last[indices_to_remove] = float('-inf')

            # Sample next token (greedy if temperature=1.0 and no top-k/top-p)
            if temperature == 1.0 and top_k == 0 and top_p == 1.0:
                next_id = Pw_last.argmax(dim=-1)
            else:
                probs = F.softmax(Pw_last, dim=-1)
                next_id = torch.multinomial(probs, 1).squeeze(1)

            # Append to sequence
            ys = torch.cat([ys, next_id.unsqueeze(1)], dim=1)

            # Check for EOS
            finished |= (next_id == eos_id)
            if bool(finished.item()):
                break

        # Extract generated sequence (remove <bos> and optionally <eos>)
        gen_ids = ys[0].tolist()
        if gen_ids and gen_ids[0] == bos_id:
            gen_ids = gen_ids[1:]
        if gen_ids and gen_ids[-1] == eos_id:
            gen_ids = gen_ids[:-1]

        return gen_ids
    
    @torch.no_grad()
    def batch_generate(
        self, 
        eval_samples: List[EvaluationSample],
        max_len: int = 32,
        use_pointer: bool = False,
        **generation_kwargs
    ) -> Tuple[List[str], List[str]]:
        """
        Generate responses for a batch of samples.
        
        Args:
            eval_samples: List of evaluation samples
            max_len: Maximum generation length
            use_pointer: Whether to use pointer-generator
            **generation_kwargs: Additional generation parameters
            
        Returns:
            Tuple of (hypotheses, references) as text strings
        """
        hyps, refs = [], []
        
        bos_id = TOKENIZER.cls_token_id if TOKENIZER.cls_token_id is not None else self.text_processor.bos_id
        eos_id = TOKENIZER.sep_token_id if TOKENIZER.sep_token_id is not None else self.text_processor.eos_id

        for sample in eval_samples:
            hyp_ids = self.generate_single_response(
                sample, max_len=max_len, use_pointer=use_pointer, **generation_kwargs
            )
            # Decode with tokenizer; ensure special tokens removed
            hyp_text = TOKENIZER.decode(hyp_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

            if sample.response_input_ids is not None:
                ref_ids = sample.response_input_ids
            else:
                # Fallback: encode response tokens on the fly
                ref_text_join = " ".join(sample.response_tokens)
                ref_ids = TOKENIZER(
                    ref_text_join,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=len(hyp_ids) + 5  # heuristic buffer
                )["input_ids"]
            ref_text = TOKENIZER.decode(ref_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            
            print(f"user input: {' '.join(sample.user_tokens)} \nmodel output: {hyp_text}\nreference: {ref_text}\n")

            hyps.append(hyp_text)
            refs.append(ref_text)
        
        return hyps, refs


class MetricsCalculator:
    """Calculates evaluation metrics: BLEU and BARTScore."""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self._bart_scorer = None
        
    def compute_bleu_metrics(self, hyps: List[str], refs: List[str]) -> Dict[str, float]:
        """
        Compute BLEU-1/2/3/4 metrics using sacrebleu.
        
        Args:
            hyps: List of hypothesis strings
            refs: List of reference strings
            
        Returns:
            Dictionary with BLEU scores
        """
        if not SACREBLEU_AVAILABLE:
            logger.warning("sacrebleu not available, returning dummy BLEU scores")
            return {
                "BLEU": 0.0, "B-1": 0.0, "B-2": 0.0, "B-3": 0.0, "B-4": 0.0, "BP": 1.0
            }
        
        try:
            bleu = BLEU(effective_order=True)
            # sacrebleu expects refs as [num_refs][num_sentences]
            result = bleu.corpus_score(hyps, [refs])
            
            P1, P2, P3, P4 = result.precisions
            scores = {
                "BLEU": float(result.score),
                "B-1": float(P1),
                "B-2": float(P2),
                "B-3": float(P3),
                "B-4": float(P4),
                "BP": float(result.bp)
            }
            return scores
            
        except Exception as e:
            logger.error(f"Error computing BLEU scores: {e}")
            return {
                "BLEU": 0.0, "B-1": 0.0, "B-2": 0.0, "B-3": 0.0, "B-4": 0.0, "BP": 1.0
            }
    
    def compute_bart_score(
        self, 
        hyps: List[str], 
        refs: List[str], 
        checkpoint: str = "facebook/bart-large-cnn",
        batch_size: int = 8
    ) -> float:
        """
        Compute BARTScore using bart-score package.
        
        Args:
            hyps: List of hypothesis strings
            refs: List of reference strings
            checkpoint: BART model checkpoint
            batch_size: Batch size for BARTScore computation
            
        Returns:
            Average BARTScore
        """
        if not BARTSCORE_AVAILABLE:
            logger.warning("bart-score not available, returning dummy BARTScore")
            return 0.0
        
        try:
            if self._bart_scorer is None:
                self._bart_scorer = BARTScorer(device=self.device, checkpoint=checkpoint)
            
            scores = self._bart_scorer.score(hyps, refs, batch_size=batch_size)
            return float(sum(scores) / len(scores)) if scores else 0.0
            
        except Exception as e:
            logger.error(f"Error computing BARTScore: {e}")
            return 0.0


class ReflectDiffuEvaluator:
    """
    Main evaluation class for ReflectDiffu model.
    
    This class provides a complete evaluation framework including:
    - Data preparation and organization
    - Autoregressive inference
    - BLEU and BARTScore computation
    - Result aggregation and reporting
    """
    
    def __init__(
        self, 
        encoder, 
        it_module, 
        decoder, 
        device: torch.device,
        bart_checkpoint: str = "facebook/bart-large-cnn"
    ):
        """
        Initialize evaluator with model components.
        
        Args:
            encoder: Emotion Contagion Encoder
            it_module: Intent Twice Module
            decoder: Response Decoder (with or without loss wrapper)
            device: Torch device
            bart_checkpoint: BART model checkpoint for BARTScore
        """
        self.device = device
        self.encoder = encoder
        self.it_module = it_module
        self.decoder = decoder
        
        # Initialize components
        self.text_processor = TextProcessor(encoder.word_embedding)
        self.inference_engine = InferenceEngine(encoder, it_module, decoder, self.text_processor, device)
        self.metrics_calculator = MetricsCalculator(device=str(device))
        self.bart_checkpoint = bart_checkpoint

        self.pad_id = TOKENIZER.pad_token_id
        self.bos_id = TOKENIZER.cls_token_id 
        self.eos_id = TOKENIZER.sep_token_id
    
    def prepare_evaluation_data(self):
        return prepare_evaluation_data()
    
    def evaluate(
        self,
        eval_samples: List[EvaluationSample],
        max_len: int = 32,
        use_pointer: bool = False,
        batch_size_metrics: int = 8,
        **generation_kwargs
    ) -> EvaluationResults:
        """
        Run complete evaluation on the provided samples.
        
        Args:
            eval_samples: List of evaluation samples
            max_len: Maximum generation length
            use_pointer: Whether to use pointer-generator mechanism
            batch_size_metrics: Batch size for BARTScore computation
            **generation_kwargs: Additional generation parameters
            
        Returns:
            EvaluationResults object with all metrics
        """
        logger.info(f"Starting evaluation on {len(eval_samples)} samples")
        start_time = time.time()
        
        # Generate responses
        hyps, refs = self.inference_engine.batch_generate(
            eval_samples, 
            max_len=max_len, 
            use_pointer=use_pointer,
            **generation_kwargs
        )
        
        generation_time = time.time() - start_time
        
        # Compute BLEU metrics
        bleu_scores = self.metrics_calculator.compute_bleu_metrics(hyps, refs)
        
        # Compute BARTScore
        bart_score = self.metrics_calculator.compute_bart_score(
            hyps, refs, 
            checkpoint=self.bart_checkpoint,
            batch_size=batch_size_metrics
        )
        
        # Calculate statistics
        hyp_lengths = [len(h.split()) for h in hyps]
        ref_lengths = [len(r.split()) for r in refs]
        
        results = EvaluationResults(
            bleu_1=bleu_scores["B-1"],
            bleu_2=bleu_scores["B-2"],
            bleu_3=bleu_scores["B-3"],
            bleu_4=bleu_scores["B-4"],
            bleu_overall=bleu_scores["BLEU"],
            bart_score=bart_score,
            brevity_penalty=bleu_scores["BP"],
            num_samples=len(eval_samples),
            avg_hyp_length=sum(hyp_lengths) / len(hyp_lengths) if hyp_lengths else 0,
            avg_ref_length=sum(ref_lengths) / len(ref_lengths) if ref_lengths else 0,
            generation_time=generation_time
        )
        
        logger.info(f"Evaluation completed in {generation_time:.2f}s")
        return results
    
    def save_results(self, results: EvaluationResults, output_path: str, hyps: List[str] = None, refs: List[str] = None):
        """
        Save evaluation results to file.
        
        Args:
            results: EvaluationResults object
            output_path: Output file path
            hyps: Optional list of hypotheses
            refs: Optional list of references
        """
        output_data = {
            "metrics": {
                "BLEU-1": results.bleu_1 / 100,
                "BLEU-2": results.bleu_2 / 100,
                "BLEU-3": results.bleu_3,
                "BLEU-4": results.bleu_4,
                "BLEU": results.bleu_overall,
                "BARTScore": results.bart_score,
                "BP": results.brevity_penalty
            },
            "statistics": {
                "num_samples": results.num_samples,
                "avg_hyp_length": results.avg_hyp_length,
                "avg_ref_length": results.avg_ref_length,
                "generation_time": results.generation_time
            },
            "config": {
                "bart_checkpoint": self.bart_checkpoint,
                "device": str(self.device)
            }
        }
        
        if hyps and refs:
            output_data["examples"] = [
                {"hypothesis": h, "reference": r} 
                for h, r in zip(hyps[:10], refs[:10])  # Save first 10 examples
            ]
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {output_path}")


def print_results(results: EvaluationResults, title: str = "Evaluation Results"):
    """
    Pretty print evaluation results.
    
    Args:
        results: EvaluationResults object
        title: Title for the results
    """
    print(f"\n{'='*50}")
    print(f"{title:^50}")
    print(f"{'='*50}")
    print(f"Samples:      {results.num_samples}")
    print(f"Gen Time:     {results.generation_time:.2f}s")
    print(f"{'='*50}")
    print(f"BLEU Metrics:")
    print(f"  BLEU-1:     {results.bleu_1:.2f}%")
    print(f"  BLEU-2:     {results.bleu_2:.2f}%")
    print(f"  BLEU-3:     {results.bleu_3:.2f}%")
    print(f"  BLEU-4:     {results.bleu_4:.2f}%")
    print(f"  BLEU:       {results.bleu_overall:.2f}")
    print(f"  BP:         {results.brevity_penalty:.3f}")
    print(f"{'='*50}")
    print(f"BARTScore:    {results.bart_score:.4f}")
    print(f"{'='*50}")
    print(f"Length Stats:")
    print(f"  Avg Hyp:    {results.avg_hyp_length:.1f} words")
    print(f"  Avg Ref:    {results.avg_ref_length:.1f} words")
    print(f"{'='*50}")


@dataclass
class EvaluationConfig:
    """Configuration for training evaluation."""
    eval_every_epochs: Optional[int] = 1
    eval_every_steps: Optional[int] = None
    eval_at_end: bool = True
    eval_data_path: Optional[str] = None
    max_eval_samples: int = 100
    max_gen_length: int = 32
    use_pointer: bool = False
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    compute_bleu: bool = True
    compute_bart_score: bool = True
    compute_informativeness: bool = True  # when True, also compute perplexity & distinct
    bart_checkpoint: str = "facebook/bart-large-cnn"
    save_results: bool = True
    results_dir: str = "output"
    log_examples: bool = True
    num_examples: int = 5
    eval_batch_size: int = 1
    disable_tqdm: bool = False


class TrainingEvaluator:
    """
    Evaluation wrapper for training integration.
    
    This class provides a training-friendly interface around ReflectDiffuEvaluator
    with features like:
    - Scheduled evaluation (every N epochs/steps)
    - Result tracking and logging
    - Status monitoring
    """
    
    def __init__(self, model, config: EvaluationConfig):
        """
        Initialize training evaluator.
        
        Args:
            model: ReflectDiffu model
            config: EvaluationConfig object
        """
        self.model = model
        self.config = config
        self.evaluator = None
        self.eval_data = None
        
        # Tracking
        self.best_bleu4 = 0.0
        # Track best bart score (maximize) and best perplexity (minimize)
        self.best_bart_score = float('-inf')
        self.best_perplexity = None  # will be set after first informativeness evaluation
        self.evaluation_history = []
        self.num_eval_samples = 0
        
        # Scheduling
        self.last_eval_epoch = 0
        self.last_eval_step = 0
        
        # Results directory
        self.results_dir = Path(config.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"TrainingEvaluator initialized with config: {config}")
    
    def prepare_evaluation_data(self):
        return prepare_evaluation_data()
    
    def initialize(self):
        """Initialize the evaluator and prepare evaluation data."""
        self.evaluator = ReflectDiffuEvaluator(
            encoder=self.model.encoder,
            it_module=self.model.it_module,
            decoder=self.model.decoder_with_loss,
            device=self.model.device,
            bart_checkpoint=self.config.bart_checkpoint
        )
        
        self.evaluator_inform = InformativenessEvaluator(
            encoder=self.model.encoder,
            it_module=self.model.it_module,
            decoder=self.model.decoder_with_loss,
            device=self.model.device
        )
        
        self.eval_data = self.prepare_evaluation_data()
        self.num_eval_samples = len(self.eval_data)
        
        logger.info(f"Evaluation initialized with {self.num_eval_samples} samples")
    
    def should_evaluate(self, epoch: int, step: int) -> bool:
        """
        Check if evaluation should be performed.
        Returns:
            True if evaluation should be performed
        """
        if not self.evaluator or not self.eval_data:
            
            return False
        
        # Check step-based evaluation
        if self.config.eval_every_steps is not None:
            if step - self.last_eval_step >= self.config.eval_every_steps:
                return True
        
        # Check epoch-based evaluation
        elif self.config.eval_every_epochs is not None:
            if epoch - self.last_eval_epoch >= self.config.eval_every_epochs:
                return True
        
        return False
    
    def evaluate(self) -> Optional[Tuple[EvaluationResults, Optional[Any]]]:
        """
        Run evaluation and return results.
        
        Returns:
            EvaluationResults object or None if evaluation failed
        """
        if not self.evaluator or not self.eval_data:
            logger.warning("Evaluator or eval data not available")
            return None
        
        relevance_results = self.evaluator.evaluate(
            eval_samples=self.eval_data,
            max_len=self.config.max_gen_length,
            use_pointer=self.config.use_pointer,
            batch_size_metrics=self.config.eval_batch_size,
            temperature=self.config.temperature,
            top_k=self.config.top_k,
            top_p=self.config.top_p
        )
        info_results = None
        if self.config.compute_informativeness:
            info_results = self.evaluator_inform.evaluate_from_file(
                data_path=self.config.eval_data_path,
                max_samples=self.config.max_eval_samples
            )
            self.last_informativeness = info_results

        # Update best metrics (maximize BLEU-4, minimize perplexity)
        if relevance_results.bleu_4 > self.best_bleu4:
            self.best_bleu4 = relevance_results.bleu_4
        # Track best bart score if available
        if info_results is not None:
            if not hasattr(self, 'best_perplexity') or self.best_perplexity is None:
                self.best_perplexity = info_results.perplexity
            else:
                if info_results.perplexity < self.best_perplexity:
                    self.best_perplexity = info_results.perplexity

        self.evaluation_history.append(relevance_results)
        return relevance_results, info_results
    
    def log_results(self, relevance_results: EvaluationResults, info_results: Optional[Any], epoch: int, step: int):
        """
        Log evaluation results.
        
        Args:
            results: EvaluationResults object
            epoch: Current epoch
            step: Current step
        
        """
        print(f"\n📊 Evaluation Results (Epoch {epoch}, Step {step}):")
        print(f"  BLEU-4: {relevance_results.bleu_4:.2f}% (Best: {self.best_bleu4:.2f}%)")
        print(f"  BLEU: {relevance_results.bleu_overall:.2f}")
        print(f"  Samples: {relevance_results.num_samples}")
        print(f"  Gen Time: {relevance_results.generation_time:.2f}s")
        if info_results is not None:
            print(f"  PPL: {info_results.perplexity:.2f} (Best: {getattr(self, 'best_perplexity', float('nan')):.2f}) | Dist-1: {info_results.distinct_1:.3f} | Dist-2: {info_results.distinct_2:.3f}")
        
        # Log examples if requested
        if self.config.log_examples and self.num_eval_samples > 0:
            print(f"\n📝 Example Generations ({self.config.num_examples} samples):")
            try:
                # Generate examples for logging
                hyps, refs = self.evaluator.inference_engine.batch_generate(
                    self.eval_data[:self.config.num_examples],
                    max_len=self.config.max_gen_length,
                    use_pointer=self.config.use_pointer,
                    temperature=self.config.temperature,
                    top_k=self.config.top_k,
                    top_p=self.config.top_p
                )
                
                for idx, (h, r) in enumerate(zip(hyps, refs)):
                    print(f"user input is: {' '.join(self.eval_data[idx].user_tokens)}")
                    print(f"hypothesis is: {h}")
                    print(f"reference is: {r}")
                    print("-" * 20)
                    if idx == 20: break
                    
            except Exception as e:
                logger.warning(f"Failed to generate examples: {e}")
        
        # Save results if requested
        if self.config.save_results:
            results_file = self.results_dir / f"eval_epoch_{epoch}_step_{step}.json"
            try:
                payload = {
                    'epoch': epoch,
                    'step': step,
                    'relevance': {
                        'BLEU-1': relevance_results.bleu_1,
                        'BLEU-2': relevance_results.bleu_2,
                        'BLEU-3': relevance_results.bleu_3,
                        'BLEU-4': relevance_results.bleu_4,
                        'BLEU': relevance_results.bleu_overall,
                        'BARTScore': relevance_results.bart_score,
                        'BP': relevance_results.brevity_penalty,
                        'avg_hyp_length': relevance_results.avg_hyp_length,
                        'avg_ref_length': relevance_results.avg_ref_length,
                        'generation_time': relevance_results.generation_time
                    }
                }
                if info_results is not None:
                    payload['informativeness'] = {
                        'perplexity': info_results.perplexity,
                        'distinct_1': info_results.distinct_1,
                        'distinct_2': info_results.distinct_2,
                        'avg_response_length': info_results.avg_response_length,
                        'valid_responses': info_results.valid_responses
                    }
                with open(results_file, 'w', encoding='utf-8') as f:
                    json.dump(payload, f, indent=2, ensure_ascii=False)
            except Exception as e:
                logger.warning(f"Failed to save combined results: {e}")
        
        # Update tracking
        self.last_eval_epoch = epoch
        self.last_eval_step = step
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current evaluator status.
        
        Returns:
            Dictionary with status information
        """
        return {
            'best_bleu4': self.best_bleu4,
            'best_bart_score': self.best_bart_score,
            'best_perplexity': self.best_perplexity,
            'num_evaluations': len(self.evaluation_history),
            'num_eval_samples': self.num_eval_samples,
            'last_eval_epoch': self.last_eval_epoch,
            'last_eval_step': self.last_eval_step,
            'evaluator_available': self.evaluator is not None,
            'eval_data_available': self.eval_data is not None
        }