import argparse
import os
import torch
import torch.nn.functional as F
from torch import nn
import time
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import logging

from evaluation import ReflectDiffuEvaluator, EvaluationResults, EvaluationSample
EVALUATION_AVAILABLE = True

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """评估配置类 - 按照设计文档实现"""
    # 评估频率控制
    eval_every_epochs: int = 1          # 每N个epoch评估一次
    eval_every_steps: Optional[int] = None  # 每N个step评估一次(优先级高)
    eval_at_end: bool = True           # 训练结束时评估
    
    # 评估数据控制
    eval_data_path: Optional[str] = None  # 评估数据路径
    max_eval_samples: int = 100        # 最大评估样本数
    
    # 生成参数
    max_gen_length: int = 32           # 最大生成长度
    use_pointer: bool = False          # 是否使用指针网络
    temperature: float = 1.0           # 温度参数
    top_k: int = 0                     # Top-k 采样
    top_p: float = 1.0                 # Top-p 采样
    
    # 指标计算
    compute_bleu: bool = True          # 计算BLEU指标
    compute_bart_score: bool = True    # 计算BARTScore
    bart_checkpoint: str = "facebook/bart-large-cnn"
    
    # 输出控制
    save_results: bool = True          # 保存结果
    results_dir: str = "evaluation_results"
    log_examples: bool = True          # 记录生成样例
    num_examples: int = 5              # 记录样例数量
    
    # 性能控制
    eval_batch_size: int = 1           # 评估批量大小
    disable_tqdm: bool = False         # 禁用进度条
    
    def __post_init__(self):
        """配置验证"""
        if self.eval_every_epochs <= 0:
            raise ValueError("eval_every_epochs must be positive")
        if self.eval_every_steps is not None and self.eval_every_steps <= 0:
            raise ValueError("eval_every_steps must be positive")
        if self.max_eval_samples <= 0:
            raise ValueError("max_eval_samples must be positive")


class EvaluationLogger:
    """评估结果日志管理器"""
    
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.log_dir / "metrics.jsonl"
        self.examples_file = self.log_dir / "examples.txt"
        
    def log_metrics(self, results: "EvaluationResults", epoch: int, step: Optional[int] = None):
        """记录指标到文件"""
        log_entry = {
            "epoch": epoch,
            "step": step,
            "timestamp": time.time(),
            "metrics": {
                "BLEU-1": results.bleu_1,
                "BLEU-2": results.bleu_2,
                "BLEU-3": results.bleu_3,
                "BLEU-4": results.bleu_4,
                "BLEU": results.bleu_overall,
                "BARTScore": results.bart_score,
                "BP": results.brevity_penalty,
                "num_samples": results.num_samples,
                "avg_hyp_length": results.avg_hyp_length,
                "avg_ref_length": results.avg_ref_length,
                "generation_time": results.generation_time
            }
        }
        
        with open(self.metrics_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            
        logger.info(f"Epoch {epoch}: BLEU-4={results.bleu_4:.2f}%, BARTScore={results.bart_score:.4f}")
    
    def log_examples(self, hyps: List[str], refs: List[str], num_examples: int, epoch: int):
        """记录生成样例"""
        with open(self.examples_file, "a", encoding="utf-8") as f:
            f.write(f"\n=== Epoch {epoch} Examples ===\n")
            for i in range(min(num_examples, len(hyps), len(refs))):
                f.write(f"Example {i+1}:\n")
                f.write(f"  Hypothesis: {hyps[i]}\n")
                f.write(f"  Reference:  {refs[i]}\n")
                f.write("\n")


class TrainingEvaluator:
    """训练期间的评估管理器 - 按照设计文档实现"""
    
    def __init__(self, model, config: EvaluationConfig):
        """
        初始化训练评估器
        
        Args:
            model: ReflectDiffu 模型实例
            config: 评估配置
        """
        if not EVALUATION_AVAILABLE:
            raise RuntimeError("Evaluation module not available. Cannot create TrainingEvaluator.")
        
        self.model = model
        self.config = config
        self.evaluator = None  # ReflectDiffuEvaluator 实例，延迟初始化
        self.eval_data = None  # 评估数据，延迟加载
        self.logger = None     # 日志记录器，延迟初始化
        self.initialized = False
        
        # 追踪状态
        self.last_eval_epoch = -1
        self.last_eval_step = -1
        self.best_bleu4 = 0.0
        self.best_bart_score = float('-inf')
        
    def initialize(self):
        """延迟初始化评估器，复用训练模型组件"""
        if self.initialized:
            return
            
        logger.info("Initializing training evaluator...")
        
        # 1. 创建评估器，直接复用训练模型组件
        self.evaluator = ReflectDiffuEvaluator(
            encoder=self.model.encoder,           # 直接复用训练模型的编码器
            it_module=self.model.it_module,       # 直接复用Intent Twice模块
            decoder=self.model.decoder_with_loss, # 直接复用解码器
            device=self.model.device,             # 使用相同设备
            bart_checkpoint=self.config.bart_checkpoint
        )
        
        # 2. 加载评估数据
        self._load_eval_data()
        
        # 3. 初始化日志记录器
        if self.config.save_results:
            self.logger = EvaluationLogger(self.config.results_dir)
        
        self.initialized = True
        logger.info("Training evaluator initialized successfully")
            
    
    def _load_eval_data(self):
        if self.config.eval_data_path:
            # 使用专门的评估数据
            eval_raw_data = self.model.dp.load_data(self.config.eval_data_path)
            logger.info(f"Loaded evaluation data from {self.config.eval_data_path}")
        else:
            # 复用训练数据作为评估数据
            eval_raw_data = self.model.raw_data
            logger.info("Using training data for evaluation")
        
        # 准备评估样本
        self.eval_data = self.evaluator.prepare_evaluation_data()
        
        logger.info(f"Prepared {len(self.eval_data)} evaluation samples")
            
    
    def should_evaluate(self, epoch: int, step: Optional[int] = None) -> bool:
        """判断是否应该进行评估"""
        if not self.initialized or not self.eval_data:
            return False
        
        # 优先检查步数评估
        if self.config.eval_every_steps is not None and step is not None:
            if step > 0 and step % self.config.eval_every_steps == 0 and step != self.last_eval_step:
                return True
        
        # 检查轮次评估
        if epoch > 0 and epoch % self.config.eval_every_epochs == 0 and epoch != self.last_eval_epoch:
            return True
        
        return False
    
    def evaluate(self, **kwargs) -> Optional["EvaluationResults"]:
        """执行评估，返回结果"""
        if not self.initialized or not self.eval_data:
            logger.warning("Evaluator not initialized or no evaluation data")
            return None
        
        try:
            logger.info(f"Starting evaluation on {len(self.eval_data)} samples...")
            
            # 合并生成参数
            generation_params = {
                'max_len': self.config.max_gen_length,
                'use_pointer': self.config.use_pointer,
                'temperature': self.config.temperature,
                'top_k': self.config.top_k,
                'top_p': self.config.top_p,
                'batch_size_metrics': self.config.eval_batch_size,
                **kwargs  # 允许临时覆盖参数
            }
            
            # 执行评估
            results = self.evaluator.evaluate(self.eval_data, **generation_params)
            
            # 更新最佳结果
            if results.bleu_4 > self.best_bleu4:
                self.best_bleu4 = results.bleu_4
            if results.bart_score > self.best_bart_score:
                self.best_bart_score = results.bart_score
            
            logger.info(f"Evaluation completed: BLEU-4={results.bleu_4:.2f}%, BARTScore={results.bart_score:.4f}")
            return results
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return None
    
    def log_results(self, results: "EvaluationResults", epoch: int, step: Optional[int] = None):
        """记录评估结果"""
        if results is None:
            return
            
        # 更新追踪状态
        if step is not None:
            self.last_eval_step = step
        self.last_eval_epoch = epoch
        
        # 记录到日志文件
        if self.logger:
            self.logger.log_metrics(results, epoch, step)
            
            # 记录生成样例
            if self.config.log_examples:
                try:
                    # 生成少量样例用于日志记录
                    hyps, refs = self.evaluator.inference_engine.batch_generate(
                        self.eval_data[:self.config.num_examples],
                        max_len=self.config.max_gen_length,
                        use_pointer=self.config.use_pointer
                    )
                    self.logger.log_examples(hyps, refs, self.config.num_examples, epoch)
                except Exception as e:
                    logger.warning(f"Failed to log examples: {e}")
        
        # 控制台输出
        self._print_results(results, epoch, step)
    
    def _print_results(self, results: "EvaluationResults", epoch: int, step: Optional[int] = None):
        """控制台输出评估结果"""
        step_info = f", Step {step}" if step is not None else ""
        print(f"\n{'='*60}")
        print(f"📊 Evaluation Results - Epoch {epoch}{step_info}")
        print(f"{'='*60}")
        print(f"Samples:      {results.num_samples}")
        print(f"Gen Time:     {results.generation_time:.2f}s")
        print(f"{'='*60}")
        print(f"BLEU Metrics:")
        print(f"  BLEU-1:     {results.bleu_1:.2f}%")
        print(f"  BLEU-2:     {results.bleu_2:.2f}%")
        print(f"  BLEU-3:     {results.bleu_3:.2f}%")
        print(f"  BLEU-4:     {results.bleu_4:.2f}% (Best: {self.best_bleu4:.2f}%)")
        print(f"  BLEU:       {results.bleu_overall:.2f}")
        print(f"  BP:         {results.brevity_penalty:.3f}")
        print(f"{'='*60}")
        print(f"BARTScore:    {results.bart_score:.4f} (Best: {self.best_bart_score:.4f})")
        print(f"{'='*60}")
        print(f"Length Stats:")
        print(f"  Avg Hyp:    {results.avg_hyp_length:.1f} words")
        print(f"  Avg Ref:    {results.avg_ref_length:.1f} words")
        print(f"{'='*60}\n")
    
    def get_status(self) -> Dict[str, Any]:
        """获取评估器状态信息"""
        return {
            "initialized": self.initialized,
            "num_eval_samples": len(self.eval_data) if self.eval_data else 0,
            "last_eval_epoch": self.last_eval_epoch,
            "last_eval_step": self.last_eval_step,
            "best_bleu4": self.best_bleu4,
            "best_bart_score": self.best_bart_score,
            "config": self.config
        }