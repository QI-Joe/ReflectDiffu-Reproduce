# ERA 模块 (Emotion Reason Annotator) 说明

> 功能定位：ERA 是 ReflectDiffu 框架中为 Emotion-Contagion Encoder 提供词级情感原因 (em/noem) 推理掩码的序列标注子系统。它以 **BERT + 线性分类头 + CRF** 结构实现情感原因词抽取，产出 token 级标签及隐藏表示 \(\tilde{h}\)，用于在上游构建 `ER` 嵌入以及下游与上下文编码器进行 `Attention(H, \tilde{h})` 融合。

---
## 1. 目录结构与文件职责
| 文件 | 主要类/函数 | 核心职责 |
|------|-------------|----------|
| `config.py` | `ERAConfig`, `ERAConfigPresets` | 训练 & 模型超参数；NuNER 基准配置、调试/小显存预设 |
| `data_processor.py` | `EmpathyDataProcessor`, `DialogueSample`, `TokenizedSample`, `ERADataset` | 加载 & LLM 标注（可API / 本地）、数据清洗、IO标签对齐、子词映射、8:1:1 划分、Batch 制作 |
| `era_model.py` | `ERA_BERT_CRF`, `ERAConfig(PretrainedConfig)` | 主模型结构：BERT 编码 + Dropout + 线性层 + CRF 序列约束 + Viterbi 解码 |
| `evaluator.py` | `ERAEvaluator`, `EvaluationResults` | Token 级精确率/召回/F1、宏/微平均、混淆矩阵、错误分析 |
| `trainer.py` | `ERATrainer`, `TrainingState` | 分层学习率、Warmup Scheduler、Early Stopping、Checkpoint、TensorBoard 记录 |
| `__init__.py` | 导出符号 | 模块封装入口 |

---
## 2. 与论文位置的对应
论文 Section 3.2 中的 *Emotion Reason Annotator (ERA)*：
- 产出：词级推理标签序列 `r`（em/noem）与对应隐藏表示 \(\tilde{h} \in \mathbb{R}^{B \times L \times 768}\)
- 作用：让 Emotion-Contagion Encoder 将 `ER`（Reason Embedding）加入 \(E^C\)，并在注意力阶段 `Attention(H, \tilde{h})` 聚焦情感触发词位置；增强情感语义显著性与可解释性。

ERA 训练独立完成后，其推理输出在主训练流程中被复用，不与响应生成/意图模块共享梯度。

---
## 3. 数据处理与标注流水线
### 3.1 输入数据（EmpatheticDialogues）字段（示例 CSV 列）
| 字段 | 说明 |
|------|------|
| `conv_id` | 对话唯一 ID (`hit:xxx_conv:yyy`) |
| `utterance` | 单句文本（轮次上下文中某一方发言） |
| `speaker_idx` | 说话人编号 (用于区分 user / response) |
| `context` | 情感上下文（源自数据集） |
| `prompt` | 主题 / 触发背景 |
| `selfeval` | 自我评价（可选） |

### 3.2 LLM 标注策略
- 用户侧 utterance：送入 LLM（ChatGLM / API Grok）生成 `word:<em/noem>` 序列
- 系统响应侧：全部标注为 `<noem>`（框架假设：情感原因来源于用户情境）
- 解析：分离 tokens / labels，异常则降级为全部 `<noem>`

### 3.3 标签编码方案（仅 IO）
| 标签文本 | 映射 ID | 说明 |
|----------|---------|------|
| `O` / `<noem>` | 0 | 非情感原因词 |
| `EM` / `<em>` | 1 | 情感原因词 |
| `-100` | 忽略 | 子词补位 / 特殊符号（不参与损失） |

### 3.4 子词对齐
步骤：
1. 原始词级 tokens 与标签（IO）
2. HuggingFace tokenizer 获取 `offset_mapping`
3. 第一个覆盖该词的子词 → 继承标签；剩余子词 → `-100`
4. `[CLS] [SEP] [PAD]` → 全部 `-100`

### 3.5 数据划分
- 比例：Train:Valid:Test = 8:1:1
- 随机种子：`ERAConfig.seed`

### 3.6 批处理输出 (DataLoader batch)
| 字段 | 形状 | 说明 |
|------|------|------|
| `input_ids` | `[B, L]` | Token IDs |
| `attention_mask` | `[B, L]` | 1=有效, 0=Pad |
| `labels` | `[B, L]` | 0/1/-100 |
| `token_type_ids` | `[B, L]` (可选) | 分句类型（BERT） |

---
## 4. 模型结构 (ERA_BERT_CRF)
```
输入 (input_ids, attention_mask, labels)
      ↓
BERT 编码器 (可冻结前 N 层, hidden=768)
      ↓
Dropout(p=0.1)
      ↓
Linear 分类头 (768 → 2)
      ↓
CRF 层 (转移约束 + Viterbi 解码)
      ↓
输出: {logits, loss, predictions, hidden_states}
```

### 4.1 输入 / 输出张量形状
| 名称 | 形状 | 含义 |
|------|------|------|
| `input_ids` | `[B,L]` | 子词 ID |
| `attention_mask` | `[B,L]` | 掩码 |
| `hidden_states` | `[B,L,768]` | BERT 最后一层表示（经 dropout） |
| `logits` | `[B,L,2]` | 线性分类输出 |
| `predictions` | `[B,L]` | CRF Viterbi 解码结果 (0/1/-100 pad) |
| `loss` | 标量 | CRF 负对数似然 (reduction="mean") |

### 4.2 关键实现细节
| 机制 | 位置 | 说明 |
|------|------|------|
| 冻结层 | `_freeze_bert_layers` | embeddings + 前 `frozen_layers` encoder blocks 不更新梯度 |
| 分类头初始化 | `_init_weights` | Xavier Uniform + bias=0 |
| CRF 掩码 | `labels != ignore_index & attention_mask` | 忽略补位与特殊子词 |
| Ignore 处理 | `labels==-100 → 0` | 仅为满足 CRF 输入要求（已被 mask） |
| 解码 | `self.crf.decode` | 返回每句有效长度路径 |

---
## 5. 训练流程 (ERATrainer)
### 5.1 优化与调度
| 项 | 设置 |
|----|------|
| 优化器 | AdamW (分组：encoder_lr / head_lr) |
| 学习率 | 编码器 3e-5，头层 5e-5 (默认) |
| Scheduler | 线性 / 余弦，带 warmup_ratio=0.1 |
| 梯度裁剪 | `max_grad_norm=1.0` |
| 累积 | `gradient_accumulation_steps` 支持大有效 batch |

### 5.2 Early Stopping & Checkpoint
| 项 | 机制 |
|----|------|
| 监控指标 | `metric_for_best_model` (默认 F1) |
| 比较方向 | `greater_is_better=True` |
| 耐心值 | `early_stopping_patience=3` |
| 保存策略 | 每 epoch 保存；记录 best_checkpoint |

### 5.3 日志与追踪
- TensorBoard：loss / F1 / 学习率 / 配置文本
- `TrainingState`：记录 `epoch, global_step, best_metric, patience_counter`

### 5.4 单 Epoch 关键伪代码
```python
for step, batch in enumerate(train_dataloader):
    outputs = model(**batch)
    loss = outputs["loss"]/grad_accum
    loss.backward()
    if (step+1)%grad_accum==0:
        clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step(); scheduler.step(); optimizer.zero_grad()
```

---
## 6. 评估 (ERAEvaluator)
### 6.1 指标集合
| 范畴 | 指标 |
|------|------|
| Token 级 | Accuracy / Precision / Recall / F1 |
| 加权 | weighted precision/recall/f1 |
| 宏/微 | macro_* / micro_* |
| 类别细分 | 每类 precision/recall/f1/support |
| 混淆矩阵 | confusion_matrix (2x2) |
| 错误分析 | 统计 FP/FN 模式（扩展字段） |

### 6.2 评估流程
1. `predictions, labels` → flatten & 过滤 ignore_index
2. 计算总体与 per-class 指标
3. 生成混淆矩阵与错误摘要
4. 返回 dict + `EvaluationResults`

---
## 7. 损失函数说明
- 使用 CRF 负对数似然：`loss = -log p(label_sequence | logits)`
- Mask 仅保留有效位置；子词补位 / 特殊 token 不计入
- 无 CRF 情况下可回落到 `CrossEntropyLoss(ignore_index=-100)`（当前默认使用 CRF）

---
## 8. 关键超参数 (config.py)
| 名称 | 默认 | 说明 |
|------|------|------|
| `bert_model` | `google-bert/bert-base-uncased` | 预训练模型名称 |
| `frozen_layers` | 6 | 冻结前 N 层编码器 |
| `batch_size` | 48 | NuNER 推荐 |
| `encoder_lr` | 3e-5 | 主体学习率 |
| `head_lr` | 5e-5 | 分类头更高学习率 |
| `warmup_ratio` | 0.1 | Scheduler 预热占比 |
| `num_epochs` | 10 | 最大轮次 |
| `dropout_rate` | 0.1 | 分类头 Dropout |
| `max_length` | 512 | 最大序列长度 |
| `ignore_index` | -100 | 损失忽略标签 |
| `use_crf` | True | 是否启用 CRF |
| `early_stopping_patience` | 3 | 早停耐心值 |

---
## 9. 输入输出一览（端到端）
| 阶段 | 输入 | 输出 |
|------|------|------|
| LLM 标注 | 原始 utterance | `tokens` + `em/noem` 标签 |
| Tokenize 对齐 | `tokens, labels` | `input_ids, attention_mask, labels(-100填充)` |
| 模型前向 | batch 张量 | `logits, predictions, loss, hidden_states` |
| 推理导出 | `predictions` | 词级标签序列（用于 Emotion-Contagion） |

---
## 10. 常见问题与排查
| 现象 | 排查建议 |
|------|----------|
| F1 长期低迷 | 检查 LLM 标注质量；标签是否错位；冻结层数是否过多 |
| loss 不下降 | 学习率过高/过低；确认梯度未被全部冻结；CRF mask 是否正确 |
| GPU OOM | 降低 `max_length` / batch_size；关闭 fp16=False 改为 True |
| 推理速度慢 | 冻结层可减少反向计算；评估时禁用不必要日志 |
| 全部预测为 O | 检查标签映射 / 对齐逻辑；确认 `num_labels=2` 一致 |
| CRF loss NaN | 是否存在全 -100 行；attention_mask 是否全 0 |

---
## 11. 扩展与改进建议
| 方向 | 说明 |
|------|------|
| BIO 扩展 | 增加 `B-EM/I-EM`，需改 `num_labels` + CRF 转移矩阵约束 |
| 软蒸馏 | 结合多 LLM 标注概率平均 / 投票融合 |
| 噪声鲁棒 | 引入标签平滑或对比学习辅助 (token rep) |
| 多任务 | 共享 BERT 主干 + 额外意图 / 情感分类头 |
| 轻量化 | 换用 DistilBERT / TinyBERT，保留 CRF 层 |
| 混合提示 | 将 prompt 拼接到输入，增强上下文判别能力 |

---
## 12. 与 Emotion-Contagion Encoder 的接口
| 输出 | 用途 |
|------|------|
| `predictions` (0/1) | 转换为 `<noem>/<em>` → ReasonEmbedding(label_ids) |
| `hidden_states` | 可选：对齐投影为 \(\tilde{h}\)，供 `Attention(H, \tilde{h})` 使用 |

---
## 13. 最小化使用示例
```python
from era import ERAConfig, ERA_BERT_CRF, ERATrainer, ERAEvaluator
from era.data_processor import EmpathyDataProcessor

config = ERAConfig()
# 构建数据 (此处需先准备 LLM 标注样本列表 dialogues)
# dialogues = ...  # List[DialogueSample]
# tokenized = processor.tokenize_and_align_labels(dialogues)
# train_samples, valid_samples, test_samples = processor.split_data(tokenized)
# train_loader = processor.create_dataloader(train_samples, batch_size=config.batch_size)

model = ERA_BERT_CRF.create_or_from_config?  # 实际使用: create_era_model 或直接 ERA_BERT_CRF(ERAConfig(...))
# trainer = ERATrainer(model, config, train_loader, valid_loader, test_loader, ERAEvaluator(config))
# trainer.train()
```

---
## 14. 快速摘要
ERA 提供：结构化、可复现、低耦合的情感原因词抽取；输出的 `em/noem` 标签与隐藏表示为主模型的情感传染建模提供显式注意力引导与嵌入偏置，实现情感-意图反射机制的第一步。

---
