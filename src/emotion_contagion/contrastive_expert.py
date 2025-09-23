from torch import nn
from typing import List, Dict, Optional, Literal, Tuple
import torch
import math
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from src.tokenizer_loader import _ids_to_text_list
PolarityV = Literal['nneg', 'nneu', 'npos']

def decide_batch_polarity_via_vader(
    texts: List[str],
    pos_thresh: float = 0.05,
    neg_thresh: float = -0.05
) -> Tuple[PolarityV, Dict[str, int], Dict[str, float]]:
    analyzer = SentimentIntensityAnalyzer()
    
    if isinstance(texts, torch.Tensor):
        texts = _ids_to_text_list(texts)

    nneg = nneu = npos = 0
    conf_neg = conf_neu = conf_pos = 0.0

    for t in texts:
        scores = analyzer.polarity_scores(t or "")
        comp = scores.get("compound", 0.0)

        if comp >= pos_thresh:
            npos += 1
            conf_pos += comp  # 正向置信度越高越好
        elif comp <= neg_thresh:
            nneg += 1
            conf_neg += -comp  # 负向置信度用 |compound|
        else:
            nneu += 1
            # 中性置信度：离阈值越“居中”越中性。这里简单用 0.05 - |comp| 的非负边际
            conf_neu += max(0.0, pos_thresh - abs(comp))

    counts = {'nneg': nneg, 'nneu': nneu, 'npos': npos}
    conf_sums = {'nneg': conf_neg, 'nneu': conf_neu, 'npos': conf_pos}

    # 选最大计数类
    max_count = max(counts.values())
    winners = [k for k, v in counts.items() if v == max_count]

    if len(winners) == 1:
        v = winners[0]
    else:
        # 平局：
        v = "nneu"

    return v, counts, conf_sums

class CONExpert(nn.Module):
    
    def __init__(self, input_dim: int, emo_dim: int=32, alpha: float = 0.5):
        super().__init__()
        self.Wpos = nn.Linear(emo_dim, emo_dim) # CxC
        self.Wneg = nn.Linear(emo_dim, emo_dim)
        # Project input features to emotion dimension
        self.proj = nn.Linear(input_dim, emo_dim) 
        self.alpha = alpha
        
        nn.init.xavier_uniform_(self.Wpos.weight)
        nn.init.xavier_uniform_(self.Wneg.weight)
        nn.init.xavier_uniform_(self.proj.weight)
        
    def forward(self, Q: torch.Tensor, batch_text: List[str]): # shape of Q: [B, D]
        """
        Forward pass for the Contrastive Expert module."""
        
        Z = self.proj(Q)  # Project Q to emotion dimension [B, D] -> [B, emo_dim]
        scale = 1.0/math.sqrt(Z.size(-1))
        
        v, counts, conf_sums = decide_batch_polarity_via_vader(batch_text)
        
        if v=="nneg":
            logit = self.Wneg(Z) * scale
        elif v=="npos":
            logit = self.Wpos(Z) * scale
        else:
            logit = self.alpha * self.Wpos(Z) + (1-self.alpha) * self.Wneg(Z)
        
        p = torch.softmax(logit, dim=-1)
        
        return p
        
