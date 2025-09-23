from transformers import AutoTokenizer
import torch
from typing import List, Tuple, Dict, Literal, Optional

from pathlib import Path
from transformers import AutoTokenizer, PreTrainedTokenizerBase

_TOKENIZER: Optional[PreTrainedTokenizerBase] = None
DEFAULT_MODEL = "bert-base-uncased"
LOCAL_DIR = Path("./configs/bert_tokenizer")

def _is_valid_local_tokenizer_dir(path: Path) -> bool:
    # 依据常见文件判断，本地是否已含可加载的 tokenizer
    if not path.exists() or not path.is_dir():
        return False
    expected_any = [
        "tokenizer.json",            # 新格式常见
        "tokenizer_config.json",     # 通用配置
        "vocab.txt",                 # BERT 字表
        "special_tokens_map.json",
    ]
    existing = {p.name for p in path.iterdir() if p.is_file()}
    return any(name in existing for name in expected_any)

def get_tokenizer(name: str = DEFAULT_MODEL) -> PreTrainedTokenizerBase:
    global _TOKENIZER
    if _TOKENIZER is not None:
        return _TOKENIZER

    use_local = _is_valid_local_tokenizer_dir(LOCAL_DIR)
    load_from = str(LOCAL_DIR) if use_local else name

    _TOKENIZER = AutoTokenizer.from_pretrained(load_from)

    if not use_local:
        LOCAL_DIR.mkdir(parents=True, exist_ok=True)
        _TOKENIZER.save_pretrained(str(LOCAL_DIR))

    return _TOKENIZER

def _ids_to_text_list(batch_ids: torch.Tensor) -> List[str]:
    """
    Convert a (B, L) tensor of token ids into a list of decoded strings.
    Assumes unified HF tokenizer. Trims at first pad if pad_token_id exists.
    """
    tokenizer = get_tokenizer()
    pad_id = tokenizer.pad_token_id
    texts: List[str] = []
    for seq in batch_ids.tolist():
        if pad_id is not None and pad_id in seq:
            first_pad = seq.index(pad_id)
            seq = seq[:first_pad]
        # decode merges WordPieces and removes specials
        txt = tokenizer.decode(seq, skip_special_tokens=True).strip()
        texts.append(txt)
    return texts

def ids_to_words(ids, tokenizer):
    toks = tokenizer.convert_ids_to_tokens(ids)
    words = []
    buf = []
    for t in toks:
        if t in tokenizer.all_special_tokens:
            continue
        if t.startswith("##"):
            buf.append(t[2:])
        else:
            if buf:
                words.append("".join(buf))
            buf = [t]
    if buf:
        words.append("".join(buf))
    return words