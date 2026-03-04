# re_cnn/data.py
import json
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable

PAD = 0
UNK = 1

@dataclass
class REExample:
    tokens: List[str]
    chem_span: Tuple[int, int]   # [start, end) token indices
    dis_span: Tuple[int, int]
    label: str                  # "CID" or "NO_RELATION"
    meta: Dict

def read_re_jsonl(path: str) -> List[REExample]:
    exs: List[REExample] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            obj = json.loads(ln)
            tokens = obj["tokens"]
            chem_span = tuple(obj["chem_span"])
            dis_span = tuple(obj["dis_span"])
            label = obj.get("label", "NO_RELATION")
            meta = {k: v for k, v in obj.items() if k not in ["tokens", "chem_span", "dis_span", "label"]}
            exs.append(REExample(tokens=tokens, chem_span=chem_span, dis_span=dis_span, label=label, meta=meta))
    return exs

def load_embeddings_vocab(emb_path: str, max_words: int = None) -> Tuple[Dict[str,int], np.ndarray]:
    """
    读取 embeddings/biowordvec_bc5_subset.txt
    生成 word2id 和 embedding matrix。
    约定：
      id=0 PAD 向量全0
      id=1 UNK 随机小值
    """
    words = []
    vecs = []
    dim = None

    with open(emb_path, "r", encoding="utf-8", errors="ignore") as f:
        for i, ln in enumerate(f):
            parts = ln.rstrip("\n").split()
            if len(parts) <= 2:
                continue
            w = parts[0]
            v = np.asarray(parts[1:], dtype=np.float32)
            if dim is None:
                dim = v.shape[0]
            if v.shape[0] != dim:
                continue
            words.append(w)
            vecs.append(v)
            if max_words is not None and len(words) >= max_words:
                break

    if dim is None:
        raise RuntimeError(f"Failed to read embeddings from: {emb_path}")

    word2id = {"<PAD>": PAD, "<UNK>": UNK}
    for w in words:
        if w not in word2id:
            word2id[w] = len(word2id)

    emb = np.zeros((len(word2id), dim), dtype=np.float32)
    emb[UNK] = np.random.normal(0, 0.05, size=(dim,)).astype(np.float32)
    for w, v in zip(words, vecs):
        emb[word2id[w]] = v

    return word2id, emb

def _dist(i: int, span: Tuple[int,int]) -> int:
    s, e = span
    # 用 span center 更平滑：center = (s+e-1)/2
    c = (s + (e - 1)) // 2
    return i - c

def make_position_ids(tokens_len: int, span: Tuple[int,int], max_dist: int) -> np.ndarray:
    """
    把距离 clip 到 [-max_dist, max_dist]，再 shift 到 [0, 2*max_dist]
    """
    out = np.zeros((tokens_len,), dtype=np.int32)
    for i in range(tokens_len):
        d = _dist(i, span)
        d = max(-max_dist, min(max_dist, d))
        out[i] = d + max_dist
    return out

def vectorize(
    exs: List[REExample],
    word2id: Dict[str,int],
    max_len: int = 256,
    max_dist: int = 64,
    label2id: Dict[str,int] = None
):
    if label2id is None:
        label2id = {"NO_RELATION": 0, "CID": 1}

    Xw = np.full((len(exs), max_len), PAD, dtype=np.int32)
    Xp1 = np.full((len(exs), max_len), max_dist, dtype=np.int32)  # 0 dist -> shift=max_dist
    Xp2 = np.full((len(exs), max_len), max_dist, dtype=np.int32)
    y = np.zeros((len(exs),), dtype=np.int32)

    for n, ex in enumerate(exs):
        toks = ex.tokens[:max_len]
        L = len(toks)

        for i, t in enumerate(toks):
            Xw[n, i] = word2id.get(t, UNK)

        p1 = make_position_ids(L, ex.chem_span, max_dist)
        p2 = make_position_ids(L, ex.dis_span, max_dist)
        Xp1[n, :L] = p1
        Xp2[n, :L] = p2

        y[n] = label2id.get(ex.label, 0)

    return (Xw, Xp1, Xp2), y, label2id
