import os, json, re
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

# --- NER repo imports ---
# Make imports work no matter where you run the script from.
import sys
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
NER_REPO = os.path.join(PROJECT_ROOT, "NER_biLSTM-CNN-master")
if NER_REPO not in sys.path:
    sys.path.insert(0, NER_REPO)

import preprocessing  # from NER_biLSTM-CNN-master/preprocessing.py
import architecture   # from NER_biLSTM-CNN-master/architecture.py

from tensorflow import keras


# ===== paths =====
TRAIN_CONLL = "data/processed/bc5cdr_ner_sentence_train.txt"
DEV_CONLL   = "data/processed/bc5cdr_ner_sentence_dev.txt"
TEST_CONLL  = "data/processed/bc5cdr_ner_sentence_test.txt"

EMB_PATH    = "embeddings/biowordvec_bc5_subset.txt"
MODEL_PATH  = os.path.join(NER_REPO, "models", "bilstm_cnn_bc5_sentence.keras")

INDEX_TEST  = "data/processed/bc5cdr_sentence_index_test.jsonl"
PUB_TEST    = "data/origin/CDR_TestSet.PubTator.txt"

OUT_DIR = "pipeline_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_TEST_PRED_ENT = os.path.join(OUT_DIR, "test_pred_entities.jsonl")
OUT_RE_TEST_PRED  = os.path.join(OUT_DIR, "re_test_pred.tsv")


# ===== PubTator parsing with mesh + CID =====
def parse_pubtator_full(path: str) -> Dict[str, Any]:
    """
    Return pmid -> {title, abstract, text, entities, cid_pairs}
    entities: list of {start,end,type,mesh}
    cid_pairs: set((chem_mesh, dis_mesh))
    """
    docs: Dict[str, Any] = {}
    with open(path, "r", encoding="utf-8") as f:
        block = []
        for ln in f:
            ln = ln.rstrip("\n")
            if ln.strip() == "":
                if block:
                    _parse_block(block, docs)
                    block = []
            else:
                block.append(ln)
        if block:
            _parse_block(block, docs)
    return docs


def _parse_block(lines: List[str], docs: Dict[str, Any]):
    pmid = None
    title, abstract = "", ""
    ents = []
    cid = set()

    for ln in lines:
        if "|t|" in ln:
            pmid, t = ln.split("|t|", 1)
            title = t
        elif "|a|" in ln:
            pmid2, a = ln.split("|a|", 1)
            pmid = pmid or pmid2
            abstract = a
        else:
            parts = ln.split("\t")
            if len(parts) >= 6 and parts[1].isdigit() and parts[2].isdigit():
                pmid = pmid or parts[0]
                etype = parts[4]  # Chemical / Disease
                mesh  = parts[5]
                ents.append({
                    "start": int(parts[1]),
                    "end": int(parts[2]),
                    "type": etype,
                    "mesh": mesh
                })
            elif len(parts) >= 4 and parts[1] == "CID":
                pmid = pmid or parts[0]
                cid.add((parts[2], parts[3]))

    if pmid is None:
        return
    docs[pmid] = {
        "title": title,
        "abstract": abstract,
        "text": (title + " " + abstract).strip(),
        "entities": ents,
        "cid_pairs": cid
    }


# ===== BIO to entity spans (token-level) =====
def bio_to_spans(labels: List[str]) -> List[Tuple[str,int,int]]:
    """
    Return list of (type, start_tok, end_tok) where end_tok is exclusive.
    Supports B-CHEM/I-CHEM/B-DIS/I-DIS/O.
    """
    spans = []
    i = 0
    while i < len(labels):
        lab = labels[i]
        if lab.startswith("B-"):
            t = lab[2:]
            j = i + 1
            while j < len(labels) and labels[j] == f"I-{t}":
                j += 1
            spans.append((t, i, j))
            i = j
        else:
            i += 1
    # normalize type names
    norm = []
    for t,s,e in spans:
        if t == "CHEM":
            norm.append(("Chemical", s, e))
        elif t == "DIS":
            norm.append(("Disease", s, e))
        else:
            norm.append((t, s, e))
    return norm


def tokens_span_to_char_span(token_offsets: List[Dict[str,int]], s_tok: int, e_tok: int) -> Tuple[int,int]:
    abs_s = token_offsets[s_tok]["start"]
    abs_e = token_offsets[e_tok-1]["end"]
    return abs_s, abs_e


def overlap_len(a: Tuple[int,int], b: Tuple[int,int]) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def match_to_gold_mesh(pred_span: Tuple[int,int], pred_type: str, gold_ents: List[Dict[str,Any]]) -> Optional[str]:
    """
    pred_span in absolute doc char offsets, pred_type 'Chemical'/'Disease'
    Choose gold entity with max overlap of same type.
    """
    best_mesh = None
    best_ov = 0
    for g in gold_ents:
        if g["type"] != pred_type:
            continue
        ov = overlap_len(pred_span, (g["start"], g["end"]))
        if ov > best_ov:
            best_ov = ov
            best_mesh = g["mesh"]
    return best_mesh


def insert_markers(tokens: List[str], e1: Tuple[int,int], e2: Tuple[int,int]) -> str:
    s1,e1_ = e1
    s2,e2_ = e2
    # keep order stable
    if s1 > s2:
        (s1,e1_), (s2,e2_) = (s2,e2_), (s1,e1_)
        tag1_open, tag1_close, tag2_open, tag2_close = "<e2>", "</e2>", "<e1>", "</e1>"
    else:
        tag1_open, tag1_close, tag2_open, tag2_close = "<e1>", "</e1>", "<e2>", "</e2>"

    out = []
    for i,t in enumerate(tokens):
        if i == s1: out.append(tag1_open)
        if i == s2: out.append(tag2_open)
        out.append(t)
        if i+1 == e1_: out.append(tag1_close)
        if i+1 == e2_: out.append(tag2_close)
    return " ".join(out)


def main():
    # 1) load index_test
    index_rows = []
    with open(INDEX_TEST, "r", encoding="utf-8") as f:
        for ln in f:
            index_rows.append(json.loads(ln))
    print(f"[OK] Loaded index rows: {len(index_rows)}")

    # 2) load gold PubTator test (mesh + CID)
    gold_docs = parse_pubtator_full(PUB_TEST)
    print(f"[OK] Loaded gold docs: {len(gold_docs)}")

    # 3) build dicts/embeddings like training
    word2idx, case2idx, char2idx, label2idx, word_embeddings, case_embeddings = \
        preprocessing.get_dicts_and_embeddings(TRAIN_CONLL, DEV_CONLL, EMB_PATH)

    # 4) build dataset for inference (use TRAIN as train, TEST as 'val')
    _, test_data_list, _, labels_test = preprocessing.get_dataset(TRAIN_CONLL, TEST_CONLL, EMB_PATH)

    # 5) load model
    custom_objects = {
        "WordEmbedding": architecture.WordEmbedding,
        "CasingEmbedding": architecture.CasingEmbedding,
        "CharacterEmbedding": architecture.CharacterEmbedding,
    }
    model = keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)
    print("[OK] Loaded NER model:", MODEL_PATH)

    # 6) predict BIO ids
    pred_probs = model.predict(test_data_list, verbose=1)
    pred_ids = np.argmax(pred_probs, axis=-1)

    pad_id = label2idx.get("PADDING", 0)
    idx2label = {v:k for k,v in label2idx.items()}

    # 7) write predicted entities + RE test_pred TSV
    n_sent = pred_ids.shape[0]
    if n_sent != len(index_rows):
        raise RuntimeError(
            f"[ALIGN ERROR] NER test sentences ({n_sent}) != index rows ({len(index_rows)}). "
            f"Run build_sentence_index.py first and ensure alignment."
        )

    with open(OUT_TEST_PRED_ENT, "w", encoding="utf-8") as ent_out, \
         open(OUT_RE_TEST_PRED, "w", encoding="utf-8") as re_out:

        total_pairs = 0
        pos_pairs = 0

        for i in range(n_sent):
            row = index_rows[i]
            pmid = row["pmid"]
            sent_id = row["sent_id"]
            tokens = row["tokens"]
            token_offsets = row["token_offsets"]

            # strip padding by gold labels length mask (or by pad_id in labels_test)
            t_seq = labels_test[i]
            keep = t_seq != pad_id
            # Some pipelines may not have pad in gold; fallback to len(tokens)
            if keep.sum() == 0:
                L = len(tokens)
                pred_labs = [idx2label[int(x)] for x in pred_ids[i][:L]]
            else:
                pred_labs = [idx2label[int(x)] for x in pred_ids[i][keep]]

            # recover spans
            spans = bio_to_spans(pred_labs)
            chems = [(t,s,e) for (t,s,e) in spans if t == "Chemical"]
            dises = [(t,s,e) for (t,s,e) in spans if t == "Disease"]

            # Build entity records and match mesh (by overlap with gold entities)
            gold = gold_docs.get(pmid, None)
            gold_ents = gold["entities"] if gold else []
            gold_cids = gold["cid_pairs"] if gold else set()

            ent_list = []
            for (t,s,e) in spans:
                abs_s, abs_e = tokens_span_to_char_span(token_offsets, s, e)
                mesh = match_to_gold_mesh((abs_s, abs_e), t, gold_ents)
                ent_list.append({
                    "type": t,
                    "start_tok": s,
                    "end_tok": e,
                    "abs_start": abs_s,
                    "abs_end": abs_e,
                    "text": " ".join(tokens[s:e]),
                    "mesh": mesh
                })

            ent_out.write(json.dumps({
                "pmid": pmid,
                "sent_id": sent_id,
                "entities": ent_list
            }, ensure_ascii=False) + "\n")

            # construct chem x dis pairs for RE test_pred
            for (_, cs, ce) in chems:
                c_abs = tokens_span_to_char_span(token_offsets, cs, ce)
                c_mesh = match_to_gold_mesh(c_abs, "Chemical", gold_ents)
                for (_, ds, de) in dises:
                    d_abs = tokens_span_to_char_span(token_offsets, ds, de)
                    d_mesh = match_to_gold_mesh(d_abs, "Disease", gold_ents)

                    # label by doc-level CID if both mesh available and pair is in gold CID set
                    label = "NO_RELATION"
                    if c_mesh is not None and d_mesh is not None and (c_mesh, d_mesh) in gold_cids:
                        label = "CID"
                        pos_pairs += 1

                    marked = insert_markers(tokens, (cs,ce), (ds,de))
                    re_out.write(label + "\t" + marked + "\n")
                    total_pairs += 1

        print(f"[OK] Wrote predicted entities: {OUT_TEST_PRED_ENT}")
        print(f"[OK] Wrote RE test_pred TSV: {OUT_RE_TEST_PRED}")
        print(f"     pairs total={total_pairs}, CID positives={pos_pairs}")


if __name__ == "__main__":
    main()