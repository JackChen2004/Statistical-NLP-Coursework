import os, re, json
from typing import Dict, List, Tuple, Any, Optional

# ========= Paths (edit if needed) =========
TRAIN_PUB = "data/origin/CDR_TrainingSet.PubTator.txt"
DEV_PUB   = "data/origin/CDR_DevelopmentSet.PubTator.txt"
TEST_PUB  = "data/origin/CDR_TestSet.PubTator.txt"

OUT_DIR = "re_data_gold"
os.makedirs(OUT_DIR, exist_ok=True)

INDEX_DIR = os.path.join(OUT_DIR, "sentence_index")
NER_INFER_DIR = os.path.join(OUT_DIR, "ner_infer_conll")
os.makedirs(INDEX_DIR, exist_ok=True)
os.makedirs(NER_INFER_DIR, exist_ok=True)

OUT_TRAIN_INDEX = os.path.join(INDEX_DIR, "bc5cdr_sentence_index_train.jsonl")
OUT_DEV_INDEX   = os.path.join(INDEX_DIR, "bc5cdr_sentence_index_dev.jsonl")
OUT_TEST_INDEX  = os.path.join(INDEX_DIR, "bc5cdr_sentence_index_test.jsonl")

OUT_TRAIN_NER_CONLL = os.path.join(NER_INFER_DIR, "bc5cdr_ner_infer_train.conll")
OUT_DEV_NER_CONLL   = os.path.join(NER_INFER_DIR, "bc5cdr_ner_infer_dev.conll")
OUT_TEST_NER_CONLL  = os.path.join(NER_INFER_DIR, "bc5cdr_ner_infer_test.conll")

# ========= Optional outputs =========
# If you already have an aligned index from your NER preprocessing (recommended), set these to False.
WRITE_SENTENCE_INDEX = True
WRITE_DUMMY_NER_CONLL = False
FORCE_OVERWRITE_INDEX = False

# ========= Helpers =========
_SENT_SPLIT_RE = re.compile(r"(?<=[\.\?\!])\s+")

def sentence_spans(text: str) -> List[Tuple[int,int,str]]:
    """
    Very simple sentence splitter that returns (start,end,sent_text) spans
    over the concatenated doc_text = title + ' ' + abstract.
    """
    spans = []
    start = 0
    # split keeping offsets by searching separators
    parts = _SENT_SPLIT_RE.split(text)
    cursor = 0
    for p in parts:
        p = p.strip()
        if not p:
            continue
        idx = text.find(p, cursor)
        if idx == -1:
            idx = cursor
        s = idx
        e = idx + len(p)
        spans.append((s, e, text[s:e]))
        cursor = e
    return spans

def tokenize_with_offsets(sent: str, sent_start: int) -> List[Tuple[str,int,int]]:
    """Return list of (token, abs_start, abs_end) where offsets are in the doc_text coordinate system."""
    out = []
    for m in re.finditer(r"\w+|[^\w\s]", sent):
        tok = m.group(0)
        s = sent_start + m.start()
        e = sent_start + m.end()
        out.append((tok, s, e))
    return out

def insert_markers(tokens: List[str], e1: Tuple[int,int], e2: Tuple[int,int]) -> str:
    """
    e1,e2 are (start_tok, end_tok) where end_tok is exclusive.
    Inserts <e1> </e1> and <e2> </e2> around token spans.
    """
    s1,e1_ = e1
    s2,e2_ = e2
    if s1 > s2:
        # keep e1 before e2 by swapping
        (s1,e1_),(s2,e2_) = (s2,e2_),(s1,e1_)
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

def parse_pubtator(path: str) -> Dict[str, Any]:
    """
    Returns dict pmid -> {title, abstract, entities, cid_pairs}
    entities: list of {start,end,text,type,mesh}
    cid_pairs: set of (chem_mesh, dis_mesh)
    """
    docs = {}
    with open(path, "r", encoding="utf-8") as f:
        block = []
        for line in f:
            line = line.rstrip("\n")
            if line.strip() == "":
                if block:
                    _parse_block(block, docs)
                    block = []
            else:
                block.append(line)
        if block:
            _parse_block(block, docs)
    return docs

def _parse_block(lines: List[str], docs: Dict[str, Any]):
    title = abstract = None
    pmid = None
    ents = []
    cid = set()

    for ln in lines:
        if "|t|" in ln:
            pmid, rest = ln.split("|t|", 1)
            title = rest
        elif "|a|" in ln:
            pmid2, rest = ln.split("|a|", 1)
            pmid = pmid or pmid2
            abstract = rest
        else:
            parts = ln.split("\t")
            if len(parts) >= 6 and parts[1].isdigit():
                # entity line: pmid start end mention type mesh
                pmid = pmid or parts[0]
                ents.append({
                    "start": int(parts[1]),
                    "end": int(parts[2]),
                    "text": parts[3],
                    "type": parts[4],   # Chemical / Disease
                    "mesh": parts[5],
                })
            elif len(parts) >= 4 and parts[1] == "CID":
                pmid = pmid or parts[0]
                cid.add((parts[2], parts[3]))

    if pmid is None:
        return
    docs[pmid] = {
        "title": title or "",
        "abstract": abstract or "",
        "entities": ents,
        "cid_pairs": cid
    }

def entities_in_sentence(entities, sent_start, sent_end):
    # Keep entities fully inside sentence span
    return [e for e in entities if e["start"] >= sent_start and e["end"] <= sent_end]

def char_to_tok_span(sent_start: int, ent_start: int, ent_end: int, tokens: List[Tuple[str,int,int]]) -> Optional[Tuple[int,int]]:
    # entity span in absolute doc offsets
    rel_tok_ids = []
    for i, (_, ts, te) in enumerate(tokens):
        # overlap check in absolute coords
        if not (te <= ent_start or ts >= ent_end):
            rel_tok_ids.append(i)
    if not rel_tok_ids:
        return None
    return (min(rel_tok_ids), max(rel_tok_ids) + 1)

def build_re_instances(docs: Dict[str,Any], split_name: str) -> List[Dict[str,Any]]:
    instances = []
    for pmid, d in docs.items():
        doc_text = (d["title"] + " " + d["abstract"]).strip()
        sents = sentence_spans(doc_text)

        for sid,(ss,se,sent_text) in enumerate(sents):
            sent_ents = entities_in_sentence(d["entities"], ss, se)
            chems = [e for e in sent_ents if e["type"].lower().startswith("chem")]
            dises = [e for e in sent_ents if e["type"].lower().startswith("dis")]

            if not chems or not dises:
                continue

            tokens_with_offs = tokenize_with_offsets(sent_text, ss)
            tokens = [t for (t, _, _) in tokens_with_offs]

            # build all chem x dis pairs within sentence
            for c in chems:
                for di in dises:
                    # label based on doc-level CID pairs (mesh ids)
                    label = "CID" if (c["mesh"], di["mesh"]) in d["cid_pairs"] else "NO_RELATION"

                    c_span = char_to_tok_span(ss, c["start"], c["end"], tokens_with_offs)
                    d_span = char_to_tok_span(ss, di["start"], di["end"], tokens_with_offs)
                    if c_span is None or d_span is None:
                        continue

                    marked = insert_markers(tokens, c_span, d_span)

                    instances.append({
                        "split": split_name,
                        "pmid": pmid,
                        "sent_id": sid,
                        "label": label,
                        "chem_mesh": c["mesh"],
                        "dis_mesh": di["mesh"],
                        "tokens": tokens,
                        "chem_span": c_span,
                        "dis_span": d_span,
                        "sentence_marked": marked
                    })
    return instances

def write_outputs(instances: List[Dict[str,Any]], out_prefix: str):
    # JSONL
    jsonl_path = os.path.join(OUT_DIR, f"{out_prefix}.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for ex in instances:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # TSV: label \t sentence_with_markers  (easy to adapt to CNN-RE repos)
    tsv_path = os.path.join(OUT_DIR, f"{out_prefix}.tsv")
    with open(tsv_path, "w", encoding="utf-8") as f:
        for ex in instances:
            f.write(ex["label"] + "\t" + ex["sentence_marked"] + "\n")

    print(f"[OK] Wrote {len(instances)} instances:")
    print(" ", jsonl_path)
    print(" ", tsv_path)

def write_sentence_index_and_dummy_conll(docs: Dict[str, Any], index_path: str, conll_path: str):
    # Optionally skip overwriting an existing index file
    if (not FORCE_OVERWRITE_INDEX) and os.path.exists(index_path):
        print(f"[SKIP] Index exists, not overwriting: {index_path}")
        # Still allow dummy generation if requested
        if not WRITE_DUMMY_NER_CONLL:
            return

    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    if WRITE_DUMMY_NER_CONLL:
        os.makedirs(os.path.dirname(conll_path), exist_ok=True)

    jf = open(index_path, "w", encoding="utf-8")
    cf = open(conll_path, "w", encoding="utf-8") if WRITE_DUMMY_NER_CONLL else None
    try:
        for pmid, d in docs.items():
            doc_text = (d["title"] + " " + d["abstract"]).strip()
            sents = sentence_spans(doc_text)

            for sent_id, (ss, se, sent_text) in enumerate(sents):
                toks = tokenize_with_offsets(sent_text, ss)
                if not toks:
                    continue

                # index record
                rec = {
                    "pmid": pmid,
                    "sent_id": sent_id,
                    "sent_start": ss,
                    "sent_end": se,
                    "sent_text": sent_text,
                    "tokens": [t for (t, _, _) in toks],
                    "token_offsets": [{"start": s, "end": e} for (_, s, e) in toks],
                }
                jf.write(json.dumps(rec, ensure_ascii=False) + "\n")

                if cf is not None:
                    # dummy CoNLL for NER inference (token + 2 placeholder cols + O)
                    for (tok, _, _) in toks:
                        cf.write(f"{tok}\tX\tX\tO\n")
                    cf.write("\n")
    finally:
        jf.close()
        if cf is not None:
            cf.close()

def main():
    train_docs = parse_pubtator(TRAIN_PUB)
    dev_docs   = parse_pubtator(DEV_PUB)
    test_docs  = parse_pubtator(TEST_PUB)

    # Extra outputs for pipeline alignment (optional)
    if WRITE_SENTENCE_INDEX:
        write_sentence_index_and_dummy_conll(train_docs, OUT_TRAIN_INDEX, OUT_TRAIN_NER_CONLL)
        write_sentence_index_and_dummy_conll(dev_docs,   OUT_DEV_INDEX,   OUT_DEV_NER_CONLL)
        write_sentence_index_and_dummy_conll(test_docs,  OUT_TEST_INDEX,  OUT_TEST_NER_CONLL)

        if WRITE_DUMMY_NER_CONLL:
            print("[OK] Wrote sentence index + dummy NER CoNLL to:")
            print(" ", OUT_TRAIN_INDEX)
            print(" ", OUT_DEV_INDEX)
            print(" ", OUT_TEST_INDEX)
            print(" ", OUT_TRAIN_NER_CONLL)
            print(" ", OUT_DEV_NER_CONLL)
            print(" ", OUT_TEST_NER_CONLL)
        else:
            print("[OK] Wrote sentence index (dummy NER CoNLL disabled):")
            print(" ", OUT_TRAIN_INDEX)
            print(" ", OUT_DEV_INDEX)
            print(" ", OUT_TEST_INDEX)

    train_inst = build_re_instances(train_docs, "train")
    dev_inst   = build_re_instances(dev_docs, "dev")
    test_inst  = build_re_instances(test_docs, "test_gold_entities")

    write_outputs(train_inst, "re_train_gold")
    write_outputs(dev_inst,   "re_dev_gold")
    write_outputs(test_inst,  "re_test_gold")

if __name__ == "__main__":
    main()