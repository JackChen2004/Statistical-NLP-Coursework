import os, json
from typing import List, Tuple
import DataProcessing as DP  # 直接复用你现有的分句/分词/parse逻辑


# ---- inputs (your existing processed NER CoNLL) ----
CONLL_TRAIN = "data/processed/bc5cdr_ner_sentence_train.txt"
CONLL_DEV   = "data/processed/bc5cdr_ner_sentence_dev.txt"
CONLL_TEST  = "data/processed/bc5cdr_ner_sentence_test.txt"

# ---- outputs ----
OUT_TRAIN_INDEX = "data/processed/bc5cdr_sentence_index_train.jsonl"
OUT_DEV_INDEX   = "data/processed/bc5cdr_sentence_index_dev.jsonl"
OUT_TEST_INDEX  = "data/processed/bc5cdr_sentence_index_test.jsonl"


def read_conll_sentences(path: str) -> List[List[str]]:
    """Read sentence-level CoNLL: blank line separates sentences. Return list of token lists."""
    sents: List[List[str]] = []
    cur: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if cur:
                    sents.append(cur)
                    cur = []
                continue
            parts = line.split()
            # token X X label
            tok = parts[0]
            cur.append(tok)
        if cur:
            sents.append(cur)
    return sents


def write_index(pub_path: str, conll_path: str, out_index: str):
    conll_sents = read_conll_sentences(conll_path)
    os.makedirs(os.path.dirname(out_index), exist_ok=True)

    idx = 0  # global sentence index following CoNLL order

    with open(out_index, "w", encoding="utf-8") as w:
        for doc in DP.parse_pubtator(pub_path):
            sents = DP.split_sentences_with_offsets(doc.text)
            sent_id = 0
            for sent_text, sent_start in sents:
                toks = DP.tokenize_with_offsets(sent_text, sent_start, doc.text)
                if not toks:
                    continue
                tokens = [t for (t, _, _) in toks]

                # ---- strict alignment check with your existing CoNLL ----
                if idx >= len(conll_sents):
                    raise RuntimeError(
                        f"[ALIGN ERROR] More PubTator sentences than CoNLL in {conll_path}. "
                        f"At pmid={doc.pmid} sent_id={sent_id} idx={idx}"
                    )
                if tokens != conll_sents[idx]:
                    # show a small diff
                    a = tokens
                    b = conll_sents[idx]
                    show_a = " ".join(a[:30])
                    show_b = " ".join(b[:30])
                    raise RuntimeError(
                        f"[ALIGN ERROR] Token mismatch at global_sent_idx={idx}\n"
                        f"  pmid={doc.pmid} sent_id={sent_id}\n"
                        f"  PubTator tokens(<=30): {show_a}\n"
                        f"  CoNLL   tokens(<=30): {show_b}\n"
                        f"Tip: this means your processed CoNLL was created with different "
                        f"sentence splitting/tokenization than current DataProcessing.py."
                    )

                rec = {
                    "pmid": doc.pmid,
                    "sent_id": sent_id,
                    "sent_start": sent_start,
                    "sent_end": sent_start + len(sent_text),
                    "sent_text": sent_text,
                    "tokens": tokens,
                    "token_offsets": [{"start": s, "end": e} for (_, s, e) in toks],
                }
                w.write(json.dumps(rec, ensure_ascii=False) + "\n")

                idx += 1
                sent_id += 1

    if idx != len(conll_sents):
        raise RuntimeError(
            f"[ALIGN ERROR] CoNLL has {len(conll_sents)} sentences but index wrote {idx}.\n"
            f"This means PubTator->sentence generation count differs from your CoNLL."
        )

    print(f"[OK] Wrote index: {out_index}")
    print(f"     aligned sentences: {idx}")


def main():
    print("Building index for TRAIN ...")
    write_index(DP.TRAIN_PUBTATOR, CONLL_TRAIN, OUT_TRAIN_INDEX)

    print("Building index for DEV ...")
    write_index(DP.DEV_PUBTATOR, CONLL_DEV, OUT_DEV_INDEX)

    print("Building index for TEST ...")
    write_index(DP.TEST_PUBTATOR, CONLL_TEST, OUT_TEST_INDEX)


if __name__ == "__main__":
    main()