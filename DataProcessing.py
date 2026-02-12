import re
from dataclasses import dataclass
from typing import List, Dict, Tuple, Iterator, Optional

TRAIN_PUBTATOR = "data/origin/CDR_TrainingSet.PubTator.txt"
DEV_PUBTATOR   = "data/origin/CDR_DevelopmentSet.PubTator.txt"
TEST_PUBTATOR  = "data/origin/CDR_TestSet.PubTator.txt"

OUT_TRAIN = "data/processed/bc5cdr_ner_sentence_train.txt"
OUT_DEV   = "data/processed/bc5cdr_ner_sentence_dev.txt"
OUT_TEST  = "data/processed/bc5cdr_ner_sentence_test.txt"

# --------- Data structures ---------
@dataclass
class Entity:
    start: int
    end: int
    etype: str  # "Chemical" or "Disease"

@dataclass
class Doc:
    pmid: str
    title: str
    abstract: str
    text: str
    entities: List[Entity]

# --------- PubTator parsing ---------
def parse_pubtator(path: str) -> Iterator[Doc]:
    """
    Parses BC5 CDR PubTator format:
      PMID|t|title
      PMID|a|abstract
      PMID  start  end  mention  type  (optional id...)
      PMID  CID  chem_id  dis_id  (relations; ignored for NER)
    Blank line separates documents.
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f]

    i = 0
    while i < len(lines):
        # skip extra blank lines
        while i < len(lines) and lines[i].strip() == "":
            i += 1
        if i >= len(lines):
            break

        # title line
        m = re.match(r"^(\d+)\|t\|(.*)$", lines[i])
        if not m:
            raise ValueError(f"Unexpected line (expected title): {lines[i][:120]}")
        pmid = m.group(1)
        title = m.group(2)
        i += 1

        # abstract line
        m = re.match(rf"^{pmid}\|a\|(.*)$", lines[i])
        if not m:
            raise ValueError(f"Unexpected line (expected abstract): {lines[i][:120]}")
        abstract = m.group(1)
        i += 1

        entities: List[Entity] = []
        # entity/relation lines until blank
        while i < len(lines) and lines[i].strip() != "":
            parts = lines[i].split("\t")
            # entity line typically: pmid, start, end, mention, type, (id...)
            # relation line starts with: pmid, CID, ...
            if len(parts) >= 5 and parts[1].isdigit() and parts[2].isdigit():
                etype = parts[4]
                if etype in ("Chemical", "Disease"):
                    entities.append(Entity(start=int(parts[1]), end=int(parts[2]), etype=etype))
            # else ignore (CID lines etc.)
            i += 1

        text = title + " " + abstract
        yield Doc(pmid=pmid, title=title, abstract=abstract, text=text, entities=entities)

# --------- Sentence splitting with offsets ---------
_sentence_end_re = re.compile(r"([.!?])\s+")  # simple baseline splitter

def split_sentences_with_offsets(text: str) -> List[Tuple[str, int]]:
    """
    Rule-based sentence splitter returning list of (sentence_text, sentence_start_char_in_doc).
    Keeps punctuation in sentence.
    """
    sents: List[Tuple[str, int]] = []
    start = 0
    for m in _sentence_end_re.finditer(text):
        end = m.end(1)  # include the punctuation
        sent = text[start:end].strip()
        if sent:
            # sentence start offset in original doc:
            # find first non-space from start
            real_start = start + (len(text[start:]) - len(text[start:].lstrip()))
            # But we trimmed; easiest: locate sent in slice
            # We'll compute accurate real_start by searching within [start, end)
            slice_ = text[start:end]
            lstrip_len = len(slice_) - len(slice_.lstrip())
            real_start = start + lstrip_len
            sents.append((sent, real_start))
        start = m.end()  # position after whitespace
    # tail
    tail = text[start:].strip()
    if tail:
        slice_ = text[start:]
        lstrip_len = len(slice_) - len(slice_.lstrip())
        real_start = start + lstrip_len
        sents.append((tail, real_start))
    return sents

# --------- Tokenization with offsets ---------
_token_re = re.compile(r"\w+|[^\w\s]", re.UNICODE)

def tokenize_with_offsets(sent: str, sent_start: int, doc_text: str) -> List[Tuple[str, int, int]]:
    """
    Tokenize a sentence into tokens and compute token offsets in DOC coordinates.
    Uses a regex that splits words/numbers and punctuation.
    """
    out = []
    for m in _token_re.finditer(sent):
        tok = m.group(0)
        tok_s = sent_start + m.start()
        tok_e = sent_start + m.end()
        out.append((tok, tok_s, tok_e))
    return out

# --------- Span-to-BIO alignment ---------
def overlaps(a_s: int, a_e: int, b_s: int, b_e: int) -> bool:
    return not (a_e <= b_s or b_e <= a_s)

def bio_labels_for_tokens(tokens: List[Tuple[str, int, int]], entities: List[Entity]) -> List[str]:
    """
    Assign BIO labels to tokens based on entity spans (doc-level char offsets).
    Assumes minimal overlap and prefers the entity with larger overlap if needed.
    """
    labels = ["O"] * len(tokens)

    # Sort entities by start then longer first (helps in rare overlap cases)
    ents = sorted(entities, key=lambda e: (e.start, -(e.end - e.start)))

    for idx, (_, ts, te) in enumerate(tokens):
        best_ent: Optional[Entity] = None
        best_ov = 0
        for ent in ents:
            if overlaps(ts, te, ent.start, ent.end):
                ov = min(te, ent.end) - max(ts, ent.start)
                if ov > best_ov:
                    best_ov = ov
                    best_ent = ent
        if best_ent and best_ov > 0:
            prefix = "CHEM" if best_ent.etype == "Chemical" else "DIS"
            # Determine B vs I: check if previous token is in same entity
            if idx == 0:
                labels[idx] = f"B-{prefix}"
            else:
                _, pts, pte = tokens[idx - 1]
                prev_in_same = overlaps(pts, pte, best_ent.start, best_ent.end)
                labels[idx] = f"I-{prefix}" if prev_in_same else f"B-{prefix}"

    # Fix illegal I- starts (optional safety): convert I- to B- if previous not same type
    for i in range(len(labels)):
        if labels[i].startswith("I-"):
            if i == 0 or labels[i-1] == "O" or labels[i-1][2:] != labels[i][2:]:
                labels[i] = "B-" + labels[i][2:]
    return labels

# --------- Deterministic train/dev split by PMID ---------
def is_dev(pmid: str, mod: int = 10, bucket: int = 0) -> bool:
    # stable split: last digit heuristic (simple & deterministic)
    # You can replace with hash for better distribution if you want.
    return int(pmid) % mod == bucket

# --------- Write CoNLL ---------
def write_conll(docs: Iterator[Doc], out_path: str, sentence_level: bool = True,
                only_dev: Optional[bool] = None) -> None:
    """
    Writes sentence-level CoNLL. If only_dev is True/False, filters docs by split.
    """
    with open(out_path, "w", encoding="utf-8") as w:
        for doc in docs:
            if only_dev is not None:
                if is_dev(doc.pmid) != only_dev:
                    continue

            sents = split_sentences_with_offsets(doc.text) if sentence_level else [(doc.text, 0)]
            for sent_text, sent_start in sents:
                tokens = tokenize_with_offsets(sent_text, sent_start, doc.text)
                if not tokens:
                    continue
                labels = bio_labels_for_tokens(tokens, doc.entities)

                # write one sentence
                for (tok, _, _), lab in zip(tokens, labels):
                    w.write(f"{tok} X X {lab}\n")
                w.write("\n")

def main():
    # Train
    write_conll(parse_pubtator(TRAIN_PUBTATOR), OUT_TRAIN, sentence_level=True)

    # Dev 
    write_conll(parse_pubtator(DEV_PUBTATOR), OUT_DEV, sentence_level=True)

    # Test
    write_conll(parse_pubtator(TEST_PUBTATOR), OUT_TEST, sentence_level=True)

    print("Wrote:")
    print(" ", OUT_TRAIN)
    print(" ", OUT_DEV)
    print(" ", OUT_TEST)

if __name__ == "__main__":
    main()