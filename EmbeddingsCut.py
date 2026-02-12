import os
import sys

TRAIN = "data/processed/bc5cdr_ner_sentence_train.txt"
DEV   = "data/processed/bc5cdr_ner_sentence_dev.txt"

# 你下载的 BioWordVec 大文件路径（按你实际改）
BIOWORDVEC = "embeddings/bio_embedding_extrinsic"

OUT_TXT = "embeddings/biowordvec_bc5_subset.txt"

def load_vocab_from_conll(paths):
    vocab = set()
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # token X X label
                tok = line.split()[0]
                vocab.add(tok)
                vocab.add(tok.lower())
    return vocab

def try_load_keyedvectors(path):
    # Try word2vec binary format first
    from gensim.models import KeyedVectors

    try:
        print(f"[INFO] Trying word2vec binary load: {path}")
        kv = KeyedVectors.load_word2vec_format(path, binary=True)
        print("[INFO] Loaded as word2vec binary.")
        return kv
    except Exception as e:
        print(f"[WARN] word2vec binary load failed: {e}")

    # Try fastText Facebook model (.bin) next
    try:
        print(f"[INFO] Trying fastText facebook model load: {path}")
        from gensim.models.fasttext import load_facebook_model
        ft = load_facebook_model(path)
        print("[INFO] Loaded as fastText facebook model.")
        return ft.wv
    except Exception as e:
        print(f"[WARN] fastText facebook model load failed: {e}")

    raise RuntimeError(
        "Could not load BioWordVec file. "
        "It may not be in word2vec-binary or fastText .bin format. "
        "Tell me the exact filename/extension you downloaded from Figshare."
    )

def write_subset(kv, vocab, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    dim = kv.vector_size
    kept = 0

    with open(out_path, "w", encoding="utf-8") as w:
        # IMPORTANT: many loaders (incl. your NER repo) prefer NO header line, like GloVe.
        # So we write "word v1 v2 ... vD" per line.
        for word in vocab:
            if word in kv:
                vec = kv[word]
                w.write(word + " " + " ".join(f"{x:.6f}" for x in vec) + "\n")
                kept += 1

    print(f"[DONE] Wrote subset embeddings: {out_path}")
    print(f"       dim={dim}, kept={kept} words (from vocab size {len(vocab)})")

def main():
    vocab = load_vocab_from_conll([TRAIN, DEV])
    print(f"[INFO] BC5 vocab size (incl. lowercase variants): {len(vocab)}")

    kv = try_load_keyedvectors(BIOWORDVEC)
    write_subset(kv, vocab, OUT_TXT)

if __name__ == "__main__":
    main()