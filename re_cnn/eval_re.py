# re_cnn/eval_re.py
import argparse
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, classification_report
from tensorflow import keras

from re_cnn.data import read_re_jsonl, load_embeddings_vocab, vectorize
from re_cnn.model import build_re_cnn

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="re_models/re_cnn_best.keras")
    ap.add_argument("--test", default="re_data_gold/re_test_gold.jsonl")
    ap.add_argument("--emb", default="embeddings/biowordvec_bc5_subset.txt")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--max_dist", type=int, default=64)
    args = ap.parse_args()

    exs = read_re_jsonl(args.test)
    word2id, _ = load_embeddings_vocab(args.emb)
    (Xw, Xp1, Xp2), y, label2id = vectorize(exs, word2id, max_len=args.max_len, max_dist=args.max_dist)
    id2label = {v: k for k, v in label2id.items()}

    model = keras.models.load_model(args.model)
    probs = model.predict([Xw, Xp1, Xp2], batch_size=128, verbose=1)
    yhat = np.argmax(probs, axis=1)

    p, r, f1, _ = precision_recall_fscore_support(y, yhat, average="binary", pos_label=1, zero_division=0)
    print(f"[TEST] CID_P={p:.4f} CID_R={r:.4f} CID_F1={f1:.4f}")
    print(classification_report(y, yhat, target_names=[id2label[0], id2label[1]], digits=4))

if __name__ == "__main__":
    main()
