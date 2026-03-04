# re_cnn/train_re.py
import os
import argparse
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from re_cnn.data import read_re_jsonl, load_embeddings_vocab, vectorize
from re_cnn.model import build_re_cnn

def cid_prf(y_true, y_pred):
    # y_pred: prob -> label
    y_hat = np.argmax(y_pred, axis=1)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_hat, average="binary", pos_label=1, zero_division=0)
    return p, r, f1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="re_data_gold/re_train_gold.jsonl")
    ap.add_argument("--dev", default="re_data_gold/re_dev_gold.jsonl")
    ap.add_argument("--emb", default="embeddings/biowordvec_bc5_subset.txt")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--max_dist", type=int, default=64)
    ap.add_argument("--out_dir", default="re_models")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    train_ex = read_re_jsonl(args.train)
    dev_ex = read_re_jsonl(args.dev)

    word2id, emb = load_embeddings_vocab(args.emb)

    (Xw_tr, Xp1_tr, Xp2_tr), y_tr, label2id = vectorize(train_ex, word2id, max_len=args.max_len, max_dist=args.max_dist)
    (Xw_de, Xp1_de, Xp2_de), y_de, _ = vectorize(dev_ex, word2id, max_len=args.max_len, max_dist=args.max_dist, label2id=label2id)

    model = build_re_cnn(
        emb_matrix=emb,
        max_len=args.max_len,
        max_dist=args.max_dist,
        trainable_word_emb=False,
    )

    best_f1 = -1.0
    best_path = os.path.join(args.out_dir, "re_cnn_best.keras")

    for epoch in range(1, args.epochs + 1):
        model.fit([Xw_tr, Xp1_tr, Xp2_tr], y_tr, batch_size=args.batch, epochs=1, verbose=1)

        dev_pred = model.predict([Xw_de, Xp1_de, Xp2_de], batch_size=args.batch, verbose=0)
        p, r, f1 = cid_prf(y_de, dev_pred)

        print(f"[DEV] epoch={epoch} CID_P={p:.4f} CID_R={r:.4f} CID_F1={f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            model.save(best_path)
            print(f"[SAVE] best model saved to {best_path}")

    # 保存配置（便于 predict）
    np.save(os.path.join(args.out_dir, "word2id.npy"), word2id, allow_pickle=True)
    np.save(os.path.join(args.out_dir, "label2id.npy"), label2id, allow_pickle=True)
    np.save(os.path.join(args.out_dir, "cfg.npy"), {"max_len": args.max_len, "max_dist": args.max_dist}, allow_pickle=True)

    print("[DONE] best_f1 =", best_f1)

if __name__ == "__main__":
    main()
