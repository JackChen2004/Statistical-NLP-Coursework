# re_cnn/predict_re.py
import os
import json
import argparse
import numpy as np
from tensorflow import keras

from re_cnn.data import read_re_jsonl, vectorize

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="re_models/re_cnn_best.keras")
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_jsonl", default="re_pred.jsonl")
    ap.add_argument("--out_tsv", default="re_pred.tsv")
    ap.add_argument("--assets_dir", default="re_models")  # word2id/label2id/cfg
    args = ap.parse_args()

    word2id = np.load(os.path.join(args.assets_dir, "word2id.npy"), allow_pickle=True).item()
    label2id = np.load(os.path.join(args.assets_dir, "label2id.npy"), allow_pickle=True).item()
    cfg = np.load(os.path.join(args.assets_dir, "cfg.npy"), allow_pickle=True).item()
    id2label = {v: k for k, v in label2id.items()}

    exs = read_re_jsonl(args.in_jsonl)
    (Xw, Xp1, Xp2), _, _ = vectorize(exs, word2id, max_len=cfg["max_len"], max_dist=cfg["max_dist"], label2id=label2id)

    model = keras.models.load_model(args.model)
    probs = model.predict([Xw, Xp1, Xp2], batch_size=128, verbose=1)
    yhat = np.argmax(probs, axis=1)

    with open(args.out_jsonl, "w", encoding="utf-8") as fj, open(args.out_tsv, "w", encoding="utf-8") as ft:
        for ex, yy, pp in zip(exs, yhat, probs):
            pred = id2label[int(yy)]
            score = float(np.max(pp))

            out = {
                "pred": pred,
                "score": score,
                "tokens": ex.tokens,
                "chem_span": list(ex.chem_span),
                "dis_span": list(ex.dis_span),
                **ex.meta
            }
            fj.write(json.dumps(out, ensure_ascii=False) + "\n")

            sent = " ".join(ex.tokens)
            ft.write(f"{pred}\t{score:.6f}\t{sent}\n")

    print("[OK] wrote:", args.out_jsonl, args.out_tsv)

if __name__ == "__main__":
    main()
