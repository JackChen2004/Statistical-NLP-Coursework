# re_cnn/build_pred_jsonl.py
import os
import json
import argparse
from typing import List, Dict, Any, Tuple

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_entities", default="pipeline_outputs/test_pred_entities.jsonl")
    ap.add_argument("--out_jsonl", default="pipeline_outputs/re_test_pred.jsonl")
    args = ap.parse_args()

    pairs = 0
    with open(args.pred_entities, "r", encoding="utf-8") as f, open(args.out_jsonl, "w", encoding="utf-8") as out:
        for ln in f:
            if not ln.strip():
                continue
            obj = json.loads(ln)
            tokens = obj["tokens"]
            ents = obj["entities"]  # list: {type,start_tok,end_tok,mesh,...}
            pmid = obj.get("pmid")
            sent_id = obj.get("sent_id")

            chems = [e for e in ents if e["type"] == "Chemical"]
            dises = [e for e in ents if e["type"] == "Disease"]

            for c in chems:
                for d in dises:
                    rec = {
                        "split": "pred",
                        "pmid": pmid,
                        "sent_id": sent_id,
                        "label": "NO_RELATION",   # unknown; placeholder
                        "tokens": tokens,
                        "chem_span": [c["start_tok"], c["end_tok"]],
                        "dis_span": [d["start_tok"], d["end_tok"]],
                        "chem_mesh": c.get("mesh", "-1"),
                        "dis_mesh": d.get("mesh", "-1"),
                    }
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    pairs += 1

    print(f"[OK] wrote {pairs} predicted pairs -> {args.out_jsonl}")

if __name__ == "__main__":
    main()
