import argparse
import json
import os
from typing import Dict, List
from collections import defaultdict

def load_json_shard(path: str) -> Dict[str, Dict[str, List[str]]]:
    with open(path) as f:
        res = json.load(f)

    data = {}
    for tid, gt in res.get("gt", {}).items():
        pred = res.get("pred", {}).get(tid, [])
        if len(pred) != len(gt):
            raise ValueError(f"Mismatch in GT / pred lengths for TID {tid}")
        data[tid] = {"gt": gt, "pred": pred}
    return data


def cal_f1(instances: List[Dict[str, str]]) -> None:
    vocab = {item["gt"].lower() for item in instances}

    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for inst in instances:
        gt = inst["gt"].lower()
        pred = inst["pred"].lower()

        if pred == gt:
            tp[gt] += 1
        elif pred in vocab and pred != "":
            fp[pred] += 1
            fn[gt]  += 1
        else:
            fn[gt] += 1

    total_tp = sum(tp.values())
    total_fp = sum(fp.values())
    total_fn = sum(fn.values())

    micro_prec = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0
    micro_rec  = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0
    micro_f1   = (2 * micro_prec * micro_rec /
                  (micro_prec + micro_rec) if micro_prec + micro_rec else 0)

    print(f"Micro F1: {micro_f1:.4f}")


def main():
    ap = argparse.ArgumentParser(description="Evaluate CTA SOTAB.")
    ap.add_argument("--dir", required=True, help="outdir")
    args = ap.parse_args()

    if not os.path.isdir(args.dir):
        raise FileNotFoundError(args.dir)

    instances = []
    for fname in os.listdir(args.dir):
        if fname.endswith(".json"):
            shard = load_json_shard(os.path.join(args.dir, fname))
            for tid, pair in shard.items():
                for g, p in zip(pair["gt"], pair["pred"]):
                    instances.append({"tid": tid, "gt": g, "pred": p})

    cal_f1(instances)


if __name__ == "__main__":
    main()

