"""CLI: embed pairs, train a predictor, and evaluate frozen scores."""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from jeval_v2.data import read_pairs
from jeval_v2.model import Predictor, scores


def embed(args):
    from sentence_transformers import SentenceTransformer

    rows = read_pairs(args.pairs)
    model = SentenceTransformer(args.encoder, device=args.device)
    originals = model.encode([r["original"] for r in rows], batch_size=args.batch_size,
                             convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
    compressed = model.encode([r["compressed"] for r in rows], batch_size=args.batch_size,
                              convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, original=originals, compressed=compressed,
                        ids=np.array([r["id"] for r in rows]),
                        groups=np.array([r["group_id"] for r in rows]),
                        splits=np.array([r["split"] for r in rows]),
                        harm=np.array([r.get("harm", -1) for r in rows], dtype=np.int8))
    dest.with_suffix(".metadata.json").write_text(json.dumps({"encoder": args.encoder,
        "pair_file": str(Path(args.pairs).resolve()), "count": len(rows)}, indent=2))


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    data = np.load(args.embeddings, allow_pickle=False)
    x = torch.from_numpy(data["compressed"]).float()
    y = torch.from_numpy(data["original"]).float()
    splits = data["splits"]
    train_ids = np.where(splits == "train")[0]
    valid_ids = np.where(splits == "validation")[0]
    if not len(train_ids) or not len(valid_ids):
        raise ValueError("both train and validation pairs are required")
    device = torch.device(args.device)
    model = Predictor(x.shape[1], args.hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    best = float("inf")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        order = np.random.permutation(train_ids)
        for start in range(0, len(order), args.batch_size):
            indices = order[start:start + args.batch_size]
            prediction = torch.nn.functional.normalize(model(x[indices].to(device)), dim=-1)
            loss = torch.nn.functional.mse_loss(prediction, y[indices].to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_prediction = torch.nn.functional.normalize(model(x[valid_ids].to(device)), dim=-1)
            val_loss = torch.nn.functional.mse_loss(val_prediction, y[valid_ids].to(device)).item()
        print(json.dumps({"epoch": epoch + 1, "validation_mse": val_loss}))
        if val_loss < best:
            best = val_loss
            torch.save({"state_dict": model.state_dict(), "dim": x.shape[1],
                        "hidden": args.hidden, "seed": args.seed}, output)


def evaluate(args):
    data = np.load(args.embeddings, allow_pickle=False)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = Predictor(checkpoint["dim"], checkpoint["hidden"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    selected = np.where(data["splits"] == args.split)[0]
    if not len(selected):
        raise ValueError(f"no rows for split {args.split}")
    with torch.no_grad():
        cosine, epe = scores(torch.from_numpy(data["original"][selected]).float(),
                             torch.from_numpy(data["compressed"][selected]).float(), model)
    labels = data["harm"][selected]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for idx, cos, residual, label in zip(selected, cosine.tolist(), epe.tolist(), labels.tolist()):
            handle.write(json.dumps({"id": str(data["ids"][idx]), "group_id": str(data["groups"][idx]),
                "split": args.split, "harm": label, "cosine": cos, "epe": residual}) + "\n")
    known = labels >= 0
    if known.sum() and len(set(labels[known])) == 2:
        metrics = {}
        for name, values in (("cosine", cosine.numpy()), ("epe", epe.numpy())):
            metrics[name] = {"auprc": average_precision_score(labels[known], values[known]),
                             "auroc": roc_auc_score(labels[known], values[known])}
            for fraction in (0.05, 0.10, 0.20):
                count = max(1, int(np.ceil(known.sum() * fraction)))
                rank = np.argsort(-values[known])[:count]
                metrics[name][f"harm_recall_at_{int(fraction*100)}pct_rejected"] = float(
                    labels[known][rank].sum() / labels[known].sum())
        print(json.dumps(metrics, indent=2))
    else:
        print(f"wrote {len(selected)} scores; metrics need labeled examples from both classes")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("embed")
    p.add_argument("--pairs", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--encoder", default="sentence-transformers/all-mpnet-base-v2")
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=64)
    p.set_defaults(run=embed)
    p = sub.add_parser("train")
    p.add_argument("--embeddings", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(run=train)
    p = sub.add_parser("evaluate")
    p.add_argument("--embeddings", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", choices=["validation", "test"], default="test")
    p.add_argument("--output", required=True)
    p.set_defaults(run=evaluate)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
