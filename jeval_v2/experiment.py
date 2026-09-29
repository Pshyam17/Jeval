"""CLI: embed pairs, train a predictor, and evaluate frozen scores."""
import argparse
import json
import random
import hashlib
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from jeval_v2.data import read_pairs
from jeval_v2.diagnose import diagnose_arrays
from jeval_v2.model import Predictor, scores


def encoder_and_lengths(args):
    from sentence_transformers import SentenceTransformer
    rows = read_pairs(args.pairs)
    model = SentenceTransformer(args.encoder, revision=args.encoder_revision, device=args.device)
    max_tokens = model.max_seq_length
    lengths = []
    for row in rows:
        lengths.append([len(model.tokenizer.encode(row[field], add_special_tokens=True))
                        for field in ("original", "compressed")])
    lengths = np.asarray(lengths)
    return rows, model, max_tokens, lengths


def inspect_lengths(args):
    rows, _, limit, lengths = encoder_and_lengths(args)
    print(json.dumps({"encoder": args.encoder, "encoder_revision": args.encoder_revision,
                      "limit": limit, "pairs": len(rows),
                      "original_exceeding": int((lengths[:, 0] > limit).sum()),
                      "compressed_exceeding": int((lengths[:, 1] > limit).sum()),
                      "original_p50_p95_max": np.percentile(lengths[:, 0], [50, 95, 100]).tolist(),
                      "compressed_p50_p95_max": np.percentile(lengths[:, 1], [50, 95, 100]).tolist()}, indent=2))


def embed(args):
    rows, model, max_tokens, lengths = encoder_and_lengths(args)
    if np.any(lengths > max_tokens):
        i, field = np.argwhere(lengths > max_tokens)[0]
        raise ValueError(f"{rows[i]['id']} {('original', 'compressed')[field]}: {lengths[i, field]} tokens exceeds encoder limit {max_tokens}")
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
                        harm=np.array([r.get("harm", -1) for r in rows], dtype=np.int8),
                        compression_ratio=np.array([len(r["compressed"].split()) / len(r["original"].split())
                                                    for r in rows], dtype=np.float32))
    dest.with_suffix(".metadata.json").write_text(json.dumps({"encoder": args.encoder,
        "encoder_revision": args.encoder_revision,
        "pair_file": str(Path(args.pairs).resolve()), "pair_sha256": hashlib.sha256(Path(args.pairs).read_bytes()).hexdigest(),
        "max_seq_length": max_tokens, "count": len(rows)}, indent=2))


def provenance(path):
    sidecar = Path(path).with_suffix(".metadata.json")
    if not sidecar.exists():
        raise ValueError(f"missing embedding provenance: {sidecar}")
    return json.loads(sidecar.read_text())


def grouped_metrics(labels, groups, values, bootstrap=1000, seed=0):
    known = labels >= 0
    labels, groups = labels[known], groups[known]
    if len(np.unique(labels)) != 2:
        return None
    result = {}
    unique = np.unique(groups)
    index = [np.where(groups == group)[0] for group in unique]
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(unique), size=(bootstrap, len(unique)))
    for name, full in values.items():
        score = full[known]
        result[name] = {"auprc": float(average_precision_score(labels, score)),
                        "auroc": float(roc_auc_score(labels, score))}
        for metric, fn in (("auprc", average_precision_score), ("auroc", roc_auc_score)):
            samples = []
            for draw in draws:
                ids = np.concatenate([index[i] for i in draw])
                if len(np.unique(labels[ids])) == 2:
                    samples.append(fn(labels[ids], score[ids]))
            result[name][f"group_bootstrap_{metric}_95pct"] = (
                np.quantile(samples, [0.025, 0.975]).tolist() if samples else None)
        for fraction in (0.05, 0.10, 0.20):
            count = max(1, int(np.ceil(len(labels) * fraction)))
            rank = np.argsort(-score)[:count]
            result[name][f"harm_recall_at_{int(fraction*100)}pct_rejected"] = float(
                labels[rank].sum() / labels.sum())
    if "epe" in values and "cosine" in values:
        paired = {}
        for metric, fn in (("auprc", average_precision_score), ("auroc", roc_auc_score)):
            differences = []
            for draw in draws:
                ids = np.concatenate([index[i] for i in draw])
                if len(np.unique(labels[ids])) == 2:
                    differences.append(fn(labels[ids], values["epe"][known][ids]) -
                                       fn(labels[ids], values["cosine"][known][ids]))
            paired[metric] = {"difference": result["epe"][metric] - result["cosine"][metric],
                              "group_bootstrap_95pct": np.quantile(differences, [0.025, 0.975]).tolist() if differences else None}
        result["epe_minus_cosine"] = paired
    return result


def diagnose(args):
    if args.split == "test" and len(args.ridge) != 1:
        raise ValueError("ridge sweep is validation-only; pass one preselected ridge for test")
    metadata = provenance(args.embeddings)
    with np.load(args.embeddings, allow_pickle=False) as data:
        results = [diagnose_arrays(data["original"], data["compressed"], data["splits"],
                                   data["groups"], ridge=ridge, split=args.split,
                                   bootstrap=args.bootstrap, seed=args.seed) for ridge in args.ridge]
    print(json.dumps({"encoder": metadata["encoder"], "split": args.split, "results": results}, indent=2))


def train(args):
    metadata = provenance(args.embeddings)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    with np.load(args.embeddings, allow_pickle=False) as data:
        x = torch.from_numpy(data["compressed"].copy()).float()
        y = torch.from_numpy(data["original"].copy()).float()
        splits = data["splits"].copy()
        labels = data["harm"].copy()
    train_ids = np.where(splits == "train")[0]
    valid_ids = np.where(splits == "validation")[0]
    if not len(train_ids) or not len(valid_ids):
        raise ValueError("both train and validation pairs are required")
    selection = ("auprc" if len(set(labels[valid_ids][labels[valid_ids] >= 0])) == 2 else "mse") if args.selection == "auto" else args.selection
    if selection == "auprc" and len(set(labels[valid_ids][labels[valid_ids] >= 0])) != 2:
        raise ValueError("validation AUPRC selection requires both labeled harm classes")
    device = torch.device(args.device)
    model = Predictor(x.shape[1], args.hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = float("-inf")
    stale = 0
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
            val_predictions = []
            for start in range(0, len(valid_ids), args.batch_size):
                ids = valid_ids[start:start + args.batch_size]
                val_predictions.append(torch.nn.functional.normalize(model(x[ids].to(device)), dim=-1).cpu())
            val_prediction = torch.cat(val_predictions)
            val_loss = torch.nn.functional.mse_loss(val_prediction, y[valid_ids]).item()
            residuals = ((val_prediction - y[valid_ids]) ** 2).sum(-1).numpy()
        labeled = labels[valid_ids] >= 0
        val_auprc = (float(average_precision_score(labels[valid_ids][labeled], residuals[labeled]))
                     if len(set(labels[valid_ids][labeled])) == 2 else None)
        criterion = val_auprc if selection == "auprc" else -val_loss
        print(json.dumps({"epoch": epoch + 1, "validation_mse": val_loss, "validation_auprc": val_auprc}))
        if criterion > best:
            best = criterion
            stale = 0
            torch.save({"state_dict": model.state_dict(), "dim": x.shape[1],
                        "hidden": args.hidden, "seed": args.seed, "encoder": metadata["encoder"],
                        "encoder_revision": metadata.get("encoder_revision"),
                        "embedding_sha256": hashlib.sha256(Path(args.embeddings).read_bytes()).hexdigest(),
                        "training": {"lr": args.lr, "epochs": args.epochs, "batch_size": args.batch_size,
                                     "weight_decay": args.weight_decay, "selection": selection,
                                     "best_epoch": epoch + 1}}, output)
        else:
            stale += 1
            if stale >= args.patience:
                break


def evaluate(args):
    metadata = provenance(args.embeddings)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if "encoder" not in checkpoint or "embedding_sha256" not in checkpoint:
        raise ValueError("legacy checkpoint lacks provenance; retrain with this version")
    if (checkpoint["encoder"] != metadata["encoder"] or
            checkpoint.get("encoder_revision") != metadata.get("encoder_revision") or
            checkpoint["embedding_sha256"] != hashlib.sha256(Path(args.embeddings).read_bytes()).hexdigest()):
        raise ValueError("checkpoint and embeddings provenance mismatch")
    device = torch.device(args.device)
    model = Predictor(checkpoint["dim"], checkpoint["hidden"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with np.load(args.embeddings, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    selected = np.where(arrays["splits"] == args.split)[0]
    if not len(selected):
        raise ValueError(f"no rows for split {args.split}")
    if arrays["original"].shape[1] != checkpoint["dim"]:
        raise ValueError("embedding dimension differs from checkpoint")
    cos_parts, epe_parts, pred_parts = [], [], []
    with torch.no_grad():
        for start in range(0, len(selected), args.batch_size):
            ids = selected[start:start + args.batch_size]
            originals = torch.from_numpy(arrays["original"][ids]).float().to(device)
            compressed = torch.from_numpy(arrays["compressed"][ids]).float().to(device)
            cosine, epe = scores(originals, compressed, model)
            predicted = torch.nn.functional.normalize(model(torch.nn.functional.normalize(compressed, dim=-1)), dim=-1)
            pred_cos = (predicted * torch.nn.functional.normalize(compressed, dim=-1)).sum(-1)
            cos_parts.append(cosine.cpu().numpy())
            epe_parts.append(epe.cpu().numpy())
            pred_parts.append(pred_cos.cpu().numpy())
    cosine, epe, prediction_cosine = map(np.concatenate, (cos_parts, epe_parts, pred_parts))
    raw_ratio = arrays["compression_ratio"][selected]
    ratio = np.maximum(0, 1 - raw_ratio)
    labels = arrays["harm"][selected]
    train_mask = (arrays["splits"] == "train") & (arrays["harm"] >= 0)
    supervised = None
    if len(np.unique(arrays["harm"][train_mask])) == 2:
        features = np.concatenate([arrays["original"], arrays["compressed"],
                                   arrays["original"] - arrays["compressed"]], axis=1)
        train_groups, inverse, counts = np.unique(arrays["groups"][train_mask], return_inverse=True, return_counts=True)
        weights = 1 / counts[inverse]
        weights *= len(weights) / weights.sum()
        baseline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=args.seed))
        baseline.fit(features[train_mask], arrays["harm"][train_mask], logisticregression__sample_weight=weights)
        supervised = baseline.predict_proba(features[selected])[:, 1]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for j, idx in enumerate(selected):
            record = {"id": str(arrays["ids"][idx]), "group_id": str(arrays["groups"][idx]),
                      "split": args.split}
            record.update(harm=int(labels[j]), cosine=float(cosine[j]), epe=float(epe[j]),
                          identity_epe=float(2 * cosine[j]), compression_removed_fraction=float(ratio[j]),
                          compression_word_ratio=float(raw_ratio[j]),
                          prediction_cosine_to_compressed=float(prediction_cosine[j]))
            if supervised is not None:
                record["supervised_probability"] = float(supervised[j])
            handle.write(json.dumps(record) + "\n")
    values = {"cosine": cosine, "epe": epe, "compression_removed_fraction": ratio}
    if supervised is not None:
        values["supervised_probability"] = supervised
    metrics = grouped_metrics(labels, arrays["groups"][selected], values,
                              bootstrap=args.bootstrap, seed=args.seed)
    print(json.dumps({"mean_prediction_cosine_to_compressed": float(prediction_cosine.mean()),
                      "mean_absolute_epe_minus_identity": float(np.abs(epe - 2 * cosine).mean()),
                      "metrics": metrics}, indent=2))
    if metrics is None:
        print(f"wrote {len(selected)} scores; metrics need labeled examples from both classes")
    else:
        print(f"wrote {len(selected)} scores")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("embed")
    p.add_argument("--pairs", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--encoder", required=True, help="explicit encoder model; run separate embeddings for ablations")
    p.add_argument("--encoder-revision", required=True, help="pinned model commit or revision")
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=64)
    p.set_defaults(run=embed)
    p = sub.add_parser("inspect-lengths", help="measure tokenizer limits before embedding")
    p.add_argument("--pairs", required=True)
    p.add_argument("--encoder", required=True)
    p.add_argument("--encoder-revision", required=True)
    p.add_argument("--device", default="cpu")
    p.set_defaults(run=inspect_lengths)
    p = sub.add_parser("diagnose", help="held-out label-free linear map versus identity")
    p.add_argument("--embeddings", required=True)
    p.add_argument("--split", choices=["validation", "test"], default="validation")
    p.add_argument("--ridge", type=float, nargs="+", default=[1e-3], help="fixed grid on validation; single value on test")
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(run=diagnose)
    p = sub.add_parser("train")
    p.add_argument("--embeddings", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--hidden", type=int, default=1536)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--selection", choices=["auto", "mse", "auprc"], default="auto",
                   help="choose checkpoint on validation; AUPRC needs both harm classes")
    p.set_defaults(run=train)
    p = sub.add_parser("evaluate")
    p.add_argument("--embeddings", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", choices=["validation", "test"], default="test")
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(run=evaluate)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
