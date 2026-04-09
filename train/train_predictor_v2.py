#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from jeval.encoders.predictor_head import PreLNTransformerPredictor
from jeval.encoders.sentence_encoder import FrozenEncoder


# ── datasets ──────────────────────────────────────────────────────────────────

class EmbeddingPairDataset(Dataset):
    def __init__(self, pairs_path: Path, encoder: FrozenEncoder, label: str):
        pairs: List[tuple] = []
        with pairs_path.open() as f:
            for line in f:
                row = json.loads(line)
                if row["label"] == label:
                    pairs.append((row["compressed"], row["original"]))

        if not pairs:
            raise ValueError(f"no '{label}' rows found in {pairs_path}")

        print(f"  encoding {len(pairs)} {label} pairs...")
        self.comp_embs = torch.from_numpy(
            encoder.encode([c for c, _ in pairs])
        ).float()
        self.orig_embs = torch.from_numpy(
            encoder.encode([o for _, o in pairs])
        ).float()
        print(f"  {label} dataset size: {len(self.comp_embs)}")

    def __len__(self) -> int:
        return len(self.comp_embs)

    def __getitem__(self, idx: int):
        return self.comp_embs[idx], self.orig_embs[idx]


# ── training loop ─────────────────────────────────────────────────────────────

def train_one(
    pairs_path: Path,
    encoder: FrozenEncoder,
    lr: float,
    epochs: int,
    batch_size: int,
    margin: float,
    alpha: float,
    ckpt_path: Path,
) -> float:
    faithful_ds   = EmbeddingPairDataset(pairs_path, encoder, "faithful")
    hard_neg_ds   = EmbeddingPairDataset(pairs_path, encoder, "hard_negative")

    faithful_loader = DataLoader(faithful_ds, batch_size=batch_size,
                                 shuffle=True, num_workers=2, drop_last=True)
    hard_neg_loader = DataLoader(hard_neg_ds, batch_size=batch_size,
                                 shuffle=True, num_workers=2, drop_last=True)

    pred      = PreLNTransformerPredictor(encoder.dim())
    optimizer = torch.optim.AdamW(pred.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pred.to(device)
    pred.train()

    n_batches = min(len(faithful_loader), len(hard_neg_loader))
    best_loss = float("inf")
    print(f"\n  lr={lr}  margin={margin}  alpha={alpha}  device={device}  batches/epoch={n_batches}")

    for epoch in range(epochs):
        epoch_loss = 0.0
        faithful_iter = iter(faithful_loader)
        hard_neg_iter = iter(hard_neg_loader)

        for _ in range(n_batches):
            f_comp, f_orig = next(faithful_iter)
            h_comp, h_orig = next(hard_neg_iter)

            f_comp = f_comp.to(device)
            f_orig = f_orig.to(device)
            h_comp = h_comp.to(device)
            h_orig = h_orig.to(device)

            # faithful term: pred should be close to original
            pred_f = pred(f_comp)
            loss_faithful = F.mse_loss(pred_f, f_orig)

            # lossy term: pred should be far from original (hinge)
            pred_h = pred(h_comp)
            dist = F.mse_loss(pred_h, h_orig, reduction="none").mean(dim=-1)
            loss_lossy = F.relu(margin - dist).mean()

            loss = loss_faithful + alpha * loss_lossy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pred.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg = epoch_loss / max(n_batches, 1)

        if avg < best_loss:
            best_loss = avg
            torch.save(pred.state_dict(), ckpt_path)

        if (epoch + 1) % 10 == 0:
            print(f"    epoch {epoch+1:3d}/{epochs}  loss={avg:.6f}  best={best_loss:.6f}")

    return best_loss


def verify_separation(ckpt_path: Path, encoder: FrozenEncoder, pairs_path: Path) -> float:
    from jeval.epe.core import EPEComputer

    pred = PreLNTransformerPredictor(encoder.dim())
    pred.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    pred.eval()

    faithful_cases, lossy_cases = [], []
    with pairs_path.open() as f:
        for line in f:
            row = json.loads(line)
            if row["label"] == "faithful" and len(faithful_cases) < 50:
                faithful_cases.append((row["compressed"], row["original"]))
            elif row["label"] == "hard_negative" and len(lossy_cases) < 50:
                lossy_cases.append((row["compressed"], row["original"]))
            if len(faithful_cases) >= 50 and len(lossy_cases) >= 50:
                break

    def compute_epes(cases):
        epes = []
        for comp, orig in cases:
            o = encoder.encode([orig])[0]
            c = encoder.encode([comp])[0]
            with torch.no_grad():
                p = pred(torch.from_numpy(c).unsqueeze(0).float()).squeeze(0).numpy()
            epes.append(EPEComputer.epe_score(o, p))
        return epes

    faithful_epes = compute_epes(faithful_cases)
    lossy_epes    = compute_epes(lossy_cases)

    def cos_sims(cases):
        sims = []
        for comp, orig in cases:
            c = encoder.encode([comp])[0]
            o = encoder.encode([orig])[0]
            sims.append(float(np.dot(c, o) / (np.linalg.norm(c) * np.linalg.norm(o))))
        return sims

    f_sims = cos_sims(faithful_cases)
    l_sims = cos_sims(lossy_cases)

    mean_faithful = float(np.mean(faithful_epes))
    mean_lossy    = float(np.mean(lossy_epes))
    ratio = mean_lossy / max(mean_faithful, 1e-9)

    print(f"\n  faithful  cos_sim: {np.mean(f_sims):.4f} +/- {np.std(f_sims):.4f}")
    print(f"  lossy     cos_sim: {np.mean(l_sims):.4f} +/- {np.std(l_sims):.4f}")
    print(f"  gap: {np.mean(f_sims) - np.mean(l_sims):.4f}")
    print(f"  EPE separation — faithful: {mean_faithful:.4f}  lossy: {mean_lossy:.4f}  ratio: {ratio:.2f}x")
    return ratio


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs",       "--data", default="train/data/pairs_v3.jsonl")
    parser.add_argument("--out-path",    default="checkpoints/predictor_v3_best.pt")
    parser.add_argument("--lr",          nargs="+", type=float, default=[3e-4, 1e-4])
    parser.add_argument("--epochs",      type=int,   default=100)
    parser.add_argument("--batch-size",  type=int,   default=64)
    parser.add_argument("--margin",      type=float, default=0.3)
    parser.add_argument("--alpha",       type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.07)  # unused, kept for CLI compat
    args = parser.parse_args()

    pairs_path = Path(args.pairs)
    out_path   = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("loading encoder (frozen — shared across lr runs)...")
    encoder = FrozenEncoder()

    best_overall_loss = float("inf")
    best_ckpt = None

    for lr in args.lr:
        ckpt = out_path.parent / f"predictor_v3_lr{lr:.0e}.pt"
        print(f"\ntraining lr={lr}  →  {ckpt}")
        t0   = time.time()
        loss = train_one(pairs_path, encoder, lr, args.epochs,
                         args.batch_size, args.margin, args.alpha, ckpt)
        print(f"  finished in {(time.time()-t0)/60:.1f}m  best_loss={loss:.6f}")

        if loss < best_overall_loss:
            best_overall_loss = loss
            best_ckpt = ckpt

    shutil.copy(best_ckpt, out_path)
    print(f"\nbest checkpoint: {best_ckpt} (loss={best_overall_loss:.6f})")
    print(f"copied to: {out_path}")

    ratio = verify_separation(out_path, encoder, pairs_path)

    if ratio >= 5.0:
        print(f"\ntarget separation (5x+) achieved: {ratio:.2f}x")
    elif ratio >= 3.0:
        print(f"\nseparation {ratio:.2f}x — below 5x target, above 3x floor")
    else:
        print(f"\nWARNING: separation < 3x. Check pair quality before deploying.")


if __name__ == "__main__":
    main()
