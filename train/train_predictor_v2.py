#!/usr/bin/env python3
"""
train/train_predictor_v2.py

Trains PreLNTransformerPredictor on pairs_v2.jsonl using triplet margin loss.
Accepts a list of learning rates; trains a separate run for each and copies
the best checkpoint (lowest final loss) to --out-path.

CLI:
    python3.12 train/train_predictor_v2.py \
        --pairs train/data/pairs_v2.jsonl \
        --out-path checkpoints/predictor_v2_best.pt \
        --lr 3e-4 1e-4 \
        --epochs 100 \
        --batch-size 32
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from jeval.encoders.predictor_head import PreLNTransformerPredictor
from jeval.encoders.sentence_encoder import FrozenEncoder


# ── Dataset ───────────────────────────────────────────────────────────────────

class TripletDataset(Dataset):
    """
    Builds (anchor_emb, positive_emb, negative_emb) triplets from pairs_v2.jsonl.

    Faithful pairs become (compressed, original, random_hard_neg_original).
    Hard negative pairs are used as the negative pool for the anchor's triplets.
    """

    def __init__(self, pairs_path: Path, encoder: FrozenEncoder):
        faithful  = []
        hard_negs = []
        with pairs_path.open() as f:
            for line in f:
                row = json.loads(line)
                if row["label"] == "faithful":
                    faithful.append((row["compressed"], row["original"]))
                else:
                    hard_negs.append(row["original"])

        if not faithful or not hard_negs:
            raise ValueError(f"pairs file needs both faithful and hard_negative rows: {pairs_path}")

        print(f"  Encoding {len(faithful)} faithfuls + {len(hard_negs)} hard neg originals...")
        comp_texts = [c for c, _ in faithful]
        pos_texts  = [o for _, o in faithful]

        comp_embs = torch.from_numpy(encoder.encode(comp_texts)).float()
        pos_embs  = torch.from_numpy(encoder.encode(pos_texts)).float()
        neg_pool  = torch.from_numpy(encoder.encode(hard_negs)).float()

        n = len(faithful)
        self.triplets: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = [
            (comp_embs[i], pos_embs[i], neg_pool[i % len(hard_negs)])
            for i in range(n)
        ]
        print(f"  Built {len(self.triplets)} triplets")

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int):
        return self.triplets[idx]


# ── Training loop ─────────────────────────────────────────────────────────────

def train_one(
    pairs_path: Path,
    encoder: FrozenEncoder,
    lr: float,
    epochs: int,
    batch_size: int,
    margin: float,
    ckpt_path: Path,
) -> float:
    """Returns best loss for this lr run."""
    dataset = TripletDataset(pairs_path, encoder)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    pred      = PreLNTransformerPredictor(encoder.dim())
    optimizer = torch.optim.AdamW(pred.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pred.to(device)
    pred.train()

    best_loss = float("inf")
    print(f"\n  lr={lr}  device={device}  batches/epoch={len(loader)}")

    for epoch in range(epochs):
        epoch_loss = 0.0
        for comp_emb, pos_emb, neg_emb in loader:
            comp_emb = comp_emb.to(device)
            pos_emb  = pos_emb.to(device)
            neg_emb  = neg_emb.to(device)

            predicted = pred(comp_emb)
            d_pos = torch.sum((predicted - pos_emb) ** 2, dim=1)
            d_neg = torch.sum((predicted - neg_emb) ** 2, dim=1)
            loss  = torch.clamp(d_pos - d_neg + margin, min=0).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pred.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg = epoch_loss / max(len(loader), 1)

        if avg < best_loss:
            best_loss = avg
            torch.save(pred.state_dict(), ckpt_path)

        if (epoch + 1) % 10 == 0:
            print(f"    epoch {epoch+1:3d}/{epochs}  loss={avg:.6f}  best={best_loss:.6f}")

    return best_loss


def verify_separation(ckpt_path: Path, encoder: FrozenEncoder) -> float:
    """Compute faithful/lossy EPE separation ratio on held-out examples."""
    from jeval.epe.core import EPEComputer

    pred = PreLNTransformerPredictor(encoder.dim())
    pred.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    pred.eval()

    faithful_cases = [
        ("test failed accuracy 10%",
         "the long test suite did not pass accuracy around 10 percent"),
        ("JWT_SECRET mismatch fixed env.ts",
         "JWT_SECRET was JWT_KEY in production corrected in src/config/env.ts"),
        ("migration failed lock timeout 30s users table",
         "migration failed on staging lock timeout after 30s on users table 847 concurrent connections"),
    ]
    lossy_cases = [
        ("good morning great weekend",
         "modified src/auth.ts JWT_SECRET mismatch corrected"),
        ("standup no blockers",
         "migration failed lock timeout 847 connections rolled back"),
        ("task completed successfully",
         "step 79 agent rolled back migration due to lock timeout affecting 847 connections"),
    ]

    faithful_epes, lossy_epes = [], []
    for comp, orig in faithful_cases:
        o = encoder.encode([orig])[0]
        c = encoder.encode([comp])[0]
        with torch.no_grad():
            p = pred(torch.from_numpy(c).unsqueeze(0).float()).squeeze(0).numpy()
        faithful_epes.append(EPEComputer.epe_score(o, p))

    for comp, orig in lossy_cases:
        o = encoder.encode([orig])[0]
        c = encoder.encode([comp])[0]
        with torch.no_grad():
            p = pred(torch.from_numpy(c).unsqueeze(0).float()).squeeze(0).numpy()
        lossy_epes.append(EPEComputer.epe_score(o, p))

    mean_faithful = float(np.mean(faithful_epes))
    mean_lossy    = float(np.mean(lossy_epes))
    ratio = mean_lossy / max(mean_faithful, 1e-9)

    print(f"\n  EPE separation — faithful: {mean_faithful:.4f}  lossy: {mean_lossy:.4f}  ratio: {ratio:.2f}x")
    return ratio


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs",      default="train/data/pairs_v2.jsonl")
    parser.add_argument("--out-path",   default="checkpoints/predictor_v2_best.pt")
    parser.add_argument("--lr",         nargs="+", type=float, default=[3e-4, 1e-4])
    parser.add_argument("--epochs",     type=int,  default=100)
    parser.add_argument("--batch-size", type=int,  default=32)
    parser.add_argument("--margin",     type=float, default=0.5)
    args = parser.parse_args()

    pairs_path = Path(args.pairs)
    out_path   = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading encoder (frozen — loaded once, shared across lr runs)...")
    encoder = FrozenEncoder()

    best_overall_loss = float("inf")
    best_ckpt: Path | None = None

    for lr in args.lr:
        ckpt = out_path.parent / f"predictor_v2_lr{lr:.0e}.pt"
        print(f"\nTraining with lr={lr}  →  {ckpt}")
        t0 = time.time()
        loss = train_one(pairs_path, encoder, lr, args.epochs, args.batch_size, args.margin, ckpt)
        elapsed = time.time() - t0
        print(f"  finished in {elapsed/60:.1f}m  best_loss={loss:.6f}")

        if loss < best_overall_loss:
            best_overall_loss = loss
            best_ckpt = ckpt

    shutil.copy(best_ckpt, out_path)
    print(f"\nBest checkpoint: {best_ckpt} (loss={best_overall_loss:.6f})")
    print(f"Copied to: {out_path}")

    ratio = verify_separation(out_path, encoder)
    if ratio >= 5.0:
        print(f"\nTarget separation (5x+) achieved: {ratio:.2f}x")
    else:
        print(f"\nSeparation {ratio:.2f}x — below 5x target. Consider more epochs or data.")


if __name__ == "__main__":
    main()
