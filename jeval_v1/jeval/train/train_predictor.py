#!/usr/bin/env python3
"""
jeval/train/train_predictor.py

Trains PreLNTransformerPredictor with two objectives:

1. train()             — MSE reconstruction on hand-written pairs (fast baseline)
2. train_contrastive() — Triplet margin loss on AMA-Bench SOFTWARE trajectories

Triplet loss forces separation:
    d(pred, positive) + margin < d(pred, negative)
which gives 5x+ EPE separation vs MSE alone.
"""
from __future__ import annotations

import json
import random
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from jeval.encoders.predictor_head import PreLNTransformerPredictor
from jeval.encoders.sentence_encoder import FrozenEncoder


# ── Hand-written training pairs (MSE baseline) ───────────────────────────────

TRAIN_PAIRS: List[Tuple[str, str]] = [
    # Faithful compressions
    ("test failed accuracy 10%",
     "the long test suite did not pass and accuracy was around 10 percent"),
    ("modified src/auth.ts added header check",
     "modified src/middleware/auth.ts to add missing Authorization header validation"),
    ("JWT_SECRET mismatch fixed in src/config/env.ts",
     "root cause found: JWT_SECRET was JWT_KEY in production, corrected in src/config/env.ts"),
    ("decided Redis over Postgres connection pool exhaustion",
     "decided to use Redis over Postgres for session storage due to connection pool exhaustion under load"),
    ("rejected localStorage XSS risk using httpOnly cookies",
     "rejected JWT stored in localStorage — XSS risk confirmed, switching to httpOnly cookies"),
    ("14 unit tests passing auth middleware",
     "created tests/auth.test.ts — 14 unit tests covering JWT middleware and cookie handling, all passing"),
    ("migration failed lock timeout 847 connections",
     "error: migration failed on staging — lock timeout after 30s on users table, 847 concurrent connections"),
    ("pt-osc succeeded 4m32s zero downtime",
     "pt-osc run succeeded on staging — migration completed in 4m32s with zero downtime"),
    ("payments-service port 3002 charge refund endpoints",
     "created services/payments-service/src/index.ts — Express app on port 3002 with /charge and /refund endpoints"),
    ("gRPC rejected proto schema versioning overhead",
     "rejected gRPC for inter-service comms — team unfamiliar, would require proto schema versioning overhead"),
    ("assertion error line 42 auth test",
     "AssertionError: expected 200 but got 401 at line 42 in auth integration test"),
    ("502 bad gateway payments-service not registered consul",
     "error 502 on POST /payments/charge — payments-service not registered in service discovery Consul"),
    # Unfaithful compressions
    ("good morning hope great weekend",
     "modified src/auth.ts JWT_SECRET mismatch fixed production"),
    ("team synced sprint goals no blockers",
     "PAYMENTS_STRIPE_SECRET_KEY rotated in Vault old key invalidated"),
    ("standup note coffee kitchen",
     "migration failed lock timeout 847 concurrent connections rolled back"),
    ("exciting sprint review today",
     "test failed 47 assertions failed auth.test.ts rate limiting broken"),
]


# ── AMA-Bench pair generation ─────────────────────────────────────────────────

def generate_pairs_from_ama_bench(
    n_episodes: int = 10,
) -> List[Tuple[str, str, str]]:
    """
    Load AMA-Bench SOFTWARE trajectories and generate labeled pairs.

    Returns list of (compressed, original, label) where label is
    'positive' (faithful compression) or 'negative' (unrelated content).
    """
    from datasets import load_dataset  # type: ignore

    ds = load_dataset("AMA-bench/AMA-bench", split="test")
    sw_episodes = [ep for ep in ds if ep["domain"] == "SOFTWARE"]

    pairs: List[Tuple[str, str, str]] = []
    for ep in sw_episodes[:n_episodes]:
        traj = ep["trajectory"]
        if not isinstance(traj, list):
            traj = json.loads(traj)

        actions: List[str] = []
        for turn in traj:
            act = turn.get("action", "")
            obs = turn.get("observation", "")
            if len(act) > 50 and "think" not in act.lower()[:20]:
                actions.append(act[:300])
            if len(obs) > 50 and "logged" not in obs.lower():
                actions.append(obs[:300])

        # Positive pairs: first-half words → full line
        for line in actions:
            words = line.split()
            if len(words) > 12:
                compressed = " ".join(words[: len(words) // 2])
                pairs.append((compressed, line, "positive"))

        # Hard negatives: random cross-pairs within episode
        if len(actions) > 4:
            for _ in range(min(10, len(actions) // 2)):
                a = random.choice(actions)
                b = random.choice(actions)
                if a != b:
                    pairs.append((a[:100], b[:300], "negative"))

    random.shuffle(pairs)
    return pairs


# ── Datasets ──────────────────────────────────────────────────────────────────

class PairDataset(Dataset):
    """
    General-purpose dataset for (anchor, target, label) pairs.
    Returns dicts with anchor_emb, target_emb, pred_emb (from predictor), label.
    """

    def __init__(self, pairs: List[Tuple[str, str, int]], encoder, predictor):
        self.items = []
        for anchor, target, label in pairs:
            anchor_emb = encoder.encode(anchor)
            target_emb = encoder.encode(target)
            pred_emb   = predictor(anchor_emb).squeeze()
            self.items.append({
                "anchor_emb": anchor_emb,
                "target_emb": target_emb,
                "pred_emb":   pred_emb,
                "label":      label,
            })

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        return self.items[idx]

class ReconstructionDataset(Dataset):
    """MSE training: (enc(compressed), enc(original)) pairs."""

    def __init__(self, pairs: List[Tuple[str, str]], encoder: FrozenEncoder):
        print(f"  Pre-encoding {len(pairs)} pairs...")
        self.compressed_embs = torch.from_numpy(
            encoder.encode([p[0] for p in pairs])
        )
        self.original_embs = torch.from_numpy(
            encoder.encode([p[1] for p in pairs])
        )
        print(f"  Encoded. Shape: {self.compressed_embs.shape}")

    def __len__(self) -> int:
        return len(self.compressed_embs)

    def __getitem__(self, idx: int):
        return self.compressed_embs[idx], self.original_embs[idx]


class ContrastivePairDataset(Dataset):
    """
    Triplet training: (enc(compressed), enc(positive), enc(negative)).

    pairs: list of (compressed, original, label) with label='positive'/'negative'
    """

    def __init__(
        self,
        pairs: List[Tuple[str, str, str]],
        encoder: FrozenEncoder,
    ):
        positives = [(c, o) for c, o, l in pairs if l == "positive"]
        negatives = [(c, o) for c, o, l in pairs if l == "negative"]

        print(f"  Encoding {len(positives)} positive, {len(negatives)} negative pairs...")

        pos_comp = torch.from_numpy(encoder.encode([p[0] for p in positives])).float()
        pos_orig = torch.from_numpy(encoder.encode([p[1] for p in positives])).float()
        neg_orig = torch.from_numpy(encoder.encode([p[1] for p in negatives])).float()

        n = min(len(positives), len(negatives))
        self.triplets = [
            (pos_comp[i], pos_orig[i], neg_orig[i % len(negatives)])
            for i in range(n)
        ]
        print(f"  Built {len(self.triplets)} triplets")

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int):
        return self.triplets[idx]


# ── Training functions ────────────────────────────────────────────────────────

def train(
    pairs: List[Tuple[str, str]] = None,
    epochs: int = 30,
    batch_size: int = 8,
    lr: float = 3e-4,
    out_path: str = "predictor_best.pt",
) -> PreLNTransformerPredictor:
    """MSE reconstruction training on hand-written pairs."""
    pairs = pairs or TRAIN_PAIRS
    print(f"Training predictor on {len(pairs)} pairs, {epochs} epochs")

    enc  = FrozenEncoder()
    pred = PreLNTransformerPredictor(enc.dim())

    dataset    = ReconstructionDataset(pairs, enc)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer  = torch.optim.AdamW(pred.parameters(), lr=lr, weight_decay=1e-4)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion  = nn.MSELoss()

    best_loss = float("inf")
    pred.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        for compressed_emb, original_emb in dataloader:
            predicted = pred(compressed_emb)
            loss = criterion(predicted, original_emb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pred.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg = epoch_loss / len(dataloader)

        if avg < best_loss:
            best_loss = avg
            torch.save(pred.state_dict(), out_path)

        if (epoch + 1) % 5 == 0:
            print(f"  epoch {epoch+1:3d}/{epochs}  loss={avg:.6f}  best={best_loss:.6f}")

    print(f"\nBest loss: {best_loss:.6f}")
    print(f"Saved to: {out_path}")
    pred.load_state_dict(torch.load(out_path, map_location="cpu"))
    pred.eval()
    return pred


def train_contrastive(
    pairs: List[Tuple[str, str, str]] = None,
    epochs: int = 100,
    batch_size: int = 16,
    lr: float = 3e-4,
    margin: float = 0.5,
    out_path: str = "predictor_best.pt",
) -> Tuple[PreLNTransformerPredictor, FrozenEncoder]:
    """
    Triplet margin loss training on AMA-Bench SOFTWARE pairs.

    Loss = max(0, d_pos - d_neg + margin)
    Forces: d(pred, positive) + margin < d(pred, negative)
    """
    if pairs is None:
        print("Generating pairs from AMA-Bench SOFTWARE trajectories...")
        pairs = generate_pairs_from_ama_bench()
        print(
            f"Generated {len(pairs)} pairs  "
            f"(pos={sum(1 for p in pairs if p[2]=='positive')}, "
            f"neg={sum(1 for p in pairs if p[2]=='negative')})"
        )

    enc  = FrozenEncoder()
    pred = PreLNTransformerPredictor(enc.dim())

    dataset    = ContrastivePairDataset(pairs, enc)
    loader     = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer  = torch.optim.AdamW(pred.parameters(), lr=lr, weight_decay=1e-4)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float("inf")
    pred.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        for comp_emb, pos_emb, neg_emb in loader:
            predicted = pred(comp_emb)
            # All vectors are already unit-normalized by the encoder / predictor head
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
            torch.save(pred.state_dict(), out_path)

        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch+1:3d}/{epochs}  loss={avg:.6f}  best={best_loss:.6f}")

    print(f"\nBest loss: {best_loss:.6f} → {out_path}")
    pred.load_state_dict(torch.load(out_path, map_location="cpu"))
    pred.eval()
    return pred, enc


# ── Verification ──────────────────────────────────────────────────────────────

def verify(pred: PreLNTransformerPredictor, enc: FrozenEncoder = None):
    """Sanity check: faithful compressions should have lower EPE than lossy ones."""
    if enc is None:
        enc = FrozenEncoder()

    from jeval.epe.core import EPEComputer

    test_cases = [
        ("test failed accuracy 10%",
         "the long test suite did not pass accuracy around 10 percent", "low"),
        ("JWT_SECRET mismatch fixed env.ts",
         "JWT_SECRET was JWT_KEY in production corrected in src/config/env.ts", "low"),
        ("good morning great weekend",
         "modified src/auth.ts JWT_SECRET mismatch corrected", "high"),
        ("standup no blockers",
         "migration failed lock timeout 847 connections rolled back", "high"),
    ]

    print("\nEPE verification (low=faithful, high=lossy):")
    for compressed, original, expected in test_cases:
        orig_emb = enc.encode([original])[0]
        comp_emb = enc.encode([compressed])[0]
        with torch.no_grad():
            pred_emb = pred(torch.from_numpy(comp_emb).unsqueeze(0).float())
            pred_emb = pred_emb.squeeze(0).numpy()
        epe = EPEComputer.epe_score(orig_emb, pred_emb)
        status = "✓" if (epe < 0.5) == (expected == "low") else "✗"
        print(f"  {status} EPE={epe:.4f} [{expected}]  \"{compressed[:40]}\"")


if __name__ == "__main__":
    pred, enc = train_contrastive()
    verify(pred, enc)
