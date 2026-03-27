from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from jeval.encoders.predictor_head import PreLNTransformerPredictor
from jeval.encoders.sentence_encoder import FrozenEncoder


class PairDataset(Dataset):
    """Dataset for anchor-target-label pairs."""

    def __init__(
        self,
        pairs: List[Tuple[str, str, int]],
        encoder: FrozenSentenceEncoder,
        predictor: PreLNTransformerPredictor,
    ):
        self.pairs = pairs
        self.encoder = encoder
        self.predictor = predictor

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        anchor_text, target_text, label = self.pairs[idx]
        anchor_emb = self.encoder.encode(anchor_text)
        target_emb = self.encoder.encode(target_text)
        pred_emb = self.predictor(torch.tensor(anchor_emb).unsqueeze(0))
        return {
            "anchor_emb": torch.tensor(anchor_emb),
            "target_emb": torch.tensor(target_emb),
            "pred_emb": pred_emb.squeeze(0),
            "label": torch.tensor(label, dtype=torch.float),
        }


def train_predictor(
    pairs: List[Tuple[str, str, int]],
    encoder: FrozenEncoder,
    predictor: PreLNTransformerPredictor,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
):
    """Train the predictor head using contrastive loss."""
    dataset = PairDataset(pairs, encoder, predictor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(predictor.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()  # For binary classification of similarity

    predictor.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            anchor_emb = batch["anchor_emb"]
            target_emb = batch["target_emb"]
            pred_emb = batch["pred_emb"]
            labels = batch["label"]

            # Compute similarity scores
            pred_sim = torch.sum(pred_emb * target_emb, dim=1)
            true_sim = torch.sum(anchor_emb * target_emb, dim=1)

            # Loss: predict similarity
            loss = criterion(pred_sim, true_sim)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")