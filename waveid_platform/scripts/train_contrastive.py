"""
Train contrastive audio encoder on anchor/positive/negative pairs.

Usage:
    python -m scripts.train_contrastive --data-dir "data/embeddings/contrastive_pairs" --output-dir "data/models" --epochs 20 --batch-size 32
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from waveid_backend.config import EMBEDDING_DIM
from waveid_backend.services.contrastive_model import AudioEncoder, triplet_loss


def main() -> int:
    parser = argparse.ArgumentParser(description="Train contrastive encoder.")
    parser.add_argument("--data-dir", default="data/embeddings/contrastive_pairs", help="Directory with anchors/positives/negatives .npy files.")
    parser.add_argument("--output-dir", default="data/models", help="Where to save the trained model.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--margin", type=float, default=0.2, help="Triplet loss margin.")
    parser.add_argument("--embedding-dim", type=int, default=EMBEDDING_DIM, help="Embedding dimension.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    for name in ("anchors", "positives", "negatives"):
        p = data_dir / f"{name}.npy"
        if not p.exists():
            print(f"Missing {p}. Run create_contrastive_data first.")
            return 1

    # Load the pre-generated triplet arrays (created by create_contrastive_data.py)
    anchors = np.load(data_dir / "anchors.npy").astype(np.float32)
    positives = np.load(data_dir / "positives.npy").astype(np.float32)
    negatives = np.load(data_dir / "negatives.npy").astype(np.float32)

    # Add a channel dimension so the shape is (N, 1, samples) as expected by the 1D CNN
    anchors = anchors[:, np.newaxis, :]
    positives = positives[:, np.newaxis, :]
    negatives = negatives[:, np.newaxis, :]

    dataset = TensorDataset(
        torch.from_numpy(anchors),
        torch.from_numpy(positives),
        torch.from_numpy(negatives),
    )
    # shuffle=True mixes up the triplets each epoch so the model doesn't learn their order
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # Use GPU if available, otherwise fall back to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioEncoder(embedding_dim=args.embedding_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()  # switch to training mode (enables gradient tracking)
        total_loss = 0.0
        n_batches = 0
        for a, p, n in loader:
            a, p, n = a.to(device), p.to(device), n.to(device)
            opt.zero_grad()               # clear gradients from the previous step
            ea = model(a)                 # fingerprint the anchor clips
            ep = model(p)                 # fingerprint the positive (distorted) clips
            en = model(n)                 # fingerprint the negative (wrong-track) clips
            loss = triplet_loss(ea, ep, en, margin=args.margin)
            loss.backward()               # compute how to adjust each weight
            opt.step()                    # apply the adjustments
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        print(f"Epoch {epoch + 1}/{args.epochs}  loss={avg_loss:.4f}")

    model_path = output_dir / "contrastive_encoder.pt"
    torch.save(
        {"state_dict": model.state_dict(), "embedding_dim": args.embedding_dim},
        model_path,
    )
    print(f"Saved model to {model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
