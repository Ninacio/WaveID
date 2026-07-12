"""
Train contrastive audio encoder on anchor/positive/negative pairs.

Usage:
    python -m scripts.train_contrastive --data-dir "data/embeddings/contrastive_pairs" --output-dir "data/models" --epochs 20 --batch-size 32
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from waveid_backend.config import EMBEDDING_DIM
from waveid_backend.services.contrastive_model import AudioEncoder, triplet_loss


class MemmapTripletDataset(Dataset):
    """Reads anchor/positive/negative triplets straight from disk-backed
    memmap arrays, one sample at a time, instead of requiring the whole
    (potentially many-GB) dataset to be resident in RAM at once.

    Stores file paths rather than opened memmap objects: with multiple
    DataLoader worker processes (num_workers > 0), an already-opened memmap
    can't be pickled across Windows' spawn-based process creation, and even
    where it can, sharing one open memmap across processes serialises what
    should be parallel I/O. Each worker instead lazily opens its own memmap
    on first access and keeps it for the worker's lifetime.
    """

    def __init__(self, anchors_path: Path, positives_path: Path, negatives_path: Path, length: int) -> None:
        self._paths = {"anchors": anchors_path, "positives": positives_path, "negatives": negatives_path}
        self._length = length
        self._arrays: dict[str, np.memmap] | None = None

    def __len__(self) -> int:
        return self._length

    def _get_array(self, name: str) -> np.memmap:
        if self._arrays is None:
            self._arrays = {}
        if name not in self._arrays:
            self._arrays[name] = np.load(self._paths[name], mmap_mode="r")
        return self._arrays[name]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # np.newaxis adds the channel dimension expected by the 1D CNN: (1, samples)
        a = np.asarray(self._get_array("anchors")[idx])[np.newaxis, :].astype(np.float32)
        p = np.asarray(self._get_array("positives")[idx])[np.newaxis, :].astype(np.float32)
        n = np.asarray(self._get_array("negatives")[idx])[np.newaxis, :].astype(np.float32)
        return torch.from_numpy(a), torch.from_numpy(p), torch.from_numpy(n)


class BlockShuffleSampler(Sampler[int]):
    """Shuffles in fixed-size contiguous blocks instead of fully at random.

    A memmap-backed dataset this large (many GB across three arrays) doesn't fit in the
    OS page cache on a memory-constrained machine, so a naive full-random shuffle turns
    every single sample read into an unpredictable disk seek - in practice this made
    training over 100x slower than the model's actual compute time. Shuffling the order
    of blocks (and the items within each block) keeps each batch's reads clustered near
    each other on disk while still randomising which triplets land in which batch and
    which batches run early vs. late in the epoch.
    """

    def __init__(self, length: int, block_size: int, generator: torch.Generator, skip: int = 0) -> None:
        self.length = length
        self.block_size = max(1, block_size)
        self.generator = generator
        # Number of leading samples to omit when resuming mid-epoch. Skipping here (rather
        # than `continue`-ing in the training loop) matters: a loop-level skip still makes
        # the DataLoader read every skipped sample off disk, which for a deep resume point
        # means gigabytes of pointless I/O before any training happens.
        self.skip = skip

    def __len__(self) -> int:
        return self.length - self.skip

    def __iter__(self):
        n_blocks = math.ceil(self.length / self.block_size)
        order: list[int] = []
        for b in torch.randperm(n_blocks, generator=self.generator).tolist():
            start = b * self.block_size
            end = min(start + self.block_size, self.length)
            order.extend((torch.randperm(end - start, generator=self.generator) + start).tolist())
        return iter(order[self.skip:])


def main() -> int:
    parser = argparse.ArgumentParser(description="Train contrastive encoder.")
    parser.add_argument("--data-dir", default="data/embeddings/contrastive_pairs", help="Directory with anchors/positives/negatives .npy files.")
    parser.add_argument("--output-dir", default="data/models", help="Where to save the trained model.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument(
        "--num-workers", type=int, default=4,
        help="DataLoader worker processes for reading triplets off disk. On a large memmap-backed "
             "dataset, single-process random-access reads (num_workers=0) can become the bottleneck "
             "well before the CPU/GPU does any real work - parallel workers prefetch while the "
             "model trains on the previous batch.",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--margin", type=float, default=0.2, help="Triplet loss margin.")
    parser.add_argument("--embedding-dim", type=int, default=EMBEDDING_DIM, help="Embedding dimension.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume training from the checkpoint in --output-dir (model + optimizer state + epoch "
             "+ in-epoch batch offset), saved every --checkpoint-every-batches batches. Useful for "
             "long CPU runs that may get interrupted at any point, not just between epochs.",
    )
    parser.add_argument(
        "--log-every", type=int, default=20,
        help="Print progress every this many batches, so a run's health is visible even mid-epoch.",
    )
    parser.add_argument(
        "--checkpoint-every-batches", type=int, default=100,
        help="Save a resumable checkpoint every this many batches (in addition to after each epoch). "
             "0 disables mid-epoch checkpointing.",
    )
    parser.add_argument(
        "--shuffle-block-size", type=int, default=2048,
        help="Number of contiguous triplets treated as one shuffle block (see BlockShuffleSampler). "
             "Should be a multiple of --batch-size. Larger = more randomisation but bigger, more "
             "memory-pressuring reads per block; smaller = more disk-friendly but less shuffling.",
    )
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

    # Peek at the anchors array just to get the dataset length; the real (memory-mapped)
    # reads happen lazily per-worker inside MemmapTripletDataset. A large dataset (e.g. a
    # full FMA pass) can be many GB across the three arrays combined, so avoiding a full
    # materialisation here matters.
    num_triplets = np.load(data_dir / "anchors.npy", mmap_mode="r").shape[0]
    dataset = MemmapTripletDataset(
        data_dir / "anchors.npy", data_dir / "positives.npy", data_dir / "negatives.npy", num_triplets
    )
    n_batches_per_epoch = math.ceil(num_triplets / args.batch_size)

    # Use GPU if available, otherwise fall back to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioEncoder(embedding_dim=args.embedding_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "contrastive_encoder.pt"

    start_epoch = 0
    start_batch = 0
    if args.resume and model_path.exists():
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        if "optimizer_state_dict" in checkpoint:
            opt.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0)
        start_batch = checkpoint.get("batch_offset", 0)
        if start_batch:
            print(f"Resuming from checkpoint: epoch {start_epoch + 1}, batch {start_batch}/{n_batches_per_epoch} already done.", flush=True)
        else:
            print(f"Resuming from checkpoint: epoch {start_epoch} already completed.", flush=True)

    def _save_checkpoint(epoch: int, batch_offset: int) -> None:
        torch.save(
            {
                "state_dict": model.state_dict(),
                "embedding_dim": args.embedding_dim,
                "optimizer_state_dict": opt.state_dict(),
                "epoch": epoch,
                "batch_offset": batch_offset,
            },
            model_path,
        )

    for epoch in range(start_epoch, args.epochs):
        model.train()  # switch to training mode (enables gradient tracking)
        epoch_start_batch = start_batch if epoch == start_epoch else 0

        # A fresh loader (with a per-epoch seed) each epoch, rather than one long-lived loader,
        # so a resumed run can reproduce the same shuffle order and skip straight past the
        # batches it already trained on instead of redoing them.
        gen = torch.Generator()
        gen.manual_seed(args.seed + epoch)
        sampler = BlockShuffleSampler(
            num_triplets, args.shuffle_block_size, gen, skip=epoch_start_batch * args.batch_size
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            prefetch_factor=4 if args.num_workers > 0 else None,
        )

        total_loss = 0.0
        n_batches = 0
        t0 = time.time()
        for batch_idx, (a, p, n) in enumerate(loader, start=epoch_start_batch):
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

            done = batch_idx + 1
            if done % args.log_every == 0 or done == n_batches_per_epoch:
                elapsed = time.time() - t0
                rate = n_batches / elapsed if elapsed > 0 else 0.0
                print(
                    f"  epoch {epoch + 1}/{args.epochs} batch {done}/{n_batches_per_epoch} "
                    f"loss={loss.item():.4f} ({rate:.2f} batch/s, {elapsed:.0f}s elapsed)",
                    flush=True,
                )

            if args.checkpoint_every_batches and done % args.checkpoint_every_batches == 0:
                _save_checkpoint(epoch, done)

        avg_loss = total_loss / max(n_batches, 1)
        print(f"Epoch {epoch + 1}/{args.epochs}  loss={avg_loss:.4f}", flush=True)

        # Checkpoint after every epoch (not just mid-epoch) so a long CPU run that gets
        # interrupted can pick back up with --resume instead of losing all progress.
        _save_checkpoint(epoch + 1, 0)

    print(f"Saved model to {model_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
