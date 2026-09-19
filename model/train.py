"""
fpGPT Model — Character-Level Training Script

Trains a MicroGPT model on a text corpus to learn character-level language
modeling. The trained weights are then fed into the fpGPT compiler for
quantization and Verilog generation.

Usage:
    python -m model.train --data data/sample.txt --epochs 50
    python -m model.train --data data/input.txt --epochs 100 --d_model 64 --num_layers 4
    python -m model.train --resume checkpoints/micro_gpt.pt --epochs 50
"""

import argparse
import json
import math
import os
import random
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from .micro_gpt import MicroGPT, DEFAULT_CONFIG
from .tokenizer import CharTokenizer

BUILTIN_TEXT = (
    "To be, or not to be, that is the question:\n"
    "Whether 'tis nobler in the mind to suffer\n"
    "The slings and arrows of outrageous fortune,\n"
    "Or to take arms against a sea of troubles,\n"
    "And by opposing end them. To die: to sleep;\n"
    "No more; and by a sleep to say we end\n"
    "The heart-ache and the thousand natural shocks\n"
    "That flesh is heir to, 'tis a consummation\n"
    "Devoutly to be wish'd. To die, to sleep;\n"
    "To sleep: perchance to dream: ay, there's the rub;\n"
)


# ─────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────
class CharDataset(Dataset):
    """Overlapping fixed-length character windows for next-token prediction."""

    def __init__(self, text: str, tokenizer: CharTokenizer, seq_len: int):
        self.tokenizer = tokenizer
        self.seq_len = seq_len

        encoded = tokenizer.encode(text)
        # Drop <pad>: the corpus may contain characters outside the vocabulary.
        encoded = [t for t in encoded if t != tokenizer.pad_id]
        self.data = torch.tensor(encoded, dtype=torch.long)

    def __len__(self):
        return max(0, len(self.data) - self.seq_len - 1)

    def __getitem__(self, idx):
        chunk = self.data[idx: idx + self.seq_len + 1]
        return chunk[:-1], chunk[1:]


def load_text(path: str) -> str:
    if path and os.path.isfile(path):
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        print(f"[fpGPT] Loaded {len(text):,} characters from {path}")
        return text
    text = BUILTIN_TEXT * 500
    print(f"[fpGPT] Using built-in sample text ({len(text):,} characters)")
    return text


def split_text(text: str, val_frac: float):
    if val_frac <= 0 or len(text) < 2:
        return text, ""
    cut = int(len(text) * (1.0 - val_frac))
    cut = max(1, min(len(text) - 1, cut))
    return text[:cut], text[cut:]


def make_scheduler(optimizer, total_steps: int, warmup_frac: float, min_lr: float,
                   base_lr: float):
    warmup = max(1, int(total_steps * warmup_frac))

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        floor = min_lr / base_lr if base_lr > 0 else 0.0
        return floor + (1.0 - floor) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(path: str, model, optimizer, tokenizer, config, args,
                    epoch: int, train_loss: float, val_loss: float):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "tokenizer": tokenizer.state_dict(),
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "args": vars(args),
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
    }, path)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total, batches = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        _, loss = model(x, targets=y)
        total += loss.item()
        batches += 1
    return total / max(batches, 1)


# ─────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────
def train(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True

    device = torch.device(args.device if args.device else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[fpGPT] Training on: {device}")

    # ── Resume or build fresh ──
    if args.resume and os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        config = checkpoint["config"]
        tokenizer = CharTokenizer.from_state_dict(checkpoint["tokenizer"]) \
            if checkpoint.get("tokenizer") else CharTokenizer(config["vocab_size"])
        model = MicroGPT(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_val = float(checkpoint.get("val_loss", float("inf")))
        print(f"[fpGPT] Resumed {args.resume} at epoch {start_epoch}")
    else:
        config = DEFAULT_CONFIG.copy()
        config.update({
            "d_model": args.d_model,
            "num_heads": args.num_heads,
            "num_layers": args.num_layers,
            "d_ff": args.d_ff,
            "max_seq_len": args.seq_len,
            "vocab_size": args.vocab_size,
            "dropout": args.dropout,
        })
        tokenizer = CharTokenizer(vocab_size=config["vocab_size"])
        model = MicroGPT(config)
        start_epoch = 0
        best_val = float("inf")

    model = model.to(device)

    # ── Data ──
    text = load_text(args.data)
    train_text, val_text = split_text(text, args.val_frac)
    train_set = CharDataset(train_text, tokenizer, config["max_seq_len"])
    if len(train_set) == 0:
        raise SystemExit("training corpus is too small for the requested --seq_len")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              drop_last=len(train_set) > args.batch_size, num_workers=0)
    val_loader = None
    if val_text:
        val_set = CharDataset(val_text, tokenizer, config["max_seq_len"])
        if len(val_set) > 0:
            val_loader = DataLoader(val_set, batch_size=args.batch_size,
                                    shuffle=False, num_workers=0)
    print(model.summary())

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    if args.resume and os.path.isfile(args.resume) and \
            checkpoint.get("optimizer_state_dict"):
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except (ValueError, KeyError):
            print("[fpGPT] WARNING: optimizer state not restored")

    total_steps = max(1, args.epochs * len(train_loader))
    scheduler = make_scheduler(optimizer, total_steps, args.warmup_frac,
                               args.min_lr, args.lr)

    print(f"\n[fpGPT] Starting training: {args.epochs} epochs, "
          f"{len(train_loader)} batches/epoch, val={'yes' if val_loader else 'no'}")
    print("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)
    best_path = os.path.join(args.output_dir, "micro_gpt.pt")
    tokenizer.save(os.path.join(args.output_dir, "tokenizer.json"))

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss, num_batches = 0.0, 0
        t0 = time.time()

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            _, loss = model(x, targets=y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            num_batches += 1

        train_loss = total_loss / max(num_batches, 1)
        val_loss = evaluate(model, val_loader, device) if val_loader else train_loss
        monitor = val_loss if val_loader else train_loss
        elapsed = time.time() - t0
        ppl = math.exp(min(val_loss, 20))

        print(f"  Epoch {epoch + 1:3d}/{args.epochs} | "
              f"train {train_loss:.4f} | val {val_loss:.4f} | PPL {ppl:.1f} | "
              f"LR {scheduler.get_last_lr()[0]:.2e} | {elapsed:.1f}s")

        if monitor < best_val:
            best_val = monitor
            save_checkpoint(best_path, model, optimizer, tokenizer, config, args,
                            epoch, train_loss, val_loss)

        if (epoch + 1) % args.sample_every == 0 or epoch == args.epochs - 1:
            model.eval()
            prompt_ids = tokenizer.encode("To be")
            idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            generated = model.generate(idx, max_new_tokens=100,
                                       temperature=0.8, top_k=10)
            print(f"  -- Sample: \"{tokenizer.decode(generated[0].tolist())[:120]}\"")

    print("=" * 60)
    print(f"[fpGPT] Training complete. Best monitored loss: {best_val:.4f}")
    print(f"[fpGPT] Model saved to: {best_path}")
    print(f"\n  Next step: Run the compiler to generate Verilog:")
    print(f"    python compile.py --model {best_path} --out build/")


def main():
    parser = argparse.ArgumentParser(description="Train a MicroGPT model for FPGA synthesis")
    parser.add_argument("--data", type=str, default="data/sample.txt",
                        help="Path to training text file")
    parser.add_argument("--output_dir", type=str, default="checkpoints",
                        help="Directory to save model checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume training from a checkpoint")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=3e-3,
                        help="Peak learning rate")
    parser.add_argument("--min_lr", type=float, default=3e-4,
                        help="Final learning rate of the cosine schedule")
    parser.add_argument("--warmup_frac", type=float, default=0.1,
                        help="Fraction of steps used for linear warmup")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay for AdamW")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Gradient norm clip")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate")
    parser.add_argument("--val_frac", type=float, default=0.1,
                        help="Fraction of the corpus held out for validation")
    parser.add_argument("--seed", type=int, default=1337,
                        help="Random seed for reproducible runs")
    parser.add_argument("--device", type=str, default=None,
                        help="Force a device (e.g. cpu or cuda)")
    parser.add_argument("--sample_every", type=int, default=10,
                        help="Generate a sample every N epochs")

    # Model architecture
    parser.add_argument("--vocab_size", type=int, default=DEFAULT_CONFIG["vocab_size"])
    parser.add_argument("--seq_len", type=int, default=DEFAULT_CONFIG["max_seq_len"])
    parser.add_argument("--d_model", type=int, default=DEFAULT_CONFIG["d_model"])
    parser.add_argument("--num_heads", type=int, default=DEFAULT_CONFIG["num_heads"])
    parser.add_argument("--num_layers", type=int, default=DEFAULT_CONFIG["num_layers"])
    parser.add_argument("--d_ff", type=int, default=DEFAULT_CONFIG["d_ff"])

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
