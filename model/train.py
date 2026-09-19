"""
fpGPT Model — Character-Level Training Script

Trains a MicroGPT model on a text corpus (e.g., Shakespeare, code snippets)
to learn character-level language modeling. The trained weights are then
fed into the fpGPT compiler for quantization and Verilog generation.

Usage:
    python -m model.train --data data/input.txt --epochs 50
    python -m model.train --data data/input.txt --epochs 50 --d_model 64 --num_layers 4
"""

import argparse
import os
import time
import math

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from .micro_gpt import MicroGPT, CharTokenizer, DEFAULT_CONFIG


# ─────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────
class CharDataset(Dataset):
    """
    Reads a text file and creates overlapping sequences of fixed length
    for character-level language modeling.
    """

    def __init__(self, text: str, tokenizer: CharTokenizer, seq_len: int):
        self.tokenizer = tokenizer
        self.seq_len = seq_len

        # Encode entire corpus
        self.data = tokenizer.encode(text)
        # Filter out <pad> (unmapped characters)
        self.data = [t for t in self.data if t != 0]
        self.data = torch.tensor(self.data, dtype=torch.long)

    def __len__(self):
        return max(0, len(self.data) - self.seq_len - 1)

    def __getitem__(self, idx):
        chunk = self.data[idx: idx + self.seq_len + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


# ─────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[fpGPT] Training on: {device}")

    # Build config from args
    config = DEFAULT_CONFIG.copy()
    config["d_model"] = args.d_model
    config["num_heads"] = args.num_heads
    config["num_layers"] = args.num_layers
    config["d_ff"] = args.d_ff
    config["max_seq_len"] = args.seq_len
    config["vocab_size"] = args.vocab_size
    config["dropout"] = args.dropout

    # Tokenizer
    tokenizer = CharTokenizer(vocab_size=config["vocab_size"])

    # Load training data
    if args.data and os.path.isfile(args.data):
        with open(args.data, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        print(f"[fpGPT] Loaded {len(text):,} characters from {args.data}")
    else:
        # Default: tiny Shakespeare-like corpus for testing
        text = (
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
        ) * 500  # Repeat to get enough data
        print(f"[fpGPT] Using built-in sample text ({len(text):,} characters)")

    dataset = CharDataset(text, tokenizer, config["max_seq_len"])
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    # Build model
    model = MicroGPT(config).to(device)
    print(model.summary())

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Cosine annealing LR schedule
    total_steps = args.epochs * len(dataloader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=args.lr * 0.1
    )

    # Training
    print(f"\n[fpGPT] Starting training: {args.epochs} epochs, "
          f"{len(dataloader)} batches/epoch")
    print("=" * 60)

    best_loss = float('inf')
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        t0 = time.time()

        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            logits, loss = model(x, targets=y)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        elapsed = time.time() - t0
        ppl = math.exp(avg_loss) if avg_loss < 20 else float('inf')

        print(f"  Epoch {epoch + 1:3d}/{args.epochs} | "
              f"Loss: {avg_loss:.4f} | PPL: {ppl:.1f} | "
              f"LR: {scheduler.get_last_lr()[0]:.2e} | "
              f"Time: {elapsed:.1f}s")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs(args.output_dir, exist_ok=True)
            save_path = os.path.join(args.output_dir, "micro_gpt.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": config,
                "tokenizer_vocab_size": tokenizer.vocab_size,
                "epoch": epoch,
                "loss": avg_loss,
            }, save_path)

        # Generate sample every N epochs
        if (epoch + 1) % args.sample_every == 0 or epoch == args.epochs - 1:
            model.eval()
            prompt = "To be"
            prompt_ids = tokenizer.encode(prompt)
            idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            generated = model.generate(idx, max_new_tokens=100, temperature=0.8, top_k=10)
            text_out = tokenizer.decode(generated[0].tolist())
            print(f"  -- Sample: \"{text_out[:120]}\"")

    print("=" * 60)
    print(f"[fpGPT] Training complete. Best loss: {best_loss:.4f}")
    print(f"[fpGPT] Model saved to: {os.path.join(args.output_dir, 'micro_gpt.pt')}")
    print(f"\n  Next step: Run the compiler to generate Verilog:")
    print(f"    python compile.py --model {os.path.join(args.output_dir, 'micro_gpt.pt')} "
          f"--out build/")


def main():
    parser = argparse.ArgumentParser(description="Train a MicroGPT model for FPGA synthesis")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to training text file")
    parser.add_argument("--output_dir", type=str, default="checkpoints",
                        help="Directory to save model checkpoints")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=3e-3,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay for AdamW")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate")
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
