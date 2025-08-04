#!/usr/bin/env python3
"""
Embed DNA sequences with Together-AI Evo-1 (StripedHyena) fully offline.

• Accepts a FASTA file or falls back to three toy DNA sequences.
• Requires a local clone/symlink of models/evo-1-131k-base (code + weights).
• Pools the mean hidden state from any layer (default = final).
• Works even though Evo-1 never returns `hidden_states`.
• Writes embeddings.tsv  →  seq_id  e0  e1 … eN
"""

import argparse
import csv
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM


# ───────────────────────── helpers ──────────────────────────
def load_sequences(path):
    if path:
        from Bio import SeqIO
        return [(rec.id, str(rec.seq)) for rec in SeqIO.parse(path, "fasta")]
    # demo DNA
    return [
        ("seq1", "ATGCGTACGTAGCTAGTACGTA"),
        ("seq2", "CGTATCGATCGATCGTACGTAGCTA"),
        ("seq3", "TTGACGTAGCTAGCTAGCATCGATC"),
    ]


def pick_device(flag):
    if flag == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return flag


def final_hidden(model, input_ids, attention_mask):
    """
    Return the final hidden representation *before* it is projected to logits.
    Shape: [seq_len, hidden_size]
    """
    with torch.no_grad():
        x = model.backbone.embedding_layer.embed(input_ids)
        x, _ = model.backbone.stateless_forward(x, padding_mask=attention_mask)
        x = model.backbone.norm(x)
    return x.squeeze(0)  # [L, D]


def hidden_at_layer(model, input_ids, attention_mask, layer_idx, total_layers):
    """
    Capture the hidden state after the requested layer (1-based index).
    """
    if layer_idx == total_layers:
        return final_hidden(model, input_ids, attention_mask)

    grab = {}

    def tap(_, __, output):
        # output is (tensor, inference_params); we need the tensor
        grab["hid"] = output[0].detach()

    handle = model.backbone.blocks[layer_idx - 1].register_forward_hook(tap)
    # run a forward pass so the hook fires
    final_hidden(model, input_ids, attention_mask)
    handle.remove()

    if "hid" not in grab:
        sys.exit("❌ Failed to capture hidden state — check layer index")
    return grab["hid"].squeeze(0)  # [L, D]


# ───────────────────────── CLI ──────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model", default="models/evo-1-131k-base",
                   help="Local path to evo-1-131k-base (code + weights)")
    p.add_argument("--layer", type=int, default=None,
                   help="Hidden layer to extract (1 = first, default = final)")
    p.add_argument("--fasta", metavar="FILE.fasta",
                   help="Input FASTA; if omitted, uses three demo DNA sequences")
    p.add_argument("--output", default="embeddings.tsv",
                   help="Destination TSV file")
    p.add_argument("--device", default="auto",
                   help="'cpu', 'cuda', or 'auto'")
    return p.parse_args()


# ───────────────────────── main ──────────────────────────
def main():
    args = parse_args()
    mpath = Path(args.model)
    if not mpath.is_dir():
        sys.exit(f"❌ '{mpath}' not found — clone the repo into models/ first")

    device = pick_device(args.device)
    print(f"✔ Loading Evo-1 from {mpath} on {device}")

    cfg = AutoConfig.from_pretrained(
        mpath, trust_remote_code=True, local_files_only=True)
    tok = AutoTokenizer.from_pretrained(
        mpath, trust_remote_code=True, local_files_only=True)
    model = (AutoModelForCausalLM
             .from_pretrained(mpath, config=cfg,
                              trust_remote_code=True, local_files_only=True)
             .eval()
             .to(device))

    # layer count comes straight from the config
    total_layers = getattr(cfg, "num_layers",
                           getattr(cfg, "num_hidden_layers", None))
    if total_layers is None:
        sys.exit("❌ Couldn’t find the layer count in the config")

    target_layer = args.layer or total_layers
    if not (1 <= target_layer <= total_layers):
        sys.exit(f"⚠️ Invalid --layer {args.layer}; model has {total_layers} layers")
    print(f"✔ Using hidden layer {target_layer} of {total_layers}")

    seqs = load_sequences(args.fasta)
    if not seqs:
        sys.exit("No sequences supplied!")
    ids, strings = zip(*seqs)

    embeds = []
    bos, eos = tok.bos_token_id, tok.eos_token_id

    for sid, seq in zip(ids, strings):
        toks = tok(seq, return_tensors="pt")  # no padding
        input_ids = toks["input_ids"].to(device)
        mask = toks["attention_mask"].to(device)

        hid = hidden_at_layer(model, input_ids, mask,
                              target_layer, total_layers)  # [L, D]
        valid = (input_ids[0] != bos) & (input_ids[0] != eos)
        embeds.append(hid[valid].mean(0).cpu())

    # save TSV
    dim = embeds[0].shape[0]
    header = ["seq_id"] + [f"e{k}" for k in range(dim)]
    with Path(args.output).open("w", newline="") as fh:
        wr = csv.writer(fh, delimiter="\t")
        wr.writerow(header)
        for sid, emb in zip(ids, embeds):
            wr.writerow([sid] + emb.tolist())

    print(f"🎉 Wrote {len(embeds)} embeddings → {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
