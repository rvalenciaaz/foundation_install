#!/usr/bin/env python3
"""
Embed protein sequences with a Hugging‑Face ESM‑2 checkpoint (offline‑ready).

1. Accepts a FASTA file  OR  falls back to three demo sequences.
2. Loads a local model directory *or* a Hub repo‑ID from cache (always offline).
3. Pools per‑sequence embeddings (mean of non‑special tokens).
4. Saves them to embeddings.tsv   →   seq_id  e0  e1 … eN
"""

from pathlib import Path
import argparse, csv, os, sys, hashlib
import torch
from transformers import AutoTokenizer, EsmModel   # ≥ 4.30
# --------------------------------------------------------------------------- CLI
def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model", default="models/esm2_t6_8M_UR50D",
                   help="Local **directory** (recommended) or Hub repo‑ID of the model")
    p.add_argument("--layer", type=int, default=None,
                   help="Which hidden layer to extract (default = final)")
    p.add_argument("--fasta", metavar="FILE.fasta",
                   help="Input FASTA; if omitted, use three toy sequences")
    p.add_argument("--output", default="embeddings.tsv",
                   help="Destination TSV file")
    p.add_argument("--device", default="auto",
                   help="'cpu', 'cuda', or 'auto' for automatic choice")
    return p.parse_args()

# --------------------------------------------------------------------------- helpers
def load_sequences(path=None):
    if path:
        from Bio import SeqIO
        return [(rec.id, str(rec.seq)) for rec in SeqIO.parse(path, "fasta")]
    return [  # demo
        ("seq1", "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAKDLG"),
        ("seq2", "GGDGKTGRDGYTHRLVHFYEKLGLDI"),
        ("seq3", "MTESTAAAVVVTSTSSADNVSKR"),
    ]

def choose_device(flag):
    if flag == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return flag

# --------------------------------------------------------------------------- main
def main():
    args = parse_args()
    device = choose_device(args.device)

    # ------------------------------------------------------------------ model load
    # If `args.model` is a directory we use it directly; otherwise we assume it
    # is a repo‑ID *already present* in the HF cache and set `local_files_only=True`
    if Path(args.model).is_dir():
        model_src = Path(args.model)
    else:
        model_src = args.model  # e.g. "facebook/esm2_t6_8M_UR50D"

    print(f"✔ Loading model from {model_src} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_src,
                                              local_files_only=True,
                                              do_lower_case=False)
    model = (EsmModel
             .from_pretrained(model_src,
                              local_files_only=True,
                              output_hidden_states=True)
             .eval()
             .to(device))

    target_layer = args.layer or model.config.num_hidden_layers
    print(f"✔ Using hidden layer {target_layer} of {model.config.num_hidden_layers}")

    # ------------------------------------------------------------------ sequences
    seqs = load_sequences(args.fasta)
    if not seqs:
        sys.exit("No sequences supplied!")
    ids, strings = zip(*seqs)

    tok = tokenizer(list(strings),
                    padding=True,
                    truncation=False,
                    return_tensors="pt")
    input_ids = tok["input_ids"].to(device)
    mask = tok["attention_mask"].to(device)

    # ------------------------------------------------------------------ forward
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=mask)
    hidden = out.hidden_states[target_layer]       # [B, L, D]

    # ------------------------------------------------------------------ pool
    pad, cls, eos = (tokenizer.pad_token_id,
                     tokenizer.cls_token_id,
                     tokenizer.eos_token_id)

    embeds = []
    for i in range(len(strings)):
        valid = (input_ids[i] != pad) & (input_ids[i] != cls) & (input_ids[i] != eos)
        embeds.append(hidden[i][valid].mean(0).cpu())

    # ------------------------------------------------------------------ save
    dim = embeds[0].shape[0]
    header = ["seq_id"] + [f"e{k}" for k in range(dim)]
    out_path = Path(args.output)
    with out_path.open("w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(header)
        for sid, emb in zip(ids, embeds):
            writer.writerow([sid] + emb.tolist())

    print(f"🎉 Wrote {len(embeds)} embeddings → {out_path.resolve()}")

if __name__ == "__main__":
    main()
