#!/usr/bin/env python3
"""
prepare_sh_dataset_from_merops.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Build balanced CSVs for sequence-level classification:
Serine hydrolase (1) vs Non-serine hydrolase (0) from MEROPS protease.lib.

What it does
------------
- Reads MEROPS `protease.lib` (FASTA).
- Uses the MEROPS catalytic-type letter (e.g., S01.001) found in headers:
    S = serine peptidase  -> label 1 (serine hydrolase)
    A,C,M,T,G,N = non-serine peptidases -> label 0
    I (inhibitors), U (unknown), P (mixed) -> excluded
- De-duplicates exact sequences.
- Balances classes by down-sampling the majority class.
- Stratified split into train/val/test (defaults 80/10/10).
- Writes CSVs with columns: sequence,label

Usage
-----
python prepare_sh_dataset_from_merops.py \
    --protease-lib /path/to/protease.lib \
    --out-dir data_sh \
    --train 0.8 --val 0.1 --test 0.1 \
    --seed 42

Notes
-----
- MEROPS family/identifier prefixes encode catalytic type (e.g., S01, S01.001). We
  regex-scan FASTA headers for these tokens to assign labels robustly.
- By default we exclude inhibitors/unknown/mixed (I/U/P).
- Sequences are uppercased; whitespace and '*' are stripped.

References
----------
- MEROPS download page (what `protease.lib` is): EMBL-EBI.
- MEROPS classification (letter -> catalytic type).
"""

from __future__ import annotations
from pathlib import Path
import argparse, random, re, csv, sys
from typing import Iterable, Tuple, Optional, List, Dict

# Catalytic-type letters per MEROPS classification
POS_LETTER = "S"                       # serine peptidase -> positive (serine hydrolase)
NEG_LETTERS = {"A","C","M","T","G","N"}# non-serine peptidases -> negative
EXCLUDE_LETTERS = {"I","U","P"}        # inhibitors, unknown/ambiguous, mixed

MEROPS_ID_RE   = re.compile(r"\b([ACGMSTUGPNI])\d{2}\.\d{3}\b")   # e.g., S01.001
MEROPS_FAM_RE  = re.compile(r"\b([ACGMSTUGPNI])\d{1,2}[A-Z]?\b")  # e.g., S1, S1A (fallback)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Prepare balanced CSVs (train/val/test) for serine hydrolase classification from MEROPS protease.lib."
    )
    p.add_argument("--protease-lib", required=True, help="Path to MEROPS protease.lib (FASTA).")
    p.add_argument("--out-dir", default="data_sh", help="Output directory for CSVs.")
    p.add_argument("--train", type=float, default=0.8, help="Train fraction.")
    p.add_argument("--val",   type=float, default=0.1, help="Validation fraction.")
    p.add_argument("--test",  type=float, default=0.1, help="Test fraction.")
    p.add_argument("--seed",  type=int,   default=42,  help="Random seed.")
    p.add_argument("--minlen",type=int,   default=1,   help="Minimum sequence length to keep.")
    p.add_argument("--maxlen",type=int,   default=1000000, help="Maximum sequence length to keep.")
    return p.parse_args()

def read_fasta(path: Path) -> Iterable[Tuple[str, str]]:
    """Yield (header, sequence) from a FASTA file. Header excludes '>'."""
    header = None
    seq_chunks: List[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield (header, "".join(seq_chunks))
                header = line[1:].strip()
                seq_chunks = []
            else:
                seq_chunks.append(line.strip())
        if header is not None:
            yield (header, "".join(seq_chunks))

def sanitize_sequence(seq: str) -> str:
    """Uppercase; remove whitespace and '*' stop chars."""
    return seq.upper().replace(" ", "").replace("\t", "").replace("*", "")

def catalytic_letter_from_header(header: str) -> Optional[str]:
    """Extract MEROPS catalytic-type letter from common tokens in FASTA headers."""
    m = MEROPS_ID_RE.search(header)     # prefer a full MEROPS identifier, e.g. S01.001
    if m:
        return m.group(1)
    # fallback: family-like token (e.g., S1, S8A)
    m = MEROPS_FAM_RE.search(header)
    if m:
        return m.group(1)
    return None

def label_from_letter(letter: str) -> Optional[int]:
    if letter == POS_LETTER:
        return 1
    if letter in NEG_LETTERS:
        return 0
    if letter in EXCLUDE_LETTERS:
        return None
    # Unknown letter -> exclude
    return None

def stratified_balanced_split(
    positives: List[str],
    negatives: List[str],
    train_frac: float, val_frac: float, test_frac: float,
    rng: random.Random
) -> Dict[str, List[Tuple[str,int]]]:
    """Downsample to balance classes, then split each class by fractions."""
    assert abs((train_frac + val_frac + test_frac) - 1.0) < 1e-6, "Fractions must sum to 1.0"

    # Balance
    k = min(len(positives), len(negatives))
    rng.shuffle(positives)
    rng.shuffle(negatives)
    positives = positives[:k]
    negatives = negatives[:k]

    def split(lst: List[str]) -> Tuple[List[str], List[str], List[str]]:
        n = len(lst)
        n_train = int(round(n * train_frac))
        n_val   = int(round(n * val_frac))
        # ensure totals add up
        n_test  = n - n_train - n_val
        return lst[:n_train], lst[n_train:n_train+n_val], lst[n_train+n_val:]

    pos_tr, pos_va, pos_te = split(positives)
    neg_tr, neg_va, neg_te = split(negatives)

    tr = [(s,1) for s in pos_tr] + [(s,0) for s in neg_tr]
    va = [(s,1) for s in pos_va] + [(s,0) for s in neg_va]
    te = [(s,1) for s in pos_te] + [(s,0) for s in neg_te]

    rng.shuffle(tr); rng.shuffle(va); rng.shuffle(te)
    return {"train": tr, "validation": va, "test": te}

def write_csv(rows: List[Tuple[str,int]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sequence","label"])
        for seq, lab in rows:
            w.writerow([seq, lab])

def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    protease_path = Path(args.protease_lib)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seen = set()
    positives: List[str] = []
    negatives: List[str] = []

    kept, skipped_no_tag, skipped_excluded, skipped_len = 0, 0, 0, 0

    for header, raw_seq in read_fasta(protease_path):
        seq = sanitize_sequence(raw_seq)
        if not (args.minlen <= len(seq) <= args.maxlen):
            skipped_len += 1
            continue

        letter = catalytic_letter_from_header(header)
        if not letter:
            skipped_no_tag += 1
            continue

        label = label_from_letter(letter)
        if label is None:
            skipped_excluded += 1
            continue

        # dedupe by exact sequence
        if seq in seen:
            continue
        seen.add(seq)

        if label == 1:
            positives.append(seq)
        else:
            negatives.append(seq)
        kept += 1

    if not positives or not negatives:
        print(f"ERROR: Not enough data after filtering. Positives={len(positives)}, Negatives={len(negatives)}", file=sys.stderr)
        sys.exit(2)

    splits = stratified_balanced_split(
        positives, negatives,
        train_frac=args.train, val_frac=args.val, test_frac=args.test,
        rng=rng
    )

    write_csv(splits["train"],      out_dir / "train.csv")
    write_csv(splits["validation"], out_dir / "val.csv")
    write_csv(splits["test"],       out_dir / "test.csv")

    print("✅ Done.")
    print(f"Kept sequences: {kept}  (unique after dedupe)")
    print(f"  Positives (S*): {len(positives)}")
    print(f"  Negatives (!S): {len(negatives)}")
    print("Excluded/skipped:")
    print(f"  No MEROPS tag in header: {skipped_no_tag}")
    print(f"  Excluded catalytic types (I/U/P): {skipped_excluded}")
    print(f"  Length outside [{args.minlen}, {args.maxlen}]: {skipped_len}")
    b = min(len(positives), len(negatives))
    print(f"Balanced per-class count used: {b}")
    for split in ("train","validation","test"):
        rows = splits[split]
        n1 = sum(1 for _,y in rows if y==1)
        n0 = sum(1 for _,y in rows if y==0)
        print(f"  {split}: total={len(rows)}  pos={n1}  neg={n0}")

if __name__ == "__main__":
    main()
