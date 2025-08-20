#!/usr/bin/env python3
"""
finetune_esm2_lora_debug.py  (ESM -> attention.self.query/key/value)
- Targets exactly: esm.encoder.layer.*.attention.self.{query,key,value}
- Version-proof TrainingArguments (eval_strategy vs evaluation_strategy)
- Gradient checkpointing fix (enable_input_require_grads + use_cache=False)
- Optional gradient_checkpointing_kwargs={'use_reentrant': False}
- Debug probes to prove adapters are attached and get grads

CSV columns: sequence (str), label (0/1)
"""

from pathlib import Path
import argparse, numpy as np, torch, torch.nn as nn
from typing import List, Tuple

# ----------------------------- CLI --------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--train", required=True, help="Training CSV file")
    p.add_argument("--val",   required=True, help="Validation CSV file")
    p.add_argument("--test",  required=True, help="Held-out test CSV file")
    p.add_argument("--model", default="models/esm2_t36_3B_UR50D",
                   help="Local directory OR cached HF repo-ID (offline-capable)")
    p.add_argument("--output", default="ft_model_sh_lora",
                   help="Directory for LoRA adapters (and tokenizer)")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--bs",     type=int, default=16)
    p.add_argument("--lr",     type=float, default=5e-5)
    p.add_argument("--max_len",type=int, default=1022)  # ≤1022 for ESM-2
    p.add_argument("--seed",   type=int, default=42)
    p.add_argument("--quant",  choices=["none","8bit","4bit"], default="none",
                   help="Optional k-bit loading via bitsandbytes (safe fallback).")
    p.add_argument("--bf16",   action="store_true", help="Use bfloat16 if supported.")
    p.add_argument("--fp16",   action="store_true", help="Use float16 if supported.")
    p.add_argument("--no_gc",  action="store_true", help="Disable gradient checkpointing.")
    p.add_argument("--tf32",   action="store_true", help="Enable TF32 on Ampere+ GPUs.")
    p.add_argument("--workers",type=int, default=4, help="DataLoader workers per process")
    p.add_argument("--debug",  action="store_true", help="Run extra adapter/grads probes.")
    return p.parse_args()

# ----------------------------- helpers ----------------------------------------
def try_make_bnb_config(quant: str):
    if quant == "none" or not torch.cuda.is_available():
        return None, ("quantization disabled" if quant == "none" else "no CUDA device")
    try:
        from transformers import BitsAndBytesConfig
        import bitsandbytes as _bnb  # noqa: F401
    except Exception as e:
        return None, f"bitsandbytes unavailable: {e}"

    if quant == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0), "using 8-bit"
    else:
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
        return (BitsAndBytesConfig(load_in_4bit=True,
                                   bnb_4bit_quant_type="nf4",
                                   bnb_4bit_use_double_quant=True,
                                   bnb_4bit_compute_dtype=dtype),
                "using 4-bit")

def softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)

def roc_auc_mann_whitney(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    pos = y_true == 1; neg = ~pos
    n_pos = int(pos.sum()); n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0: return float("nan")
    order = np.argsort(y_score, kind="mergesort")
    scores_sorted = y_score[order]
    ranks = np.empty_like(y_score, dtype=float)
    n = y_score.shape[0]; i = 0
    while i < n:
        j = i + 1
        while j < n and scores_sorted[j] == scores_sorted[i]: j += 1
        avg_rank = 0.5 * ((i + 1) + j); ranks[order[i:j]] = avg_rank; i = j
    rank_pos_sum = ranks[pos].sum()
    return float((rank_pos_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))

def compute_metrics_local(pred) -> dict:
    logits = pred.predictions[0] if isinstance(pred.predictions, (tuple, list)) else pred.predictions
    labels = pred.label_ids
    logits = np.asarray(logits); labels = np.asarray(labels)
    preds = logits.argmax(-1)
    acc = float((preds == labels).mean())
    probs = softmax_np(logits)[:, 1]
    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(labels, probs))
    except Exception:
        auc = roc_auc_mann_whitney(labels, probs)
    return {"accuracy": acc, "auroc": auc}

# ---- DEBUG / probes ----------------------------------------------------------
def find_projection_modules_names(model: torch.nn.Module) -> list[str]:
    hits = []
    keys = ("q_proj","k_proj","v_proj","o_proj","out_proj","query","key","value")
    for n, _m in model.named_modules():
        if any(k in n for k in keys):
            hits.append(n)
    return hits

def sample_trainable_params(model: torch.nn.Module, limit: int = 20):
    return [(n, p.shape) for n, p in model.named_parameters() if p.requires_grad][:limit]

def list_lora_modules(model: torch.nn.Module, limit: int = 20):
    mods = [n for n, _m in model.named_modules() if "lora" in n.lower()]
    return mods[:limit] if len(mods) > limit else mods

def get_adapter_state_dict(model: torch.nn.Module):
    try:
        from peft import get_peft_model_state_dict
        return get_peft_model_state_dict(model)
    except Exception:
        return {}

def one_step_grad_check(trainer, model: torch.nn.Module) -> bool:
    """Tiny fwd/bwd to confirm LoRA grads exist."""
    try:
        model.train()
        batch = next(iter(trainer.get_train_dataloader()))
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        labels = batch["labels"].to(device)
        outputs = model(**inputs)
        loss = nn.functional.cross_entropy(outputs.logits, labels)
        loss.backward()
        flags = [(n, p.grad is not None) for n, p in model.named_parameters() if "lora_" in n.lower()]
        print("🔬 One-step LoRA grad flags (first 10):", flags[:10])
        return any(flag for _, flag in flags)
    except Exception as e:
        print(f"⚠ one_step_grad_check failed: {e}")
        return False

# ----------------------------- main -------------------------------------------
def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    if args.tf32 and torch.cuda.is_available():
        try: torch.backends.cuda.matmul.allow_tf32 = True
        except Exception: pass

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, do_lower_case=False, local_files_only=True)

    from datasets import load_dataset
    print("✔ Loading CSV datasets")
    ds = load_dataset("csv", data_files={"train": args.train, "validation": args.val, "test": args.test})

    def preprocess(batch):
        enc = tok(batch["sequence"], truncation=True, max_length=args.max_len)
        enc["labels"] = batch["label"]
        return enc

    ds = ds.map(preprocess, batched=True, remove_columns=ds["train"].column_names)

    # Class weights (imbalance)
    labels = ds["train"]["labels"]
    pos_count = int(sum(labels)); neg_count = len(labels) - pos_count
    pos_weight = (neg_count / max(pos_count, 1)) if pos_count else 1.0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    class_weights = torch.tensor([1.0, pos_weight], dtype=torch.float, device=device)

    # Base model (with quantization fallback)
    from transformers import EsmForSequenceClassification
    bnb_cfg, bnb_note = try_make_bnb_config(args.quant)

    def load_model_with_fallback():
        if bnb_cfg is not None:
            print(f"ℹ Attempting quantized load ({bnb_note}) …")
            try:
                base = EsmForSequenceClassification.from_pretrained(
                    args.model, num_labels=2, quantization_config=bnb_cfg, local_files_only=True
                )
                try:
                    from peft import prepare_model_for_kbit_training
                    base = prepare_model_for_kbit_training(base)
                except Exception as e:
                    print(f"⚠ Could not prepare k-bit training; continuing anyway: {e}")
                return base
            except Exception as e:
                print(f"⚠ Quantized load failed ({e}). Falling back to full precision.")
        elif args.quant != "none":
            print("⚠ Quantization requested but not usable here. Using full precision.")
        return EsmForSequenceClassification.from_pretrained(
            args.model, num_labels=2, local_files_only=True
        )

    base_model = load_model_with_fallback()

    # Show projection-like module names (you saw "attention.self.query/key/value")
    proj_hits = find_projection_modules_names(base_model)
    print(f"🔎 Projection-like modules found ({len(proj_hits)}):")
    print("   ", proj_hits[:30], "..." if len(proj_hits) > 30 else "")

    # LoRA: match exactly attention.self.query/key/value on ESM
    # (PEFT accepts string regex OR list of suffix names; docs confirm regex behavior.) :contentReference[oaicite:2]{index=2}
    from peft import LoraConfig, TaskType, get_peft_model
    lcfg = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=r".*attention\.self\.(query|key|value)$",
        bias="none",
        task_type=TaskType.SEQ_CLS,
    )
    model = get_peft_model(base_model, lcfg)

    # PEFT + checkpointing: standard fix to avoid "requires_grad=False" warning
    model.config.use_cache = False
    if not args.no_gc:
        try: model.enable_input_require_grads()
        except AttributeError: pass

    # Attachment summary
    print("🧩 LoRA modules (sample):", list_lora_modules(model))
    model.print_trainable_parameters()
    print("🔩 Trainable parameters (sample):", sample_trainable_params(model))

    from transformers import DataCollatorWithPadding
    pad_to_mult = 8 if (torch.cuda.is_available() and (args.fp16 or args.bf16)) else None
    data_collator = DataCollatorWithPadding(tokenizer=tok, pad_to_multiple_of=pad_to_mult)  # tensor cores friendly :contentReference[oaicite:3]{index=3}

    from transformers import Trainer, TrainingArguments

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits  = outputs.logits
            loss_fn = nn.CrossEntropyLoss(weight=class_weights)
            loss = loss_fn(logits, labels)
            return (loss, outputs) if return_outputs else loss

    optim = "paged_adamw_8bit" if bnb_cfg is not None else "adamw_torch_fused"

    # Version-proof TrainingArguments (eval_strategy vs evaluation_strategy)
    def build_targs():
        base = dict(
            output_dir="ft_sh_lora_run",
            per_device_train_batch_size=args.bs,
            #per_device_eval_batch_size=args.bs,
            learning_rate=args.lr,
            num_train_epochs=args.epochs,
            save_strategy="epoch",
            logging_strategy="epoch",
            gradient_checkpointing=(not args.no_gc),
            gradient_checkpointing_kwargs={"use_reentrant": False},  # safer w/ recent PT/TF combos :contentReference[oaicite:4]{index=4}
            bf16=bool(args.bf16 and torch.cuda.is_available()),
            fp16=bool(args.fp16 and torch.cuda.is_available() and not args.bf16),
            tf32=args.tf32 if torch.cuda.is_available() else None,
            dataloader_num_workers=args.workers,
            dataloader_pin_memory=True,
            save_total_limit=2,
            seed=args.seed,
            report_to="none",
            optim=optim,
            label_names=["labels"],  # belongs in TrainingArguments :contentReference[oaicite:5]{index=5}
            group_by_length=True,              # batches similar lengths together
            per_device_eval_batch_size=1,      # evaluation often peaks memory
            eval_accumulation_steps=1,         # stream preds off GPU (avoid huge buffers)
        )
        try:
            return TrainingArguments(eval_strategy="epoch", **base)
        except TypeError:
            return TrainingArguments(evaluation_strategy="epoch", **base)

    targs = build_targs()
    model.to(device)

    trainer = WeightedTrainer(
        model=model,
        args=targs,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        compute_metrics=compute_metrics_local,
        data_collator=data_collator,
    )

    # Optional deep probe before training
    if args.debug:
        adapter_sd = get_adapter_state_dict(model)
        print(f"🧪 Adapter state_dict keys: {len(adapter_sd)}; sample:", list(adapter_sd.keys())[:10])
        ok = one_step_grad_check(trainer, model)
        print("✅ LoRA grads present?" if ok else "❌ No LoRA grads detected!")

    # Train
    trainer.train()

    # Test
    print("🔎 Evaluating on held-out test set …")
    test_metrics = trainer.evaluate(ds["test"])
    print(f"Test metrics: {test_metrics}")

    # Save LoRA + tokenizer
    Path(args.output).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output)
    tok.save_pretrained(args.output)
    print(f"🎉 LoRA adapters saved to  →  {Path(args.output).resolve()}")

if __name__ == "__main__":
    main()
