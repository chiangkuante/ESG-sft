"""Quick dry-run evaluation: train fold_0 (3 steps) then inference on fold_0 test set."""

import json
import logging
import re
import sys
import warnings
from collections import Counter
from pathlib import Path

import torch

# Must import unsloth first
from unsloth import FastLanguageModel, FastModel, FastVisionModel

# Re-use the training script's logic
sys.path.insert(0, str(Path(__file__).parent))
from run_cv_finetune import MODELS_CONFIG, train_fold, MAX_SEQ_LENGTH


# -----------------------------
# Suppress noisy HF / transformers warnings
# -----------------------------
warnings.filterwarnings(
  "ignore",
  message=r"The attention mask API under `transformers\.modeling_attn_mask_utils`.*",
)
warnings.filterwarnings("ignore", category=FutureWarning)

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_attn_mask_utils").setLevel(logging.ERROR)

try:
  from transformers.utils import logging as hf_logging

  hf_logging.set_verbosity_error()
except Exception:
  pass


INPUT_SFT = Path("data/processed/step6_sft/train_sft.json")
INPUT_FOLDS = Path("data/processed/step7_cv/cv_folds.json")
MODELS_DIR = Path("models/cv")

VALID_LABELS = [
  "Climate Change",
  "Natural Capital",
  "Pollution & Waste",
  "Human Capital",
  "Product Liability",
  "Community Relations",
  "Corporate Governance",
  "Business Ethics & Values",
  "Non-ESG",
]

LABEL_ALIASES = {
  "climate change": "Climate Change",
  "natural capital": "Natural Capital",
  "pollution & waste": "Pollution & Waste",
  "pollution and waste": "Pollution & Waste",
  "human capital": "Human Capital",
  "product liability": "Product Liability",
  "community relations": "Community Relations",
  "corporate governance": "Corporate Governance",
  "business ethics & values": "Business Ethics & Values",
  "business ethics and values": "Business Ethics & Values",
  "business ethics": "Business Ethics & Values",
  "non-esg": "Non-ESG",
  "non esg": "Non-ESG",
}

LABEL_PATTERN = re.compile(
  r"(Climate Change|Natural Capital|Pollution\s*(?:&|and)\s*Waste|"
  r"Human Capital|Product Liability|Community Relations|"
  r"Corporate Governance|Business Ethics\s*(?:&|and)\s*Values|"
  r"Non-ESG|Non ESG)",
  flags=re.IGNORECASE,
)


def normalize_spaces(text: str) -> str:
  return re.sub(r"\s+", " ", text).strip()


def canonicalize_label(text: str) -> str:
  if not text:
    return "Unclassified"

  text = normalize_spaces(text)
  text = text.strip().strip("*").strip("`").strip('"').strip("'")
  text = text.rstrip(".:;!，。；：")
  lower = text.lower()

  if lower in LABEL_ALIASES:
    return LABEL_ALIASES[lower]

  for label in VALID_LABELS:
    if lower == label.lower():
      return label

  return "Unclassified"


def extract_label(text: str) -> str:
  """
  Robust label extraction for outputs like:
  - Corporate Governance
  - **Corporate Governance**
  - Corporate Governance.
  - Based on the paragraph, the label is "Corporate Governance"
  - **Climate Change**\n\nReasoning: ...
  - The most relevant category is:\n\n**Non-ESG**
  """
  if not text:
    return "Unclassified"

  raw = text.strip()
  cleaned = raw.replace("`", "").replace("*", "").strip()

  # 1) Exact whole-string match
  exact = canonicalize_label(cleaned)
  if exact != "Unclassified":
    return exact

  # 2) First non-empty line
  lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
  if lines:
    first_line = canonicalize_label(lines[0])
    if first_line != "Unclassified":
      return first_line

  # 3) Common explicit phrasing
  explicit_patterns = [
  r"classified as:\s*([^\n]+)",
  r"label\s*:\s*([^\n]+)",
  r"category\s*:\s*([^\n]+)",
  r"most relevant category is\s*:\s*([^\n]+)",
  r"most relevant ESG category is\s*:\s*([^\n]+)",
  r"the answer is\s*([^\n]+)",
  r"i would classify(?: this| the paragraph| it)?(?: into| as)?\s*(?:the category)?\s*[\"']?([^\n\"']+)",
  r"this paragraph (?:should be|is) classified as\s*[\"']?([^\n\"']+)",
]
  multiline_patterns = [
  r"most relevant ESG category is\s*:\s*\n+\*{0,2}([A-Za-z&\-\s]+)\*{0,2}",
  r"most relevant category is\s*:\s*\n+\*{0,2}([A-Za-z&\-\s]+)\*{0,2}",
  r"here is the classification\s*:\s*\n+\*{0,2}([A-Za-z&\-\s]+)\*{0,2}",
  r"classification of the paragraph\s*:\s*\n+\*{0,2}([A-Za-z&\-\s]+)\*{0,2}",
]

  for pat in multiline_patterns:
    m = re.search(pat, raw, flags=re.IGNORECASE)
    if m:
      candidate = canonicalize_label(m.group(1))
      if candidate != "Unclassified":
        return candidate

  # 4) Search top few lines first
  top_chunk = "\n".join(lines[:6]) if lines else cleaned
  m = LABEL_PATTERN.search(top_chunk)
  if m:
    return canonicalize_label(m.group(1))

  # 5) Search full text fallback
  m = LABEL_PATTERN.search(cleaned)
  if m:
    return canonicalize_label(m.group(1))

  return "Unclassified"


def normalize_label(text: str) -> str:
  return canonicalize_label(text)


def get_ground_truth(sample: dict) -> str:
  for turn in sample["conversations"]:
    role = turn.get("role") or turn.get("from")
    if role in ["assistant", "gpt"]:
      content = turn.get("content") or turn.get("value") or ""
      return normalize_label(extract_label(content))
  return "Unknown"


def build_sft_messages(sample: dict, use_list_content: bool = False) -> list[dict]:
  """Build messages for inference.

  use_list_content: True for vision models AND Gemma 3 (multimodal chat template).
  """
  messages = []
  role_map = {"human": "user", "gpt": "assistant"}

  for turn in sample["conversations"]:
    role = turn.get("role") or turn.get("from")
    content = turn.get("content") or turn.get("value") or ""
    role = role_map.get(role, role)

    if role in ["user", "system"]:
      if use_list_content:
        messages.append(
          {"role": role, "content": [{"type": "text", "text": content}]}
        )
      else:
        messages.append({"role": role, "content": content})

  return messages


def decode_new_tokens(tokenizer, outputs, prompt_len: int) -> str:
  generated_ids = outputs[0][prompt_len:]
  return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def evaluate_fold0(model_key: str, sft_data: list, fold: dict, max_samples: int = 20):
  """Load saved adapter, run inference on fold_0 test set."""
  cfg = MODELS_CONFIG[model_key]
  is_vision = cfg.get("is_vision", False)
  loader = cfg["loader"]
  model_path = MODELS_DIR / model_key / "fold_0"

  print(f"\n--- Loading {model_key} adapter from {model_path} ---")

  if is_vision:
    model, tokenizer = loader.from_pretrained(
      str(model_path),
      load_in_4bit=False,
      use_gradient_checkpointing="unsloth",
    )
  elif loader is FastModel:
    model, tokenizer = loader.from_pretrained(
      model_name=str(model_path),
      max_seq_length=MAX_SEQ_LENGTH,
      load_in_4bit=True,
      load_in_8bit=False,
      full_finetuning=False,
    )
  else:
    model, tokenizer = loader.from_pretrained(
      model_name=str(model_path),
      max_seq_length=MAX_SEQ_LENGTH,
      dtype=None,
      load_in_4bit=True,
    )

  if not is_vision:
    from unsloth.chat_templates import get_chat_template

    tokenizer = get_chat_template(tokenizer, chat_template=cfg["chat_template"])

  if is_vision:
    FastVisionModel.for_inference(model)
  elif loader is FastModel:
    FastModel.for_inference(model)
  else:
    FastLanguageModel.for_inference(model)

  test_indices = fold["test_indices"][:max_samples]
  test_samples = [sft_data[i] for i in test_indices]

  correct = 0
  total = 0
  errors = 0
  class_correct = Counter()
  class_total = Counter()

  # Gemma 3 uses multimodal chat template (list content) even for text-only
  use_list_content = is_vision or (loader is FastModel)

  for idx, sample in enumerate(test_samples):
    gt = get_ground_truth(sample)
    messages = build_sft_messages(sample, use_list_content=use_list_content)

    try:
      if is_vision:
        input_text = tokenizer.apply_chat_template(
          messages,
          add_generation_prompt=True,
        )
        inputs = tokenizer(
          None,
          input_text,
          add_special_tokens=False,
          return_tensors="pt",
        )
        inputs = {
          k: v.to("cuda") if hasattr(v, "to") else v
          for k, v in inputs.items()
        }
        prompt_len = inputs["input_ids"].shape[1]
      else:
        tokenized = tokenizer.apply_chat_template(
          messages,
          tokenize=True,
          add_generation_prompt=True,
          return_tensors="pt",
        )

        if isinstance(tokenized, torch.Tensor):
          input_ids = tokenized.to("cuda")
          attention_mask = torch.ones_like(input_ids)
        else:
          input_ids = tokenized["input_ids"].to("cuda")
          attention_mask = tokenized.get(
            "attention_mask", torch.ones_like(input_ids)
          ).to("cuda")

        inputs = {
          "input_ids": input_ids,
          "attention_mask": attention_mask,
        }
        prompt_len = input_ids.shape[1]

      with torch.no_grad():
        outputs = model.generate(
          **inputs,
          max_new_tokens=80,
          do_sample=False,
          temperature=None,
          top_p=None,
          use_cache=True,
          pad_token_id=tokenizer.eos_token_id,
        )

      raw = decode_new_tokens(tokenizer, outputs, prompt_len)
      pred = normalize_label(extract_label(raw))

    except Exception as e:
      pred = "Unclassified"
      raw = ""
      errors += 1
      print(f" [Error] {type(e).__name__}: {e}")

    is_correct = pred == gt
    correct += int(is_correct)
    total += 1
    class_total[gt] += 1
    if is_correct:
      class_correct[gt] += 1

    mark = "O" if is_correct else "X"
    print(f" [{mark}] #{idx} gt={gt} pred={pred} raw={raw[:120]}")

  acc = correct / total if total > 0 else 0.0

  print(f"\n {model_key} fold_0 (3-step dry run):")
  print(f"  Accuracy: {correct}/{total} = {acc:.1%}")
  print(f"  Errors: {errors}")
  for label in sorted(class_total.keys()):
    c = class_correct[label]
    t = class_total[label]
    print(f"  {label}: {c}/{t}")

  del model, tokenizer
  torch.cuda.empty_cache()

  return {
    "model": model_key,
    "accuracy": acc,
    "correct": correct,
    "total": total,
    "errors": errors,
  }


def main():
  with open(INPUT_SFT, "r", encoding="utf-8") as f:
    sft_data = json.load(f)
  with open(INPUT_FOLDS, "r", encoding="utf-8") as f:
    folds = json.load(f)

  fold0 = folds[0]
  model_keys = ["gemma", "llama", "qwen", "ministral"]
  results = []

  for key in model_keys:
    model_id = MODELS_CONFIG[key]["model_id"]
    model_path = MODELS_DIR / key / "fold_0"

    # Train if not exists
    if not (model_path / "adapter_config.json").exists():
      print(f"\n{'=' * 50}")
      print(f" Training {key} fold_0 (3 steps)...")
      print(f"{'=' * 50}")
      train_fold(
        fold_id=0,
        train_indices=fold0["train_indices"],
        sft_data=sft_data,
        model_name=model_id,
        model_key=key,
        dry_run=3,
      )

    # Evaluate
    r = evaluate_fold0(key, sft_data, fold0, max_samples=20)
    results.append(r)

  print(f"\n{'=' * 60}")
  print(" DRY RUN ACCURACY SUMMARY (3 steps, fold_0, 20 samples)")
  print(f"{'=' * 60}")
  print(f" {'Model':<12} {'Accuracy':>10} {'Correct':>10} {'Errors':>8}")
  print(f" {'-' * 42}")
  for r in results:
    print(
      f" {r['model']:<12} {r['accuracy']:>9.1%} "
      f"{r['correct']:>5}/{r['total']:<4} {r['errors']:>8}"
    )


if __name__ == "__main__":
  main()