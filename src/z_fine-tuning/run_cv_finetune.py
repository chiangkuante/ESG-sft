import argparse
import json
import logging
from pathlib import Path
from datasets import load_dataset
import torch
from transformers import StoppingCriteria, StoppingCriteriaList

import re
import math
import random
from collections import defaultdict

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logging.getLogger("transformers").setLevel(logging.ERROR)

LABELS = [
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

LABELS_SORTED = sorted(LABELS, key=len, reverse=True)


def canonicalize_label(text):
    if text is None:
        return None

    text = re.sub(r"\s+", " ", str(text)).strip().strip("\"'")
    if not text:
        return None

    aliases = {
        "pollution and waste": "Pollution & Waste",
        "business ethics and values": "Business Ethics & Values",
        "non esg": "Non-ESG",
    }

    lower = text.lower()
    if lower in aliases:
        return aliases[lower]

    for label in LABELS_SORTED:
        if lower == label.lower():
            return label

    return None


def extract_label(text):
    if text is None:
        return None

    text = str(text).strip()
    if not text:
        return None

    match = re.search(r"<label>\s*(.*?)\s*</label>", text, re.IGNORECASE | re.DOTALL)
    if match:
        return canonicalize_label(match.group(1))

    for label in LABELS_SORTED:
        if text == label or text.startswith(label):
            return label

    patterns = [
        r"(?i)(?:answer|label|category)\s*:\s*(.+)$",
        r"(?i)(?:classified as|category is|label is)\s*:?\s*(.+)$",
        r"(?i)(?:the correct category is)\s*:?\s*(.+)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            candidate = canonicalize_label(match.group(1).strip())
            if candidate is not None:
                return candidate

    tail = text[-200:]
    matched = [label for label in LABELS_SORTED if re.search(rf"(?i)\b{re.escape(label)}\b", tail)]
    if len(matched) == 1:
        return matched[0]

    matched = [label for label in LABELS_SORTED if re.search(rf"(?i)\b{re.escape(label)}\b", text)]
    matched = list(dict.fromkeys(matched))
    if len(matched) == 1:
        return matched[0]

    return None


class StopOnLabel(StoppingCriteria):
    """Stop generation once all sequences in the batch have '</label>'."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.stop_str = "</label>"

    def __call__(self, input_ids, scores, **kwargs):
        # For batched generation: stop only when ALL sequences have </label>
        for seq in input_ids:
            tail = self.tokenizer.decode(seq[-20:], skip_special_tokens=True)
            if self.stop_str not in tail:
                return False
        return True


def setup_base_model(model_type):
    """Load base model + tokenizer only (no LoRA). Used for inference with saved adapters."""
    if model_type == "gemma":
        from unsloth import FastModel
        from unsloth.chat_templates import get_chat_template
        model, tokenizer = FastModel.from_pretrained(
            model_name="unsloth/gemma-3-4b-it",
            max_seq_length=2048,
            load_in_4bit=True,
            load_in_8bit=False,
            full_finetuning=False,
        )
        tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")
        return model, tokenizer, "FastModel"

    elif model_type == "llama":
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import get_chat_template
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/Llama-3.2-3B-Instruct",
            max_seq_length=4096,
            dtype=None,
            load_in_4bit=True,
        )
        tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")
        return model, tokenizer, "FastLanguageModel"

    elif model_type == "qwen":
        from unsloth import FastVisionModel
        model, tokenizer = FastVisionModel.from_pretrained(
            "unsloth/Qwen3.5-4B",
            load_in_4bit=True,
            use_gradient_checkpointing="unsloth",
        )
        return model, tokenizer, "FastVisionModel"

    elif model_type == "ministral":
        from unsloth import FastVisionModel
        model, tokenizer = FastVisionModel.from_pretrained(
            "unsloth/Ministral-3-3B-Instruct-2512",
            load_in_4bit=True,
            use_gradient_checkpointing="unsloth",
        )
        return model, tokenizer, "FastVisionModel"

    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def setup_model(model_type):
    if model_type == "gemma":
        from unsloth import FastModel
        from unsloth.chat_templates import get_chat_template
        
        model, tokenizer = FastModel.from_pretrained(
            model_name="unsloth/gemma-3-4b-it",
            max_seq_length=2048,
            load_in_4bit=True,
            load_in_8bit=False,
            full_finetuning=False,
        )
        
        model = FastModel.get_peft_model(
            model,
            finetune_vision_layers=False,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            r=16,
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            random_state=3407,
        )
        
        tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")
        
        def formatting_func(examples):
            convos = examples["conversations"]
            texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False).removeprefix('<bos>') for convo in convos]
            return {"text": texts}
            
        kwargs = {
            "instruction_part": "<start_of_turn>user\n",
            "response_part": "<start_of_turn>model\n"
        }
        
        return model, tokenizer, formatting_func, kwargs, "FastModel"

    elif model_type == "llama":
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import get_chat_template
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/Llama-3.2-3B-Instruct",
            max_seq_length=4096,
            dtype=None,
            load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )
        tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")
        
        def formatting_func(examples):
            convos = examples["conversations"]
            texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
            return {"text": texts}
            
        kwargs = {
            "instruction_part": "<|start_header_id|>user<|end_header_id|>\n\n",
            "response_part": "<|start_header_id|>assistant<|end_header_id|>\n\n"
        }
        return model, tokenizer, formatting_func, kwargs, "FastLanguageModel"

    elif model_type == "qwen":
        from unsloth import FastVisionModel
        model, tokenizer = FastVisionModel.from_pretrained(
            "unsloth/Qwen3.5-4B",
            load_in_4bit=True,
            use_gradient_checkpointing="unsloth",
        )
        model = FastVisionModel.get_peft_model(
            model,
            finetune_vision_layers=False,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            r=16,
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )
        
        def convert_to_conversation(sample):
            conversation = []
            for message in sample["conversations"]:
                conversation.append({
                    "role": message["role"],
                    "content": [{"type": "text", "text": message["content"]}]
                })
            return {"messages": conversation}
            
        return model, tokenizer, convert_to_conversation, None, "FastVisionModel"

    elif model_type == "ministral":
        from unsloth import FastVisionModel
        model, tokenizer = FastVisionModel.from_pretrained(
            "unsloth/Ministral-3-3B-Instruct-2512",
            load_in_4bit=True,
            use_gradient_checkpointing="unsloth",
        )
        model = FastVisionModel.get_peft_model(
            model,
            finetune_vision_layers=False,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            r=16,
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )
        
        def convert_to_conversation(sample):
            conversation = []
            for message in sample["conversations"]:
                conversation.append({
                    "role": message["role"],
                    "content": [{"type": "text", "text": message["content"]}]
                })
            return {"messages": conversation}
            
        return model, tokenizer, convert_to_conversation, None, "FastVisionModel"
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def _build_messages(item, model_type):
    """Build message history from a dataset item, excluding the last (assistant) turn."""
    msg_history = []
    if model_type == "llama":
        role_map = {"human": "user", "gpt": "assistant", "system": "system"}
        for m in item["conversations"][:-1]:
            r = role_map.get(m.get("from", m.get("role")), m.get("role"))
            msg_history.append({"role": r, "content": m.get("value", m.get("content"))})
    elif model_type in ["gemma", "qwen", "ministral"]:
        for m in item["conversations"][:-1]:
            msg_history.append({
                "role": m["role"],
                "content": [{"type": "text", "text": m["content"]}]
            })
    return msg_history


def _run_inference(model, tokenizer, test_ds, test_indices, model_type, batch_size, fold_idx):
    """Run inference sample-by-sample with early stopping.

    Note: Unsloth's compiled forward for Gemma3/Qwen/Ministral (VLM architecture)
    does not support batch_size > 1 in model.generate(). We use single-sample
    inference with StopOnLabel for early termination instead.
    """
    import tqdm

    stop_criteria = StoppingCriteriaList([StopOnLabel(tokenizer)])
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    results = []
    for i, item in enumerate(tqdm.tqdm(test_ds, desc=f"Inferencing Fold {fold_idx}")):
        msgs = _build_messages(item, model_type)

        # Tokenize: use apply_chat_template with return_dict for dict-based generate
        if model_type == "llama":
            tokenized = tokenizer.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt",
            )
            if isinstance(tokenized, torch.Tensor):
                input_ids = tokenized.to("cuda")
                attention_mask = torch.ones_like(input_ids)
            else:
                input_ids = tokenized["input_ids"].to("cuda")
                attention_mask = tokenized.get("attention_mask", torch.ones_like(input_ids)).to("cuda")
            inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        else:
            # Gemma / Qwen / Ministral: return_dict=True
            inputs = tokenizer.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=True,
                return_tensors="pt", return_dict=True,
            )
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=None,
                top_p=None,
                use_cache=True,
                pad_token_id=pad_id,
                stopping_criteria=stop_criteria,
            )

        input_len = inputs["input_ids"].shape[1]
        decoded = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()

        gt_raw = item["conversations"][-1].get("content", item["conversations"][-1].get("value"))
        results.append({
            "test_index": test_indices[i],
            "raw_output": decoded,
            "parsed_label": extract_label(decoded),
            "ground_truth_label": extract_label(gt_raw),
        })

    return results


def oversample_dataset(train_ds):
    """
    Oversample minority classes using square root of the ratio to the majority class.
    """
    label_to_indices = defaultdict(list)
    for i, item in enumerate(train_ds):
        gt_msg = item["conversations"][-1]
        gt_raw = gt_msg.get("content", gt_msg.get("value", ""))
        label = extract_label(gt_raw)
        if label is None:
            label = "Unknown"
        label_to_indices[label].append(i)

    if not label_to_indices:
        return train_ds

    max_count = max(len(indices) for indices in label_to_indices.values())
    
    new_indices = []
    rng = random.Random(3407)

    for label, indices in label_to_indices.items():
        count = len(indices)
        if count == 0:
            continue
        
        multiplier = math.sqrt(max_count / count)
        full_copies = int(multiplier)
        fraction = multiplier - full_copies

        # Full copies
        for _ in range(full_copies):
            new_indices.extend(indices)
            
        # Fractional copy: random sample without replacement
        num_fractional = int(count * fraction)
        if num_fractional > 0:
            new_indices.extend(rng.sample(indices, num_fractional))

    rng.shuffle(new_indices)
    return train_ds.select(new_indices)


def train_all_folds(model_type, dataset_sft, cv_folds):
    """Phase 1: Train all folds sequentially and save LoRA adapters."""
    adapter_dirs = []
    for fold_idx, fold_data in enumerate(cv_folds):
        logger.info(f"========== Training Fold {fold_idx} for {model_type} ==========")
        train_indices = fold_data["train_indices"]
        train_ds = dataset_sft.select(train_indices)

        logger.info(f"Original train size: {len(train_ds)}")
        train_ds = oversample_dataset(train_ds)
        logger.info(f"Oversampled train size: {len(train_ds)}")

        model, tokenizer, formatting_func, kwargs, api_class = setup_model(model_type)

        if api_class == "FastLanguageModel" and model_type == "llama":
            from unsloth.chat_templates import standardize_sharegpt
            train_ds = standardize_sharegpt(train_ds)

        if api_class in ["FastModel", "FastLanguageModel"]:
            train_ds = train_ds.map(formatting_func, batched=True)
            from trl import SFTTrainer, SFTConfig
            from transformers import DataCollatorForSeq2Seq

            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=train_ds,
                dataset_text_field="text",
                max_seq_length=2048 if model_type == "gemma" else 4096,
                data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer) if model_type == "llama" else None,
                packing=False,
                args=SFTConfig(
                    per_device_train_batch_size=2,
                    gradient_accumulation_steps=4,
                    warmup_steps=5,
                    num_train_epochs=1,
                    learning_rate=2e-4,
                    logging_steps=1,
                    optim="adamw_8bit",
                    weight_decay=0.001,
                    lr_scheduler_type="linear",
                    seed=3407,
                    output_dir=f"models/cv/{model_type}/fold_{fold_idx}/checkpoints",
                    report_to="none",
                ),
            )
            from unsloth.chat_templates import train_on_responses_only
            trainer = train_on_responses_only(
                trainer,
                instruction_part=kwargs["instruction_part"],
                response_part=kwargs["response_part"],
            )
        else:
            # FastVisionModel
            from unsloth import FastVisionModel
            FastVisionModel.for_training(model)
            from unsloth.trainer import UnslothVisionDataCollator
            from trl import SFTTrainer, SFTConfig

            converted_dataset = [formatting_func(sample) for sample in train_ds]
            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                data_collator=UnslothVisionDataCollator(model, tokenizer),
                train_dataset=converted_dataset,
                args=SFTConfig(
                    per_device_train_batch_size=2,
                    gradient_accumulation_steps=4,
                    warmup_steps=5,
                    num_train_epochs=1,
                    learning_rate=2e-4,
                    logging_steps=1,
                    optim="adamw_8bit",
                    weight_decay=0.001,
                    lr_scheduler_type="linear",
                    seed=3407,
                    output_dir=f"models/cv/{model_type}/fold_{fold_idx}/checkpoints",
                    report_to="none",
                    remove_unused_columns=False,
                    dataset_text_field="",
                    dataset_kwargs={"skip_prepare_dataset": True},
                    max_length=2048,
                ),
            )

        logger.info(f"Training fold {fold_idx}...")
        trainer.train()

        # Save adapter
        adapter_dir = Path(f"models/cv/{model_type}/fold_{fold_idx}") / "adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
        adapter_dirs.append(str(adapter_dir))
        logger.info(f"Saved adapter for fold {fold_idx} to {adapter_dir}")

        # Release GPU memory
        del model, tokenizer, trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc; gc.collect()
        logger.info(f"Released GPU memory after fold {fold_idx} training")

    return adapter_dirs


def _infer_single_fold(model_type, adapter_dir, test_indices, dataset_sft,
                       output_dir, batch_size, fold_idx):
    """Load base model + adapter, run inference, save results. Designed for subprocess."""
    logger.info(f"[Fold {fold_idx}] Loading base model + adapter from {adapter_dir}")

    # Load base model WITHOUT LoRA (avoid double adapter)
    model, tokenizer, api_class = setup_base_model(model_type)

    # Load saved LoRA adapter weights onto base model
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, adapter_dir)

    # Switch to inference mode
    if api_class == "FastLanguageModel":
        from unsloth import FastLanguageModel
        FastLanguageModel.for_inference(model)
    elif api_class == "FastModel":
        from unsloth import FastModel
        FastModel.for_inference(model)
    else:
        from unsloth import FastVisionModel
        FastVisionModel.for_inference(model)

    test_ds = dataset_sft.select(test_indices)

    logger.info(f"[Fold {fold_idx}] Starting inference ({len(test_ds)} samples)...")
    results = _run_inference(
        model, tokenizer, test_ds, test_indices, model_type, batch_size, fold_idx
    )

    out_file = Path(output_dir) / f"fold_{fold_idx}_{model_type}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"[Fold {fold_idx}] Saved {len(results)} results to {out_file}")


def _inference_worker(model_type, adapter_dir, test_indices, train_sft_path,
                      output_dir, batch_size, fold_idx):
    """Subprocess entry point: reload dataset and run inference for one fold."""
    # Each subprocess reloads data independently to avoid shared-memory issues
    dataset_sft = load_dataset("json", data_files=str(train_sft_path), split="train")
    _infer_single_fold(
        model_type, adapter_dir, test_indices, dataset_sft,
        output_dir, batch_size, fold_idx,
    )


def run_inference_parallel(model_type, cv_folds, adapter_dirs, train_sft_path,
                           output_dir, batch_size):
    """Phase 2: Run inference in parallel subprocess batches.

    Batch layout (tuned for 24GB 4090, ~6.3GB per model in 4bit):
      - Batch 1: fold 0, 1  (2 x ~6.3 ≈ 12.6 GB)
      - Batch 2: fold 2     (1 x ~6.3 ≈  6.3 GB)
    """
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    if model_type == "qwen":
        inference_batches = [[0], [1], [2]]
    else:
        inference_batches = [[0, 1], [2]]

    for batch_idx, fold_ids in enumerate(inference_batches):
        # Filter out folds that exceed the actual number available
        fold_ids = [f for f in fold_ids if f < len(cv_folds)]
        if not fold_ids:
            continue

        logger.info(f"===== Inference Batch {batch_idx}: folds {fold_ids} =====")
        processes = []
        for fold_idx in fold_ids:
            test_indices = cv_folds[fold_idx]["test_indices"]
            p = mp.Process(
                target=_inference_worker,
                args=(
                    model_type,
                    adapter_dirs[fold_idx],
                    test_indices,
                    str(train_sft_path),
                    str(output_dir),
                    batch_size,
                    fold_idx,
                ),
            )
            p.start()
            processes.append((fold_idx, p))

        # Wait for all processes in this batch to finish
        for fold_idx, p in processes:
            p.join()
            if p.exitcode != 0:
                logger.error(f"[Fold {fold_idx}] Inference process exited with code {p.exitcode}")
            else:
                logger.info(f"[Fold {fold_idx}] Inference process completed successfully")

        logger.info(f"===== Inference Batch {batch_idx} done =====")


def main():
    parser = argparse.ArgumentParser(description="Run 3-Fold cross validation fine-tuning with Unsloth")
    parser.add_argument("--model", type=str, required=True, choices=["gemma", "llama", "qwen", "ministral"],
                        help="The model type to finetune and evaluate.")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for inference (default: 4). Increase if GPU memory allows.")
    args = parser.parse_args()

    cv_folds_path = Path("data/processed/step7_cv/cv_folds.json")
    train_sft_path = Path("data/processed/step6_sft/train_sft_xml.json")

    with open(cv_folds_path, "r", encoding="utf-8") as f:
        cv_folds = json.load(f)

    dataset_sft = load_dataset("json", data_files=str(train_sft_path), split="train")

    output_dir = Path("results/cv") / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: Train all folds sequentially, save adapters
    logger.info("===== Phase 1: Sequential Training =====")
    adapter_dirs = train_all_folds(args.model, dataset_sft, cv_folds)

    # Phase 2: Parallel inference in batches
    logger.info("===== Phase 2: Parallel Inference =====")
    run_inference_parallel(
        args.model, cv_folds, adapter_dirs, train_sft_path,
        output_dir, args.batch_size,
    )

    logger.info("All folds done.")


if __name__ == "__main__":
    main()

