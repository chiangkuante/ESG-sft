from __future__ import annotations

import logging
import random
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.step4_cv.common import load_step4_config, resolve_experiment_dir, resolve_experiment_name
from src.step5_reasoning.common import load_json, load_step5_config, resolve_path, save_json
from src.step5_reasoning.reason import (
    CATEGORY_DEFINITIONS,
    CLASSIFY_SYSTEM_PROMPT,
    CLASSIFY_USER_TEMPLATE,
    build_structured_answer,
)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")


def build_sft_record(item: dict) -> dict:
    return {
        "paragraph_id": item["paragraph_id"],
        "label": item["label"],
        "source": item["source"],
        "conversations": [
            {"role": "system", "content": CLASSIFY_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": CLASSIFY_USER_TEMPLATE.format(
                    definitions=CATEGORY_DEFINITIONS,
                    text=item["combined_text"],
                ),
            },
            {
                "role": "assistant",
                "content": build_structured_answer(item["reasoning"], item["label"]),
            },
        ],
    }


def main() -> None:
    config = load_step5_config()
    paths_cfg = config["paths"]
    sft_cfg = config["sft"]

    experiment_name = resolve_experiment_name(load_step4_config())
    step4_output_dir = resolve_experiment_dir(resolve_path(paths_cfg["step4_output_dir"]), experiment_name)
    reasoning_output_dir = resolve_experiment_dir(resolve_path(paths_cfg["output_dir"]), experiment_name)
    sft_output_dir = resolve_experiment_dir(resolve_path(paths_cfg["sft_output_dir"]), experiment_name)

    manifest = load_json(step4_output_dir / "manifest.json")
    shuffle_seed = int(sft_cfg["shuffle_seed"])

    build_manifest = {"folds": []}
    for fold_name in manifest["fold_directories"]:
        fold_idx = int(fold_name.split("_")[-1])
        fold_reasoning_dir = reasoning_output_dir / fold_name
        fold_sft_dir = sft_output_dir / fold_name
        train_with_reasoning = load_json(fold_reasoning_dir / "train_with_reasoning.json")
        valid_items = [item for item in train_with_reasoning if not item.get("failed", False) and item.get("reasoning")]
        invalid_items = [item for item in train_with_reasoning if item not in valid_items]

        sft_records = [build_sft_record(item) for item in valid_items]
        rng = random.Random(shuffle_seed + fold_idx)
        rng.shuffle(sft_records)

        save_json(fold_sft_dir / "train_sft_text.json", sft_records)
        save_json(fold_sft_dir / "excluded_items.json", invalid_items)

        human_val = load_json(step4_output_dir / fold_name / "human_val.json")
        save_json(fold_sft_dir / "val_eval.json", human_val)

        build_manifest["folds"].append(
            {
                "fold": fold_idx,
                "sft_size": len(sft_records),
                "excluded_size": len(invalid_items),
                "sft_path": str((fold_sft_dir / "train_sft_text.json").relative_to(Path.cwd())),
                "val_eval_path": str((fold_sft_dir / "val_eval.json").relative_to(Path.cwd())),
            }
        )
        logger.info("Fold %s SFT ready: kept=%s excluded=%s", fold_idx, len(sft_records), len(invalid_items))

    save_json(sft_output_dir / "manifest.json", build_manifest)


if __name__ == "__main__":
    main()
