"""
Stage 3b: Generate annotation guidelines markdown.

Reads classified paragraphs and produces a guidelines document with
category definitions, high-confidence example paragraphs, and
disambiguation notes.

Usage:
    python -m src.pre_label.annotation_guidelines [--top-k 3]
"""

import argparse
import json
import os
from collections import defaultdict

from src.logger import setup_logger

logger = setup_logger(__name__)

# Category definitions based on standard ESG taxonomy
CATEGORY_DEFINITIONS = {
    "Climate Change": (
        "Risks and opportunities related to climate change, including greenhouse gas emissions, "
        "carbon footprint, climate-related regulations, physical climate risks (extreme weather, "
        "sea-level rise), transition risks, and energy efficiency initiatives."
    ),
    "Natural Capital": (
        "Risks related to the use and management of natural resources, including biodiversity, "
        "water usage and scarcity, land use, deforestation, and ecosystem impacts."
    ),
    "Pollution & Waste": (
        "Risks related to pollution, toxic emissions, hazardous waste management, "
        "environmental remediation, packaging waste, and circular economy practices."
    ),
    "Human Capital": (
        "Risks related to workforce management, including employee health and safety, "
        "labor relations, talent attraction and retention, diversity and inclusion, "
        "training and development, and compensation practices."
    ),
    "Product Liability": (
        "Risks related to product safety, quality control, customer welfare, "
        "data privacy and security, responsible marketing, and product recalls."
    ),
    "Community Relations": (
        "Risks related to community impact, social license to operate, human rights, "
        "supply chain labor standards, community engagement, and philanthropy."
    ),
    "Corporate Governance": (
        "Risks related to board structure and independence, executive compensation, "
        "shareholder rights, audit practices, regulatory compliance, "
        "and corporate transparency."
    ),
    "Business Ethics & Values": (
        "Risks related to business ethics, anti-corruption, anti-competitive behavior, "
        "lobbying and political contributions, whistleblower protections, "
        "and tax transparency."
    ),
    "Non-ESG": (
        "General business risks not directly related to ESG factors, including "
        "market competition, macroeconomic conditions, foreign exchange, "
        "interest rate risks, litigation, and operational disruptions."
    ),
}

# Common confusion pairs and disambiguation guidance
CONFUSION_PAIRS = [
    {
        "categories": ("Human Capital", "Community Relations"),
        "guidance": (
            "Human Capital focuses on the company's OWN workforce (employees). "
            "Community Relations addresses impact on EXTERNAL stakeholders "
            "(local communities, supply chain workers, society at large). "
            "If the paragraph discusses employee working conditions -> Human Capital. "
            "If it discusses supply chain labor or community impact -> Community Relations."
        ),
    },
    {
        "categories": ("Climate Change", "Pollution & Waste"),
        "guidance": (
            "Climate Change is specifically about greenhouse gas / carbon emissions "
            "and climate-related physical or transition risks. "
            "Pollution & Waste covers OTHER pollutants (toxic chemicals, waste disposal, "
            "water pollution) that are not primarily GHG-related. "
            "If the risk is about CO2/GHG regulations -> Climate Change. "
            "If it's about hazardous waste or non-GHG pollutants -> Pollution & Waste."
        ),
    },
    {
        "categories": ("Corporate Governance", "Business Ethics & Values"),
        "guidance": (
            "Corporate Governance focuses on governance STRUCTURE (board, audit, "
            "shareholder rights, regulatory compliance frameworks). "
            "Business Ethics & Values concerns ethical BEHAVIOR (bribery, corruption, "
            "anti-competitive practices, political spending). "
            "If about board independence -> Corporate Governance. "
            "If about anti-bribery compliance -> Business Ethics & Values."
        ),
    },
    {
        "categories": ("Product Liability", "Non-ESG"),
        "guidance": (
            "Product Liability includes data privacy/security and product safety, "
            "which can overlap with general business risks. "
            "If the primary concern is about CUSTOMER welfare or data protection "
            "-> Product Liability. If it's about generic IT infrastructure risks "
            "without clear stakeholder impact -> Non-ESG."
        ),
    },
]


def pick_examples(records: list[dict], top_k: int = 3) -> dict[str, list[dict]]:
    """
    Pick top-k highest-confidence examples for each category.
    """
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_cat[r["predicted_category"]].append(r)

    examples = {}
    for cat, items in by_cat.items():
        sorted_items = sorted(items, key=lambda x: -x["confidence"])
        # Pick top_k, prefer shorter texts for readability
        candidates = sorted_items[:top_k * 3]  # pick from a wider pool
        candidates.sort(key=lambda x: len(x["text"]))  # prefer shorter
        examples[cat] = candidates[:top_k]

    return examples


def generate_guidelines(
    input_path: str = "data/10k_1A/classified.json",
    output_dir: str = "data/10k_1A",
    top_k: int = 3,
):
    """Generate annotation guidelines markdown file."""
    logger.info("Generating annotation guidelines...")

    with open(input_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    examples = pick_examples(records, top_k)

    lines = []
    lines.append("# ESG 段落分類標註指引\n")
    lines.append("本文件提供 9 個 ESG 類別的定義、範例段落、以及容易混淆的類別區分說明。\n")
    lines.append("標註者的任務是**確認或修正** FinBERT 的預標註結果。\n")
    lines.append("---\n")

    # Category definitions + examples
    lines.append("## 類別定義與範例\n")
    for cat in CATEGORY_DEFINITIONS:
        lines.append(f"### {cat}\n")
        lines.append(f"**定義**: {CATEGORY_DEFINITIONS[cat]}\n")

        cat_examples = examples.get(cat, [])
        if cat_examples:
            lines.append("**範例段落** (FinBERT 高信心預測):\n")
            for i, ex in enumerate(cat_examples, 1):
                # Truncate for readability
                text_preview = ex["text"][:300]
                if len(ex["text"]) > 300:
                    text_preview += "..."
                lines.append(
                    f"{i}. (Confidence: {ex['confidence']:.2f}, "
                    f"ID: {ex['paragraph_id']})\n"
                    f"   > {text_preview}\n"
                )
        lines.append("")

    # Confusion pairs
    lines.append("---\n")
    lines.append("## 容易混淆的類別區分\n")
    for pair in CONFUSION_PAIRS:
        cat_a, cat_b = pair["categories"]
        lines.append(f"### {cat_a} vs {cat_b}\n")
        lines.append(f"{pair['guidance']}\n")
        lines.append("")

    # General notes
    lines.append("---\n")
    lines.append("## 標註注意事項\n")
    lines.append("1. 每個段落只選擇**一個**最適合的類別\n")
    lines.append("2. 優先檢查**低信心分數**的段落（已按信心分數排序）\n")
    lines.append("3. 若段落涵蓋多個 ESG 議題，選擇**最主要**的那個\n")
    lines.append("4. 若不確定是否為 ESG 相關，預設標為 Non-ESG\n")
    lines.append("5. 被截斷的段落 (is_truncated=True) 可能資訊不完整，請特別留意\n")

    output_path = os.path.join(output_dir, "annotation_guidelines.md")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info("Annotation guidelines saved to %s", output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate annotation guidelines")
    parser.add_argument("--input", default="data/10k_1A/classified.json")
    parser.add_argument("--output-dir", default="data/10k_1A")
    parser.add_argument("--top-k", type=int, default=3,
                        help="Number of example paragraphs per category")
    args = parser.parse_args()
    generate_guidelines(args.input, args.output_dir, args.top_k)


if __name__ == "__main__":
    main()
