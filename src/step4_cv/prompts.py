from __future__ import annotations

from typing import Any

from src.step4_cv.common import ESG_CATEGORIES


CATEGORY_DEFINITIONS = """Climate Change: Discussion of carbon emissions, climate change, renewable energy, climate regulation, decarbonization, or climate-related transition/physical risk.

Natural Capital: Discussion of water stress, biodiversity, land use, remediation, or environmentally preferable raw-material sourcing.

Pollution & Waste: Discussion of pollution, toxic emissions, wastewater, contamination, packaging waste, recycling, or e-waste disposal.

Human Capital: Discussion of labor management, employee health and safety, workforce retention/development, diversity, compensation fairness, or supply-chain labor standards.

Product Liability: Discussion of product safety/quality, recalls, data privacy, cybersecurity, consumer protection, chemical safety, or public-health risk tied to products/services.

Community Relations: Discussion of community engagement, underserved populations, access to communications/finance/healthcare, community impact, or philanthropy.

Corporate Governance: Discussion of shareholders, ownership/control, board structure, executive compensation, audit, or internal controls.

Business Ethics & Values: Discussion of fraud, corruption, bribery, misconduct, fiduciary duty, conflicts of interest, political contributions, misrepresentation, or ethical controversy.

Non-ESG: Paragraph does not primarily discuss any of the above ESG topics."""


SYNTHETIC_SYSTEM_PROMPT = """You are generating high-quality synthetic training data for ESG risk classification on U.S. 10-K Item 1A risk-factor paragraphs.

Requirements:
- Produce realistic 10-K risk-factor language.
- Keep the target label correct and unambiguous.
- Vary wording, scenario, and company context across samples.
- Do not copy or lightly rewrite the seed examples.
- Return valid JSON only.
- Every item must contain: risk_heading, paragraph_text, combined_text, label, generation_notes.
- combined_text must equal risk_heading + "\\n" + paragraph_text.
- label must be exactly one of the valid ESG labels.
"""


def build_synthetic_user_prompt(
    label: str,
    needed_count: int,
    seed_examples: list[dict[str, Any]],
    boundary_focus: bool,
) -> str:
    examples_text = []
    for index, example in enumerate(seed_examples, start=1):
        examples_text.append(
            "\n".join(
                [
                    f"Seed Example {index}",
                    f"paragraph_id: {example['paragraph_id']}",
                    f"label: {example['label']}",
                    "combined_text:",
                    example["combined_text"],
                ]
            )
        )

    boundary_line = (
        "Include a meaningful share of boundary cases that FinBERT might wrongly collapse into Non-ESG, while keeping the final label clearly correct."
        if boundary_focus
        else "Focus on diverse but clear in-class examples."
    )

    return f"""Generate {needed_count} synthetic 10-K Item 1A paragraphs for the label "{label}".

Valid labels:
{chr(10).join(f"- {item}" for item in ESG_CATEGORIES)}

Category definitions:
{CATEGORY_DEFINITIONS}

Generation guidance:
- Output a JSON array with exactly {needed_count} objects.
- Each object must have keys: risk_heading, paragraph_text, combined_text, label, generation_notes.
- risk_heading should look like a plausible risk-factor heading.
- paragraph_text should read like a realistic 10-K risk paragraph, typically 120-260 words.
- Avoid company-specific names copied from the seeds.
- Avoid boilerplate placeholders like "Company X".
- {boundary_line}

Seed examples for style and label boundary:

{chr(10).join(examples_text)}
"""
