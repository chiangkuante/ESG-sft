from __future__ import annotations

from textwrap import dedent


CATEGORY_DEFINITIONS = dedent(
    """\
    Climate Change: This topic includes discussions about carbon emissions or climate change, including initiatives to increase carbon efficiency, environmental technologies, renewable energy, and the development or refurbishment of buildings with leading ecological design features.

    Natural Capital: This topic includes discussions about water stress, biodiversity, land use, and raw materials sourcing. For water stress, we include discussions of how companies manage risks of water shortages, such as by employing efficient water processes, water recycling, and alternative water sources. For biodiversity and land use, we include discussions about programs and policies designed to protect biodiversity and address community land-use concerns. For raw materials sourcing, we include discussions about policies and procedures to source materials with lower environmental impact, such as seafood/aquaculture, timber/paper, palm oil, beef/dairy, leather, and cotton.

    Pollution and Waste: This topic includes discussions about toxic emissions, packaging materials, and electronic waste. For toxic emissions and waste, we include discussions of pollution, contamination, and emission of toxic and carcinogenic substances and wastewater. For packaging materials and waste, we include discussions of product packaging content and end-of-life recycling or disposal of packaging materials. We include discussions about the recycling and removal of end-of-life electronic products for electronic waste.

    Human Capital: This topic includes discussions about labor management, health and safety, human capital development and training, and supply chain labor standards. For labor management, we include discussions workforce management, risk of workflow disruptions, labor productivity issues, employee diversity, and pay equality (non-executive). For health and safety, we include discussions of employee health and safety (H&S) programs such as H&S policies and their implementations, H&S training, and safety certifications. For human capital development and training, we include discussions of the ability to attract, retain, and develop human capital based on benefits, training, development programs, and employee engagement provided. For supply chain labor standards, we include discussions of supply chain production disruptions and brand value damage due to sub-standard treatment of workers in the company’s supply chain or reliance on raw materials that originate in areas associated with severe human rights and labor rights issues (e.g., slave labor and child labor).

    Product Liability: This topic includes discussing product safety and quality, privacy and data security, chemical safety, consumer financial protection, and health and demographic risk. For product safety and quality, we include discussion of product recalls, losing customer trust through product quality concerns, or product safety and quality certifications. For privacy and data security, we have discussions of data security breaches, the controversial use of personal data, and company data privacy policies and data security management systems. For chemical safety, we include discussions of the use or presence of chemicals of concern and procedures relating to chemical safety and its impact on customers. For consumer financial protection, we include discussions of the transparency of financial products based on borrowers’ ability to repay and initiatives to protect customers through product transparency. For health and demographic risk, we include discussions of public health trends and demographic changes, growth opportunities in the market for healthier products, and improved nutritional profiles.

    Community Relations: This topic includes discussions of a firm’s interaction with its local communities, including access to communications, access to finance, and access to healthcare. We include discussions about opportunities in historically underserved markets, such as developing countries and underserved populations, and relevant philanthropic efforts.

    Corporate Governance: This topic includes discussions on shareholders and ownership, board of directors, executive pay, and internal controls. For shareholders and ownership, we include discussions regarding ownership structure, control structure, and shareholders. For the board of directors, we include discussions of the board’s independence from management, board skills and diversity, and board effectiveness. For executive pay, we include CEO and other executives’ pay practices and specific pay figures, performance incentives, and overall pay plan design. For internal control, we consider internal controls, audit matters, audit committee matters, and internal audit matters.

    Business Ethics and Values: This topic includes discussions about ethical components such as a firm’s values and controversies. We include discussions about the ethical conduct of business, fraud, corruption, bribery, fiduciary responsibilities, conflicts of interest, misrepresentation, bias, negligence, political contributions, negative accounting events, and other behaviors which may have ethical components.

    Non-ESG: The paragraph does not primarily discuss any of the ESG categories above."""
)

VALID_CATEGORY_NAMES = dedent(
    """\
    Valid category names:
    - Climate Change
    - Natural Capital
    - Pollution & Waste
    - Human Capital
    - Product Liability
    - Community Relations
    - Corporate Governance
    - Business Ethics & Values
    - Non-ESG"""
)

CLASSIFICATION_RULES = dedent(
    """\
    Classification rules:
    - Choose the SINGLE most relevant category.
    - Only classify as an ESG category if the paragraph primarily discusses that topic.
    - If uncertain between an ESG category and Non-ESG, lean toward Non-ESG.
    - Focus on the main risk discussed in the paragraph, not incidental keywords."""
)


CLASSIFY_SYSTEM_PROMPT = dedent(
    f"""\
    You are an ESG (Environmental, Social, and Governance) risk classification expert specialized in analyzing U.S. 10-K Item 1A risk-factor paragraphs.

    Your job is to determine the single best label among the 9 ESG categories and explain the reasoning clearly.

    Use the following topic definitions as the primary decision standard:

    {CATEGORY_DEFINITIONS}

    {CLASSIFICATION_RULES}

    {VALID_CATEGORY_NAMES}"""
)


CLASSIFY_USER_TEMPLATE = dedent(
    """\
    Classify the following paragraph from a 10-K filing into one of the 9 ESG categories.

    === OUTPUT FORMAT ===
    - Return the final answer using exactly these two lines:
      Reasoning: <brief explanation>
      Label: <one valid category name>

    === PARAGRAPH TO CLASSIFY ===
    {text}"""
)


REASONING_SYSTEM_PROMPT = dedent(
    """\
    You are generating high-quality training rationales for ESG classification of U.S. 10-K Item 1A risk-factor paragraphs.

    Requirements:
    - Use the provided gold label as correct.
    - Explain which phrases or concepts support the label.
    - Briefly rule out the most plausible alternative categories when useful.
    - Do not use markdown, bullet points, XML, or JSON inside the reasoning text itself.
    - Return valid JSON only when requested."""
)


def build_reasoning_batch_prompt(batch: list[dict[str, str]]) -> str:
    blocks = []
    for item in batch:
        blocks.append(
            "\n".join(
                [
                    f"paragraph_id: {item['paragraph_id']}",
                    f"gold_label: {item['label']}",
                    "combined_text:",
                    item["combined_text"],
                ]
            )
        )

    return dedent(
        f"""\
        Generate one reasoning paragraph for each item below.

        Output format:
        - Return a JSON array.
        - Each object must contain exactly: paragraph_id, reasoning.
        - Keep the same paragraph_id values.
        - Each reasoning should be 3-5 sentences, concise but specific.
        - Each reasoning should explain why the gold label is correct using the definitions below.

        Category definitions:
        {CATEGORY_DEFINITIONS}

        Items:

        {chr(10).join(blocks)}"""
    )


def build_structured_answer(reasoning: str, label: str) -> str:
    cleaned_reasoning = " ".join(str(reasoning).split())
    return f"Reasoning: {cleaned_reasoning}\nLabel: {label}"
