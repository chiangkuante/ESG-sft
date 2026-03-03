"""
ESG 推理蒸餾腳本

功能：
1. 呼叫 GPT-5-mini 為每筆訓練資料生成 chain-of-thought 推理
2. 使用 checkpoint 機制，斷線後可接續
3. 將結果組裝成 Unsloth ShareGPT 對話格式（用於 Gemma 3 微調）

使用方式：
    uv run src/reasoning/reason.py

環境變數：
    OPENAI_API_KEY : OpenAI API key（從 .env 載入）
    DRY_RUN=1      : 只跑前 5 筆，用於快速驗證
"""

import os
import json
import time
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ============================================================
# 路徑設定
# ============================================================
TRAIN_FILE = "data/fine-tuning-data/train.json"
OUTPUT_DIR = Path("data/fine-tuning-data")
CHECKPOINT_FILE = OUTPUT_DIR / "reasoning_checkpoint.json"
REASONING_OUTPUT = OUTPUT_DIR / "train_with_reasoning.json"
SFT_OUTPUT = OUTPUT_DIR / "train_sft.json"

# ============================================================
# 類別定義（與 classify_with_llms.py 完全一致）
# ============================================================
CATEGORY_DEFINITIONS = """Climate Change: This topic includes discussions about carbon emissions or climate change, including \
initiatives to increase carbon efficiency, environmental technologies, renewable energy, and the \
development or refurbishment of buildings with leading ecological design features. The following are \
some examples:
- We are also excited about our commitment to securing 100% of our purchased electricity from renewable sources by 2025, reducing our operational carbon footprint by 30%.
- In 2018, we acquired a 43.83% interest in Silicon Ranch, a leading US developer, owner, and operator of solar assets.
- Beginning in 2012 through the end of 2016, we have converted 19 plants from coal to natural gas or steam.
- 40% of our operations are to be certified under a green building standard by 2018.

Natural Capital: This topic includes discussions about water stress, biodiversity, land use, and raw \
materials sourcing. For water stress, we include discussions of how companies manage risks of water \
shortages, such as by employing efficient water processes, water recycling, and alternative water \
sources. For biodiversity and land use, we include discussions about programs and policies designed to \
protect biodiversity and address community land-use concerns. For raw materials sourcing, we include \
discussions about policies and procedures to source materials with lower environmental impact, such \
as seafood/aquaculture, timber/paper, palm oil, beef/dairy, leather, and cotton.

Pollution & Waste: This topic includes discussions about toxic emissions, packaging materials, and \
electronic waste. For toxic emissions and waste, we include discussions of pollution, contamination, \
and emission of toxic and carcinogenic substances and wastewater. For packaging materials and waste, \
we include discussions of product packaging content and end-of-life recycling or disposal of \
packaging materials. We include discussions about the recycling and removal of end-of-life electronic \
products for electronic waste.

Human Capital: This topic includes discussions about labor management, health and safety, human \
capital development and training, and supply chain labor standards. For labor management, we include \
discussions workforce management, risk of workflow disruptions, labor productivity issues, employee \
diversity, and pay equality (non-executive). For health and safety, we include discussions of employee \
health and safety (H&S) programs. For human capital development and training, we include discussions \
of the ability to attract, retain, and develop human capital based on benefits, training, development \
programs, and employee engagement provided. For supply chain labor standards, we include discussions \
of supply chain production disruptions and brand value damage due to sub-standard treatment of workers.

Product Liability: This topic includes discussing product safety and quality, privacy and data \
security, chemical safety, consumer financial protection, and health and demographic risk. For product \
safety and quality, we include discussion of product recalls, losing customer trust through product \
quality concerns. For privacy and data security, we have discussions of data security breaches, the \
controversial use of personal data. For chemical safety, we include discussions of the use or presence \
of chemicals of concern. For consumer financial protection, we include discussions of the transparency \
of financial products. For health and demographic risk, we include discussions of public health trends.

Community Relations: This topic includes discussions of a firm's interaction with its local \
communities, including access to communications, access to finance, and access to healthcare. We \
include discussions about opportunities in historically underserved markets, such as developing \
countries and underserved populations, and relevant philanthropic efforts.

Corporate Governance: This topic includes discussions on shareholders and ownership, board of \
directors, executive pay, and internal controls. For shareholders and ownership, we include discussions \
regarding ownership structure, control structure, and shareholders. For the board of directors, we \
include discussions of the board's independence from management, board skills and diversity, and \
board effectiveness. For executive pay, we include CEO and other executives' pay practices. For \
internal control, we consider internal controls, audit matters, and internal audit matters.

Business Ethics & Values: This topic includes discussions about ethical components such as a \
firm's values and controversies. We include discussions about the ethical conduct of business, fraud, \
corruption, bribery, fiduciary responsibilities, conflicts of interest, misrepresentation, bias, negligence, \
political contributions, negative accounting events, and other behaviors which may have ethical components.

Non-ESG: The paragraph does not primarily discuss any of the above ESG topics. Typical Non-ESG content \
includes: general financial performance, revenue/earnings discussions, market competition, product/service \
descriptions, operational logistics, legal boilerplate, accounting policies, and general business strategy \
unrelated to ESG."""


# ============================================================
# 1. 推理生成 Prompt
# ============================================================

REASON_SYSTEM_PROMPT = """\
You are an ESG classification expert generating teaching-style reasoning traces for training data.

Given a 10-K paragraph from Item 1A (Risk Factors), its CORRECT ESG category label, and the category definitions, \
produce a concise step-by-step reasoning (3-5 sentences) that explains why this paragraph belongs to that category.

Your reasoning MUST:
1. Identify the key ESG-related keywords or concepts in the paragraph.
2. Explain how those concepts map to the correct category's definition.
3. Briefly explain why other plausible categories do not apply (1 sentence).
4. End with exactly: "Therefore, this paragraph is classified as: {the correct label}"

Constraints:
- Plain text only. No markdown formatting, no bullet points, no headers.
- Total output should be 3-5 sentences, concise and instructional.
- Never contradict the provided correct label.
- Only reference content explicitly present in the paragraph."""

REASON_USER_TEMPLATE = """\
=== CATEGORY DEFINITIONS ===
{definitions}

=== PARAGRAPH ===
{text}

=== CORRECT LABEL ===
{label}

Generate the reasoning trace. Your reasoning must support the correct label above."""


# ============================================================
# 2. 分類用 Prompt（與 classify_with_llms.py 的 user prompt 一致，
#    用於 SFT 時的 instruction 部分，不含答案）
# ============================================================

CLASSIFY_SYSTEM_PROMPT = """\
You are an ESG (Environmental, Social, and Governance) risk classification expert specialized in \
analyzing U.S. 10-K filings. Your task is to classify paragraphs from Item 1A (Risk Factors) sections \
into exactly one of the following 9 categories.

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

CLASSIFY_USER_TEMPLATE = """\
Classify the following paragraph from a 10-K filing into one of the 9 ESG categories based on the definitions below.

=== CATEGORY DEFINITIONS ===
{definitions}

=== IMPORTANT RULES ===
- Choose the SINGLE most relevant category.
- Only classify as an ESG category if the paragraph PRIMARILY discusses that topic.
- If uncertain between an ESG category and Non-ESG, lean toward Non-ESG.

=== PARAGRAPH TO CLASSIFY ===
{text}"""


# ============================================================
# 3. 批次呼叫 API 生成推理
# ============================================================

MAX_RETRIES = 3
RETRY_DELAYS = [2, 4, 8]  # 指數退避秒數


def generate_reasoning(text: str, label: str, model: str = "gpt-5-mini") -> str:
    """呼叫 GPT-5-mini 生成一段推理文字，含重試邏輯。"""
    user_content = REASON_USER_TEMPLATE.format(
        definitions=CATEGORY_DEFINITIONS,
        text=text,
        label=label,
    )
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.responses.create(
                model=model,
                instructions=REASON_SYSTEM_PROMPT,
                input=user_content,
                max_output_tokens=500,
                reasoning={"effort": "minimal"},
            )
            result = (resp.output_text or "").strip()
            if result:  # 非空才算成功
                return result
            # 空回應，重試
            print(f"    [RETRY {attempt+1}/{MAX_RETRIES}] Empty response")
        except Exception as e:
            last_error = e
            print(f"    [RETRY {attempt+1}/{MAX_RETRIES}] {e}")
        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_DELAYS[attempt])
    # 全部重試失敗
    raise RuntimeError(f"All {MAX_RETRIES} retries failed: {last_error}")


def validate_reasoning(reasoning: str, label: str) -> dict:
    """驗證推理是否包含正確標籤作為結論。接受多種格式變體。"""
    issues = []
    text_lower = reasoning.lower()
    label_lower = label.lower()
    # 接受多種結論格式
    patterns = [
        f"classified as: {label_lower}",
        f"classified as {label_lower}",
        f"classification is: {label_lower}",
        f"classification is {label_lower}",
        f"category is: {label_lower}",
        f"category is {label_lower}",
        f"categorized as: {label_lower}",
        f"categorized as {label_lower}",
    ]
    if not any(p in text_lower for p in patterns):
        issues.append(f"MISSING_CONCLUSION: label '{label}' not found in conclusion")
    return {"valid": len(issues) == 0, "issues": issues}


def batch_generate(train_data: list, model: str = "gpt-5-mini", delay: float = 0.5) -> list:
    """批次生成推理，支援 checkpoint。"""
    # 載入 checkpoint
    results = []
    done_ids = set()
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            results = json.load(f)
        done_ids = {r["paragraph_id"] for r in results}
        print(f"Resumed {len(results)} from checkpoint")

    remaining = [d for d in train_data if d["paragraph_id"] not in done_ids]
    total = len(train_data)

    print(f"Total: {total}, Remaining: {len(remaining)}")

    for i, item in enumerate(remaining):
        try:
            reasoning = generate_reasoning(item["text"], item["label"], model=model)

            # 驗證
            val = validate_reasoning(reasoning, item["label"])
            if not val["valid"]:
                # 自動修補：附加正確結尾
                reasoning += f"\nTherefore, this paragraph is classified as: {item['label']}"
                print(f"  [FIX] {item['paragraph_id']}: appended conclusion")

        except Exception as e:
            print(f"  FAILED {item['paragraph_id']}: {e}")
            reasoning = None  # 標記為失敗，不使用空殼 fallback

        results.append({
            "paragraph_id": item["paragraph_id"],
            "text": item["text"],
            "label": item["label"],
            "reasoning": reasoning or "",
            "failed": reasoning is None,
        })

        completed = len(done_ids) + i + 1
        print(f"  [{completed}/{total}] {item['paragraph_id']}: {item['label']}")

        # Checkpoint every 10 items
        if (i + 1) % 10 == 0:
            with open(CHECKPOINT_FILE, "w") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

        time.sleep(delay)

    # Final save
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return results


# ============================================================
# 4. 組裝成 SFT 訓練格式（ShareGPT for Unsloth）
# ============================================================

def convert_to_sft(results: list) -> list:
    """
    轉換成 Unsloth ShareGPT 對話格式。
    跳過推理失敗（failed=True）的項目。

    每筆資料：
    - system: 分類系統 prompt
    - user: 分類 user prompt（包含類別定義 + 段落文字，不含答案）
    - assistant: GPT-5-mini 生成的推理 + 最終分類
    """
    sft_data = []
    skipped = 0
    for r in results:
        if r.get("failed"):
            skipped += 1
            continue
        user_content = CLASSIFY_USER_TEMPLATE.format(
            definitions=CATEGORY_DEFINITIONS,
            text=r["text"],
        )

        sft_data.append({
            "conversations": [
                {"role": "system", "content": CLASSIFY_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": r["reasoning"]},
            ]
        })
    if skipped:
        print(f"  Skipped {skipped} failed items")
    return sft_data


# ============================================================
# Main
# ============================================================

def main():
    # 載入訓練資料
    with open(TRAIN_FILE) as f:
        train_data = json.load(f)
    print(f"Loaded {len(train_data)} training items from {TRAIN_FILE}")

    # DRY_RUN 模式
    if os.getenv("DRY_RUN"):
        n = int(os.getenv("DRY_RUN", "5"))
        train_data = train_data[:n]
        print(f"DRY_RUN mode: only processing first {n} items")

    # Step 1: 批次生成推理
    print(f"\n{'='*60}")
    print("  Step 1: Generating reasoning with GPT-5-mini")
    print(f"{'='*60}")
    results = batch_generate(train_data)

    # 儲存帶推理的中間結果
    with open(REASONING_OUTPUT, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    failed_count = sum(1 for r in results if r.get("failed"))
    print(f"\nSaved reasoning to {REASONING_OUTPUT}")
    print(f"  Success: {len(results) - failed_count}, Failed: {failed_count}")

    # Step 2: 轉換為 SFT 格式
    print(f"\n{'='*60}")
    print("  Step 2: Converting to SFT format")
    print(f"{'='*60}")
    sft_data = convert_to_sft(results)

    with open(SFT_OUTPUT, "w") as f:
        json.dump(sft_data, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(sft_data)} SFT samples to {SFT_OUTPUT}")

    # 顯示一個 sample
    print(f"\n{'='*60}")
    print("  Sample SFT entry")
    print(f"{'='*60}")
    sample = sft_data[0]
    print(f"System: {sample['conversations'][0]['content'][:100]}...")
    print(f"User: {sample['conversations'][1]['content'][:100]}...")
    print(f"Assistant: {sample['conversations'][2]['content'][:200]}...")

    # 清理 checkpoint
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        print(f"\nRemoved checkpoint file")

    print("\nDone!")


if __name__ == "__main__":
    main()
