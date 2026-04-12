"""
ESG Classification Benchmark - Unified Script

Usage:
  uv run src/step7_cv/classify_with_llms.py --model gpt
  uv run src/step7_cv/classify_with_llms.py --model claude
  uv run src/step7_cv/classify_with_llms.py --model gemini
  uv run src/step7_cv/classify_with_llms.py --model gemma
  uv run src/step7_cv/classify_with_llms.py --model all
  uv run src/step7_cv/classify_with_llms.py --model gpt --dry-run 3
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
  accuracy_score,
  classification_report,
  cohen_kappa_score,
  f1_score,
)
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# 1. VALID LABELS & NORMALIZATION
# ============================================================

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
  "pollution and waste": "Pollution & Waste",
  "pollution&waste": "Pollution & Waste",
  "business ethics and values": "Business Ethics & Values",
  "business ethics & value": "Business Ethics & Values",
  "business ethics": "Business Ethics & Values",
  "non esg": "Non-ESG",
  "nonesg": "Non-ESG",
  "non-esg.": "Non-ESG",
  "n/a": "Non-ESG",
  "none": "Non-ESG",
  "community": "Community Relations",
  "governance": "Corporate Governance",
  "corp governance": "Corporate Governance",
  "corporate gov": "Corporate Governance",
}


def normalize_label(raw: str) -> str:
  """Normalize LLM output to a valid category name."""
  cleaned = raw.strip().strip("\"'`").strip()
  cleaned = re.sub(r"^\d+[\.)\-]\s*", "", cleaned)
  cleaned = cleaned.strip("*").strip()
  # Pre-process: "and" -> "&"
  cleaned = cleaned.replace("Pollution and Waste", "Pollution & Waste")
  cleaned = cleaned.replace("Business Ethics and Values", "Business Ethics & Values")

  for label in VALID_LABELS:
    if cleaned.lower() == label.lower():
      return label
  if cleaned.lower() in LABEL_ALIASES:
    return LABEL_ALIASES[cleaned.lower()]
  for label in VALID_LABELS:
    if label.lower() in cleaned.lower():
      return label
  return "Unclassified"


# ============================================================
# 2. PROMPTS
# ============================================================

SYSTEM_PROMPT = """You are an ESG (Environmental, Social, and Governance) risk classification expert specialized in analyzing U.S. 10-K filings. Your task is to classify paragraphs from Item 1A (Risk Factors) sections into exactly one of the following 9 categories.

You must respond with ONLY the category name. Do not include any explanation, reasoning, numbering, or additional text.

Valid category names (respond with one of these exactly):
- Climate Change
- Natural Capital
- Pollution & Waste
- Human Capital
- Product Liability
- Community Relations
- Corporate Governance
- Business Ethics & Values
- Non-ESG

CRITICAL: Your entire response must be EXACTLY one of the 9 category names above. No other words, punctuation, or characters. Maximum 4 words."""

USER_PROMPT_TEMPLATE = """Classify the following paragraph from a 10-K filing into one of the 9 ESG categories based on the definitions and examples below.

=== CATEGORY DEFINITIONS ===

Appendix: Detail description and examples of the eight ESG topics:
Climate Change: This topic includes discussions about carbon emissions or climate change, including
initiatives to increase carbon efficiency, environmental technologies, renewable energy, and the
development or refurbishment of buildings with leading ecological design features. The following are
some examples:
- We are also excited about our commitment to securing 100% of our purchased electricity from
renewable sources by 2025, reducing our operational carbon footprint by 30%.
- In 2018, we acquired a 43.83% interest in Silicon Ranch, a leading US developer, owner, and
operator of solar assets. In 2019, Silicon Ranch announced the launch of a program that
combines clean electricity generation with carbon sequestration and ecosystem restoration.
- Beginning in 2012 through the end of 2016, we have converted 19 plants from coal to natural
gas or steam.
- 40% of our operations are to be certified under a green building standard by 2018.
Natural Capital: This topic includes discussions about water stress, biodiversity, land use, and raw
materials sourcing. For water stress, we include discussions of how companies manage risks of water
shortages, such as by employing efficient water processes, water recycling, and alternative water
sources. For biodiversity and land use, we include discussions about programs and policies designed to
protect biodiversity and address community land-use concerns. For raw materials sourcing, we include
discussions about policies and procedures to source materials with lower environmental impact, such
as seafood/aquaculture, timber/paper, palm oil, beef/dairy, leather, and cotton. The following are some
examples:
- This year, we continued Stella Artois' Buy a Lady a Drink program with Water.org, which aims
to tackle the global water crisis. Having to date supported nearly 800,000 people, we recently
announced our ambition to provide access to safe water to 3.5 million people in the developing
world by 2020.
- At the chemical complex in the Netherlands, our biodiversity action plan has helped maintain a
variety of flora and fauna. It has also raised awareness about biodiversity among technical
staff. For example, in 2019, planned maintenance work was adjusted after operators found a
protected bird species nesting in equipment.
- 100% of the paper used in our U.S direct marketing efforts was certified to be from sustainably
managed forests.
- During the period, Kimco remediated soil as a part of major development or redevelopment
projects at the following locations: Dania Pointe (Dania Beach, FL), Suburban Square
(Ardmore, PA), Lincoln Square (Philadelphia, PA).
Pollution and Waste: This topic includes discussions about toxic emissions, packaging materials, and
electronic waste. For toxic emissions and waste, we include discussions of pollution, contamination,
and emission of toxic and carcinogenic substances and wastewater. For packaging materials and waste,
we include discussions of product packaging content and end-of-life recycling or disposal of
packaging materials. We include discussions about the recycling and removal of end-of-life electronic
products for electronic waste. The following are some examples:
- We also have programs in place to reduce the number of operational spills over the long term.
In 2019, we continued to carry out vital work to clean up Bodo, an area badly affected by oil
spills.
- Through lightweighting and packaging reduction initiatives, we have reduced the amount of
packaging we use by 126,800 tons since 2012, exceeding our 2017 reduction goal of 100,000
tons a year early.
- All of these programs reduce waste and encourage reuse by ensuring that valuable products
can go back into the hands of customers rather than being sent to landfills.
- Several years ago, we saw an opportunity to create a single streamlined solution for tenants
that could provide them with more reliable and cost-effective waste services.
Human Capital: This topic includes discussions about labor management, health and safety, human
capital development and training, and supply chain labor standards. For labor management, we include
discussions workforce management, risk of workflow disruptions, labor productivity issues, employee
diversity, and pay equality (non-executive). For health and safety, we include discussions of employee
health and safety (H&S) programs such as H&S policies and their implementations, H&S training, and
safety certifications. For human capital development and training, we include discussions of the ability
to attract, retain, and develop human capital based on benefits, training, development programs, and
employee engagement provided. For supply chain labor standards, we include discussions of supply
chain production disruptions and brand value damage due to sub-standard treatment of workers in the
company's supply chain or reliance on raw materials that originate in areas associated with severe
human rights and labor rights issues (e.g., slave labor and child labor). The following are some
examples:
- We provide formal channels to guide colleagues and leaders on decisions related to flextime,
part-time, compressed work weeks, job sharing, and remote work.
- We use technology systems on our trucks to track driver behaviors, which has increased
accountability among our managers and resulted in a reduction in speeding and safer fleet
operations.
- Enterprise Leadership delivers targeted leadership development programs to colleagues at
specific stages of their careers.
- This can lead to unfair treatment of workers, which is why we continue our efforts to source
secondary raw materials in a way that's aligned with our "Fair and Equal" agenda.
Product Liability: This topic includes discussing product safety and quality, privacy and data
security, chemical safety, consumer financial protection, and health and demographic risk. For product
safety and quality, we include discussion of product recalls, losing customer trust through product
quality concerns, or product safety and quality certifications. For privacy and data security, we have
discussions of data security breaches, the controversial use of personal data, and company data privacy
policies and data security management systems. For chemical safety, we include discussions of the use
or presence of chemicals of concern and procedures relating to chemical safety and its impact on
customers. For consumer financial protection, we include discussions of the transparency of financial
products based on borrowers' ability to repay and initiatives to protect customers through product
transparency. For health and demographic risk, we include discussions of public health trends and
demographic changes, growth opportunities in the market for healthier products, and improved
nutritional profiles. The following are some examples:
- Our products are designed and tested to comply with all applicable safety regulations in the
countries where the products are sold.
- Information Security oversees a comprehensive program to help predict, protect, detect,
respond to, and recover from cyberattacks.
- For example, we have phased out any chemicals suspected of causing medical reactions or
harm.
- The Smarter CreditTM Center includes resources to help customers understand, build, and
improve credit, as well as manage debt and plan for large purchases.
Community Relations: This topic includes discussions of a firm's interaction with its local
communities, including access to communications, access to finance, and access to healthcare. We
include discussions about opportunities in historically underserved markets, such as developing
countries and underserved populations, and relevant philanthropic efforts. The following are some
examples:
- We've gone from sourcing 70% of barley locally in 2014 to 86% in 2016, and we're on track to
achieve 100% local sourcing by 2017.
- In addition, we have committed to a package of public interest commitments, which include a
commitment to invest 1 billion ZAR in areas including supporting smallholder farmers and
enterprise development.
- We aim to improve access for local people to health care and treatments for diseases such as
cancer.
- Participants volunteer for significant causes such as disaster relief, hunger, medical research,
home building or youth mentoring, and groups are encouraged to serve together as a means of
multiplying their impact and fostering team spirit.
Corporate Governance: This topic includes discussions on shareholders and ownership, board of
directors, executive pay, and internal controls. For shareholders and ownership, we include discussions
regarding ownership structure, control structure, and shareholders. For the board of directors, we
include discussions of the board's independence from management, board skills and diversity, and
board effectiveness. For executive pay, we include CEO and other executives' pay practices and 
specific pay figures, performance incentives, and overall pay plan design. For internal control, we
consider internal controls, audit matters, audit committee matters, and internal audit matters. The
following are some examples:
- Based on filings made under Sections 13(d) and 13(g) of the Securities Exchange Act of 1934,
as amended, as of December 31, 2020, the only persons or entities known by us to be a
beneficial owner of more than 5% of our common stock were as follows.
- Our Board is composed entirely of independent directors other than our chairman and CEO,
and is diverse, with diversity reflecting gender, age, race, ethnicity, background, professional
experience, and perspectives.
- We announced plans in 2018 to link executive remuneration to short-term targets to reduce the
Net Carbon Footprint of the energy products we sell, including our customers' emissions from
their use of our energy products.
- The appointment by the Audit and Compliance Committee of the Company's Board of
Directors of PricewaterhouseCoopers LLP, as an independent registered public accounting
firm for the Company, to audit the financial statements of the Company and its subsidiaries for
2022 is hereby ratified and approved.
Business Ethics and Values: This topic includes discussions about ethical components such as a
firm's values and controversies. We include discussions about the ethical conduct of business, fraud,
corruption, bribery, fiduciary responsibilities, conflicts of interest, misrepresentation, bias, negligence,
political contributions, negative accounting events, and other behaviors which may have ethical
components. The following is an example:
- At the heart of our culture is what we call our Blue Box Values - a set of seven guiding
principles that every employee pledges to embrace and work by each day.
- Huntington is dedicated to uncompromising integrity in all that it does and how it relates to its
internal colleagues and to persons outside Huntington.
- In reaching this settlement, we neither admitted nor denied the claims in the order and agreed
to pay a civil monetary penalty of $5.5 million.
- In addition, certain private parties as well as state attorneys general and other antitrust
authorities may challenge the transactions under U.S. or foreign antitrust laws under certain
circumstances.

Non-ESG
The paragraph does not primarily discuss any of the above ESG topics. Typical Non-ESG content includes: general financial performance, revenue/earnings discussions, market competition, product/service descriptions, operational logistics, legal boilerplate, accounting policies, and general business strategy unrelated to ESG.

=== IMPORTANT RULES ===
- Choose the SINGLE most relevant category. If a paragraph touches multiple ESG topics, select the PRIMARY one.
- Only classify as an ESG category if the paragraph PRIMARILY discusses that topic. Incidental mentions do not qualify.
- If uncertain between an ESG category and Non-ESG, lean toward Non-ESG.
- Respond with ONLY the category name, nothing else.

=== PARAGRAPH TO CLASSIFY ===
{text}

Remember: Output ONLY the category name. One of the 9 valid labels. No explanation. No punctuation. Maximum 4 words."""


# ============================================================
# 3. API CALLERS
# ============================================================

def call_openai_model(text: str, api_key: str, model: str = "gpt-5-mini-2025-08-07") -> str:
  from openai import OpenAI
  client = OpenAI(api_key=api_key)
  prompt = USER_PROMPT_TEMPLATE.replace("{text}", text)
  resp = client.responses.create(
    model=model,
    instructions=SYSTEM_PROMPT,
    input=prompt,
    max_output_tokens=2000,
    reasoning={"effort": "low"},
  )
  return (resp.output_text or "").strip()


def call_anthropic_model(text: str, api_key: str, model: str = "claude-sonnet-4-5-20250929") -> str:
  import anthropic
  client = anthropic.Anthropic(api_key=api_key)
  response = client.messages.create(
    model=model,
    max_tokens=200,
    temperature=0,
    system=SYSTEM_PROMPT,
    messages=[
      {"role": "user", "content": USER_PROMPT_TEMPLATE.replace("{text}", text)},
    ],
  )
  return response.content[0].text.strip()


def call_gemini_model(text: str, api_key: str, model: str = "gemini-3-flash-preview") -> str:
  import google.generativeai as genai
  genai.configure(api_key=api_key)
  gen_model = genai.GenerativeModel(
    model_name=model,
    system_instruction=SYSTEM_PROMPT,
  )
  response = gen_model.generate_content(
    USER_PROMPT_TEMPLATE.replace("{text}", text),
    generation_config=genai.types.GenerationConfig(
      temperature=0,
      max_output_tokens=200,
    ),
  )
  return response.text.strip()


# ============================================================
# 4. LOCAL GEMMA MODEL (Lazy Singleton)
# ============================================================

_gemma_model = None
_gemma_tokenizer = None


def _load_gemma(model_path: str):
  """Load Gemma model once (lazy singleton)."""
  global _gemma_model, _gemma_tokenizer
  if _gemma_model is not None:
    return _gemma_model, _gemma_tokenizer

  print(f" Loading Gemma model from {model_path}...")
  from unsloth import FastModel
  from unsloth.chat_templates import get_chat_template

  model, tokenizer = FastModel.from_pretrained(
    model_name=model_path,
    max_seq_length=2048,
    load_in_4bit=True,
  )
  tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")
  FastModel.for_inference(model)

  _gemma_model = model
  _gemma_tokenizer = tokenizer
  print(" Gemma model loaded.")
  return model, tokenizer


def call_gemma_model(text: str, model_path: str, **_kwargs) -> str:
  """Run local Gemma inference."""
  model, tokenizer = _load_gemma(model_path)

  messages = [
    {"role": "user", "content": SYSTEM_PROMPT + "\n\n" + USER_PROMPT_TEMPLATE.replace("{text}", text)},
  ]
  inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
  ).to("cuda")

  outputs = model.generate(
    input_ids=inputs,
    max_new_tokens=64,
    temperature=1.0,
    top_p=0.95,
    top_k=64,
  )

  # Decode only generated tokens
  generated = outputs[0][inputs.shape[-1]:]
  decoded = tokenizer.decode(generated, skip_special_tokens=True)
  return decoded.strip()


# ============================================================
# 5. MODEL REGISTRY
# ============================================================

MODELS = {
  "gpt": {
    "name": "gpt-5-mini",
    "model_str": "gpt-5-mini-2025-08-07",
    "api_key_env": "OPENAI_API_KEY",
    "caller": call_openai_model,
    "type": "api",
  },
  "claude": {
    "name": "Claude-3.5-Sonnet",
    "model_str": "claude-sonnet-4-5-20250929",
    "api_key_env": "ANTHROPIC_API_KEY",
    "caller": call_anthropic_model,
    "type": "api",
  },
  "gemini": {
    "name": "gemini-3-flash",
    "model_str": "gemini-3-flash-preview",
    "api_key_env": "GEMINI_API_KEY",
    "caller": call_gemini_model,
    "type": "api",
  },
  "gemma": {
    "name": "Gemma-3-4B",
    "model_str": "src/step7_cv/gemma_3_lora", # LoRA adapter path
    "caller": call_gemma_model,
    "type": "local",
  },
}


# ============================================================
# 6. BATCH CLASSIFICATION
# ============================================================

def classify_batch(
  data: list[dict],
  model_info: dict,
  output_dir: str,
  delay: float = 0.5,
  dry_run: bool = False,
) -> list[dict]:
  """Classify all paragraphs using a specific model."""
  model_key = model_info["name"].split("-")[0].lower()
  caller = model_info["caller"]
  model_str = model_info["model_str"]
  is_local = model_info["type"] == "local"

  # API key check for remote models
  if not is_local:
    api_key = os.getenv(model_info["api_key_env"])
    if not api_key:
      print(f"Skipping {model_info['name']}: {model_info['api_key_env']} not found.")
      return []

  os.makedirs(output_dir, exist_ok=True)
  checkpoint_path = os.path.join(output_dir, f"{model_key}_checkpoint.json")

  # Resume from checkpoint (skip in dry-run)
  results = []
  done_ids = set()
  if not dry_run and os.path.exists(checkpoint_path):
    with open(checkpoint_path) as f:
      results = json.load(f)
    done_ids = {r["paragraph_id"] for r in results}
    print(f"Resumed {len(results)} results from checkpoint for {model_info['name']}")

  remaining = [d for d in data if d["paragraph_id"] not in done_ids]
  total = len(data)

  for i, item in enumerate(remaining):
    error_msg = None
    try:
      raw = ""
      if is_local:
        raw = caller(item["text"], model_path=model_str)
      else:
        for attempt in range(3):
          raw = caller(item["text"], api_key, model=model_str)
          if raw:
            break
          time.sleep(2)
      label = normalize_label(raw)
    except Exception as e:
      print(f" ERROR on {item['paragraph_id']}: {e}")
      error_msg = str(e)
      raw = ""
      label = "Unclassified"

    result_entry = {
      "paragraph_id": item["paragraph_id"],
      "ground_truth": item["label"],
      "finbert_label": item.get("finbert_label", ""),
      "raw_response": raw,
      "predicted_label": label,
    }
    if error_msg:
      result_entry["error"] = error_msg

    results.append(result_entry)

    completed = len(done_ids) + i + 1
    print(f" [{completed}/{total}] {item['paragraph_id']}: {label} (GT: {item['label']})")

    # Checkpoint every 10 items (skip in dry-run)
    if not dry_run and (i + 1) % 10 == 0:
      with open(checkpoint_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    if not is_local:
      time.sleep(delay)

  # Save results (skip in dry-run)
  if not dry_run:
    final_path = os.path.join(output_dir, f"{model_key}_results.json")
    with open(final_path, "w") as f:
      json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(results)} results to {final_path}")

    if os.path.exists(checkpoint_path):
      os.remove(checkpoint_path)
  else:
    print(f"\n[DRY-RUN] {len(results)} results (not saved)")

  return results


# ============================================================
# 7. EVALUATION METRICS
# ============================================================

def evaluate_model(results: list[dict], model_name: str) -> dict:
  """Compute accuracy, F1, and classification report."""
  y_true = [r["ground_truth"] for r in results]
  y_pred = [r["predicted_label"] for r in results]

  valid = [(t, p) for t, p in zip(y_true, y_pred) if p != "Unclassified"]
  if not valid:
    print(f"No valid predictions for {model_name}")
    return {}

  y_true_v, y_pred_v = zip(*valid)
  unclassified = len(y_true) - len(valid)

  acc = accuracy_score(y_true_v, y_pred_v)
  macro_f1 = f1_score(y_true_v, y_pred_v, average="macro", labels=VALID_LABELS, zero_division=0)
  micro_f1 = f1_score(y_true_v, y_pred_v, average="micro", labels=VALID_LABELS, zero_division=0)

  print(f"\n{'='*60}")
  print(f" {model_name} Evaluation Results")
  print(f"{'='*60}")
  print(f" Accuracy:   {acc:.4f}")
  print(f" Macro F1:   {macro_f1:.4f}")
  print(f" Micro F1:   {micro_f1:.4f}")
  print(f" Unclassified: {unclassified}")
  print(f"\n{classification_report(y_true_v, y_pred_v, labels=VALID_LABELS, zero_division=0)}")

  return {
    "model": model_name,
    "accuracy": round(acc, 4),
    "macro_f1": round(macro_f1, 4),
    "micro_f1": round(micro_f1, 4),
    "unclassified": unclassified,
  }


def compute_kappa_matrix(all_results: dict[str, list[dict]]) -> pd.DataFrame:
  """Compute pairwise Cohen's Kappa between all models and ground truth."""
  models = list(all_results.keys())
  all_models = ["Ground Truth"] + models

  id_sets = [set(r["paragraph_id"] for r in results) for results in all_results.values()]
  common_ids = sorted(set.intersection(*id_sets)) if id_sets else []

  pred_lookup = {}
  for model_name, results in all_results.items():
    pred_lookup[model_name] = {r["paragraph_id"]: r["predicted_label"] for r in results}

  gt_lookup = {}
  for results in all_results.values():
    for r in results:
      gt_lookup[r["paragraph_id"]] = r["ground_truth"]
    break

  vectors = {"Ground Truth": [gt_lookup[pid] for pid in common_ids]}
  for model_name in models:
    vectors[model_name] = [pred_lookup[model_name].get(pid, "Unclassified") for pid in common_ids]

  n = len(all_models)
  kappa_matrix = pd.DataFrame(index=all_models, columns=all_models, dtype=float)
  for i in range(n):
    for j in range(n):
      if i == j:
        kappa_matrix.iloc[i, j] = 1.00
      elif j > i:
        k = cohen_kappa_score(vectors[all_models[i]], vectors[all_models[j]])
        kappa_matrix.iloc[i, j] = round(k, 2)
        kappa_matrix.iloc[j, i] = round(k, 2)

  return kappa_matrix


# ============================================================
# 8. MAIN
# ============================================================

def main():
  parser = argparse.ArgumentParser(description="ESG Classification Benchmark")
  parser.add_argument(
    "--model",
    choices=list(MODELS.keys()) + ["all"],
    required=True,
    help="Which model to run, or 'all' for all models",
  )
  parser.add_argument(
    "--test-data", type=str,
    default="data/fine-tuning-data/test.json",
    help="Path to test.json",
  )
  parser.add_argument(
    "--output-dir", type=str,
    default="results_test_set",
    help="Output directory",
  )
  parser.add_argument(
    "--gemma-model", type=str,
    default="src/step7_cv/gemma_3_lora",
    help="Local Gemma model path (HF name or local dir)",
  )
  parser.add_argument(
    "--dry-run", type=int, metavar="N",
    help="Only process the first N items (no saving)",
  )
  args = parser.parse_args()

  # Load test data
  with open(args.test_data) as f:
    test_data = json.load(f)
  print(f"Loaded {len(test_data)} test items from {args.test_data}")

  # Apply dry-run limit
  dry_run = args.dry_run is not None
  if dry_run:
    test_data = test_data[:args.dry_run]
    print(f"[DRY-RUN] Processing first {args.dry_run} items only")

  # Override gemma model path
  MODELS["gemma"]["model_str"] = args.gemma_model

  # Determine which models to run
  if args.model == "all":
    model_keys = list(MODELS.keys())
  else:
    model_keys = [args.model]

  all_results = {}

  # Run inference
  for key in model_keys:
    info = MODELS[key]
    print(f"\n{'='*60}")
    print(f" Running inference for {info['name']}")
    print(f"{'='*60}")
    res = classify_batch(test_data, info, args.output_dir, delay=1.0, dry_run=dry_run)
    if res:
      all_results[info["name"]] = res

  # Load existing results for models not run this time
  if not dry_run:
    for key, info in MODELS.items():
      if info["name"] not in all_results:
        model_key = info["name"].split("-")[0].lower()
        res_path = os.path.join(args.output_dir, f"{model_key}_results.json")
        if os.path.exists(res_path):
          with open(res_path) as f:
            all_results[info["name"]] = json.load(f)

  # Add FinBERT baseline
  finbert_results = [
    {
      "paragraph_id": d["paragraph_id"],
      "ground_truth": d["label"],
      "predicted_label": d.get("finbert_label", "Unclassified"),
    }
    for d in test_data
  ]
  all_results["FinBERT-ESG-9"] = finbert_results

  # Evaluate
  print(f"\n{'='*60}")
  print(f" Evaluating Models")
  print(f"{'='*60}")

  all_metrics = []
  for name, res in all_results.items():
    metrics = evaluate_model(res, name)
    if metrics:
      all_metrics.append(metrics)

  if all_metrics:
    summary = pd.DataFrame(all_metrics)
    if not dry_run:
      summary.to_csv(os.path.join(args.output_dir, "summary_metrics.csv"), index=False)
    print(f"\n{'='*60}")
    print(" SUMMARY TABLE")
    print(f"{'='*60}")
    print(summary[["model", "accuracy", "macro_f1", "micro_f1"]].to_string(index=False))

  # Kappa matrix
  if len(all_results) >= 2:
    kappa = compute_kappa_matrix(all_results)
    if not dry_run:
      kappa.to_csv(os.path.join(args.output_dir, "kappa_matrix.csv"))
    print(f"\n{'='*60}")
    print(" COHEN'S KAPPA MATRIX")
    print(f"{'='*60}")
    print(kappa.to_string())


if __name__ == "__main__":
  main()
