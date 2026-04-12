#!/usr/bin/env python
# coding: utf-8

# To run this, press "*Runtime*" and press "*Run all*" on a **free** Tesla T4 Google Colab instance!
# <div class="align-center">
# <a href="https://unsloth.ai/"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
# <a href="https://discord.gg/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/Discord button.png" width="145"></a>
# <a href="https://unsloth.ai/docs/"><img src="https://github.com/unslothai/unsloth/blob/main/images/documentation%20green%20button.png?raw=true" width="125"></a> Join Discord if you need help + ⭐ <i>Star us on <a href="https://github.com/unslothai/unsloth">Github</a> </i> ⭐
# </div>
# 
# To install Unsloth on your local device, follow [our guide](https://unsloth.ai/docs/get-started/install). This notebook is licensed [LGPL-3.0](https://github.com/unslothai/notebooks?tab=LGPL-3.0-1-ov-file#readme).
# 
# You will learn how to do [data prep](#Data), how to [train](#Train), how to [run the model](#Inference), & how to save it

# ### News

# Train MoEs - DeepSeek, GLM, Qwen and gpt-oss 12x faster with 35% less VRAM. [Blog](https://unsloth.ai/docs/new/faster-moe)
# 
# You can now train embedding models 1.8-3.3x faster with 20% less VRAM. [Blog](https://unsloth.ai/docs/new/embedding-finetuning)
# 
# Ultra Long-Context Reinforcement Learning is here with 7x more context windows! [Blog](https://unsloth.ai/docs/new/grpo-long-context)
# 
# 3x faster LLM training with 30% less VRAM and 500K context. [3x faster](https://unsloth.ai/docs/new/3x-faster-training-packing) • [500K Context](https://unsloth.ai/docs/new/500k-context-length-fine-tuning)
# 
# New in Reinforcement Learning: [FP8 RL](https://unsloth.ai/docs/new/fp8-reinforcement-learning) • [Vision RL](https://unsloth.ai/docs/new/vision-reinforcement-learning-vlm-rl) • [Standby](https://unsloth.ai/docs/basics/memory-efficient-rl) • [gpt-oss RL](https://unsloth.ai/docs/new/gpt-oss-reinforcement-learning)
# 
# Visit our docs for all our [model uploads](https://unsloth.ai/docs/get-started/unsloth-model-catalog) and [notebooks](https://unsloth.ai/docs/get-started/unsloth-notebooks).

# ### Installation

# In[1]:


# %%capture
# import os, re
# if "COLAB_" not in "".join(os.environ.keys()):
#   !pip install unsloth # Do this in local & cloud setups
# else:
#   import torch; v = re.match(r'[\d]{1,}\.[\d]{1,}', str(torch.__version__)).group(0)
#   xformers = 'xformers==' + {'2.10':'0.0.34','2.9':'0.0.33.post1','2.8':'0.0.32.post2'}.get(v, "0.0.34")
#   !pip install sentencepiece protobuf "datasets==4.3.0" "huggingface_hub>=0.34.0" hf_transfer
#   !pip install --no-deps unsloth_zoo bitsandbytes accelerate {xformers} peft trl triton unsloth
# !pip install transformers==5.3.0
# !pip install --no-deps trl==0.22.2


# ### Unsloth

# In[2]:


from unsloth import FastVisionModel # FastLanguageModel for LLMs
import torch

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
  "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit", # Llama 3.2 vision support
  "unsloth/Llama-3.2-11B-Vision-bnb-4bit",
  "unsloth/Llama-3.2-90B-Vision-Instruct-bnb-4bit", # Can fit in a 80GB card!
  "unsloth/Llama-3.2-90B-Vision-bnb-4bit",

  "unsloth/Pixtral-12B-2409-bnb-4bit",       # Pixtral fits in 16GB!
  "unsloth/Pixtral-12B-Base-2409-bnb-4bit",     # Pixtral base model

  "unsloth/Qwen2-VL-2B-Instruct-bnb-4bit",     # Qwen2 VL support
  "unsloth/Qwen2-VL-7B-Instruct-bnb-4bit",
  "unsloth/Qwen2-VL-72B-Instruct-bnb-4bit",

  "unsloth/llava-v1.6-mistral-7b-hf-bnb-4bit",   # Any Llava variant works!
  "unsloth/llava-1.5-7b-hf-bnb-4bit",
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastVisionModel.from_pretrained(
  "unsloth/Qwen3.5-4B",
  load_in_4bit = False, # Use 4bit to reduce memory use. False for 16bit LoRA.
  use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
)


# We now add LoRA adapters for parameter efficient finetuning - this allows us to only efficiently train 1% of all parameters.
# 
# **[NEW]** We also support finetuning ONLY the vision part of the model, or ONLY the language part. Or you can select both! You can also select to finetune the attention or the MLP layers!

# In[3]:


model = FastVisionModel.get_peft_model(
  model,
  finetune_vision_layers   = False, # False if not finetuning vision layers
  finetune_language_layers  = True, # False if not finetuning language layers
  finetune_attention_modules = True, # False if not finetuning attention layers
  finetune_mlp_modules    = True, # False if not finetuning MLP layers

  r = 16,      # The larger, the higher the accuracy, but might overfit
  lora_alpha = 16, # Recommended alpha == r at least
  lora_dropout = 0,
  bias = "none",
  random_state = 3407,
  use_rslora = False, # We support rank stabilized LoRA
  loftq_config = None, # And LoftQ
  # target_modules = "all-linear", # Optional now! Can specify a list if needed
)


# <a name="Data"></a>
# ### Data Prep
# We'll be using a sampled dataset of handwritten maths formulas. The goal is to convert these images into a computer readable form - ie in LaTeX form, so we can render it. This can be very useful for complex formulas.
# 
# You can access the dataset [here](https://huggingface.co/datasets/unsloth/LaTeX_OCR). The full dataset is [here](https://huggingface.co/datasets/linxy/LaTeX_OCR).

# In[ ]:


from datasets import load_dataset
# dataset = load_dataset("unsloth/LaTeX_OCR", split = "train")
dataset = load_dataset("json", data_files="../../data/processed/step6_sft/train_sft_xml.json", split="train")


# Let's take an overview look at the dataset. We shall see what the 3rd image is, and what caption it had.

# In[ ]:


dataset


# In[ ]:


# dataset[2]["image"]


# In[ ]:


dataset[2]["conversations"]


# We can also render the LaTeX in the browser directly!

# In[ ]:


# from IPython.display import display, Math, Latex

# latex = dataset[2]["text"]
# display(Math(latex))


# To format the dataset, all vision finetuning tasks should be formatted as follows:
# 
# ```python
# [
# { "role": "user",
#  "content": [{"type": "text", "text": Q}, {"type": "image", "image": image} ]
# },
# { "role": "assistant",
#  "content": [{"type": "text", "text": A} ]
# },
# ]
# ```

# In[ ]:


def convert_to_conversation(sample):
  conversation = []
  for message in sample["conversations"]:
    conversation.append({
      "role": message["role"],
      "content": [{"type": "text", "text": message["content"]}]
    })
  return { "messages" : conversation }
pass


# Let's convert the dataset into the "correct" format for finetuning:

# In[ ]:


converted_dataset = [convert_to_conversation(sample) for sample in dataset]


# We look at how the conversations are structured for the first example:

# In[ ]:


converted_dataset[0]


# Let's first see before we do any finetuning what the model outputs for the first example!

# In[ ]:


FastVisionModel.for_inference(model) # Enable for inference!

sample = dataset[2]
messages = []
for message in sample["conversations"][:-1]:
  messages.append({
    "role": message["role"],
    "content": [{"type": "text", "text": message["content"]}]
  })

input_text = tokenizer.apply_chat_template(messages, add_generation_prompt = True)
inputs = tokenizer(
  text=input_text,
  add_special_tokens = False,
  return_tensors = "pt",
).to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer, skip_prompt = True)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128,
          use_cache = True, temperature = 0.1, min_p = 0.1)


# <a name="Train"></a>
# ### Train the model
# Now let's train our model. We do 60 steps to speed things up, but you can set `num_train_epochs=1` for a full run, and turn off `max_steps=None`. We also support `DPOTrainer` and `GRPOTrainer` for reinforcement learning!
# 
# We use our new `UnslothVisionDataCollator` which will help in our vision finetuning setup.

# In[ ]:


from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

FastVisionModel.for_training(model) # Enable for training!

trainer = SFTTrainer(
  model = model,
  tokenizer = tokenizer,
  data_collator = UnslothVisionDataCollator(model, tokenizer), # Must use!
  train_dataset = converted_dataset,
  args = SFTConfig(
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4,
    warmup_steps = 5,
    # max_steps = 200,
    num_train_epochs = 1, # Set this instead of max_steps for full training runs
    learning_rate = 2e-4,
    logging_steps = 1,
    optim = "adamw_8bit",
    weight_decay = 0.001,
    lr_scheduler_type = "linear",
    seed = 3407,
    output_dir = "outputs",
    report_to = "none",   # For Weights and Biases

    # You MUST put the below items for vision finetuning:
    remove_unused_columns = False,
    dataset_text_field = "",
    dataset_kwargs = {"skip_prepare_dataset": True},
    max_length = 2048,
  ),
)


# In[ ]:


# @title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")


# In[ ]:


trainer_stats = trainer.train()


# In[ ]:


# @title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(
  f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
)
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


# <a name="Inference"></a>
# ### Inference
# Let's run the model! You can change the instruction and input - leave the output blank!
# 
# We use `min_p = 0.1` and `temperature = 1.5`. Read this [Tweet](https://x.com/menhguin/status/1826132708508213629) for more information on why.

# In[ ]:


FastVisionModel.for_inference(model) # Enable for inference!

sample = dataset[2]
messages = []
for message in sample["conversations"][:-1]:
  messages.append({
    "role": message["role"],
    "content": [{"type": "text", "text": message["content"]}]
  })

input_text = tokenizer.apply_chat_template(messages, add_generation_prompt = True)
inputs = tokenizer(
  text=input_text,
  add_special_tokens = False,
  return_tensors = "pt",
).to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer, skip_prompt = True)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128,
          use_cache = True, temperature = 0.1, min_p = 0.1)


# <a name="Evaluation"></a>
# ### Model Evaluation
# Let's evaluate the model on the validation dataset and calculate metrics including F1 score, accuracy, precision, and recall.

# In[ ]:


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


# In[ ]:


import json
import pandas as pd

TEST_PATH = "../../data/processed/step3_annotation/test_labeled.json"

with open(TEST_PATH, "r", encoding="utf-8") as f:
  test_data = json.load(f)

test_df = pd.DataFrame(test_data)

print("Test size:", len(test_df))
print(test_df[["paragraph_id", "combined_text", "label"]].head())


# In[ ]:


import re
import pandas as pd
from IPython.display import display
from tqdm import tqdm
from sklearn.metrics import (
  accuracy_score,
  precision_recall_fscore_support,
  classification_report,
  confusion_matrix,
)

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
INVALID_LABEL = "__INVALID__"

XML_SYSTEM_PROMPT = SYSTEM_PROMPT + "\n\nYou must respond in XML using exactly this format:\n<reasoning>your reasoning</reasoning>\n<label>one valid category</label>"
XML_USER_PROMPT_TEMPLATE = USER_PROMPT_TEMPLATE + "\n\nReturn your final answer in this exact XML format:\n<reasoning>brief reasoning</reasoning>\n<label>one valid category</label>"


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


def build_messages(paragraph_text):
  return [
    {
      "role": "system",
      "content": [{"type": "text", "text": XML_SYSTEM_PROMPT}],
    },
    {
      "role": "user",
      "content": [{"type": "text", "text": XML_USER_PROMPT_TEMPLATE.replace("{text}", paragraph_text)}],
    },
  ]


y_true_parseable = []
y_pred_parseable = []
y_true_all = []
y_pred_all = []
failed_cases = []

print(f"Evaluating on {len(test_df)} test samples...\n")

for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Evaluating"):
  paragraph_id = row.get("paragraph_id")
  paragraph_text = row["combined_text"]
  gt_label = canonicalize_label(row["label"])

  if gt_label is None:
    failed_cases.append({
      "paragraph_id": paragraph_id,
      "reason": "invalid_ground_truth",
      "ground_truth": row.get("label"),
    })
    continue

  y_true_all.append(gt_label)

  try:
    messages = build_messages(paragraph_text)

    inputs = tokenizer.apply_chat_template(
      messages,
      add_generation_prompt=True,
      tokenize=True,
      return_tensors="pt",
      return_dict=True,
    )
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    outputs = model.generate(
      **inputs,
      max_new_tokens=512,
      do_sample=False,
      temperature=None,
      top_p=None,
      use_cache=True,
      pad_token_id=tokenizer.eos_token_id,
    )

    input_len = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_len:]
    decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    pred_label = extract_label(decoded)

    if len(y_pred_all) < 5:
      print("RAW OUTPUT:", repr(decoded))
      print("PARSED PRED:", pred_label)
      print("GT:", gt_label)
      print("-" * 80)

    if pred_label is None:
      y_pred_all.append(INVALID_LABEL)
      failed_cases.append({
        "paragraph_id": paragraph_id,
        "reason": "prediction_parse_failed",
        "ground_truth": gt_label,
        "raw_output": decoded,
      })
      continue

    y_true_parseable.append(gt_label)
    y_pred_parseable.append(pred_label)
    y_pred_all.append(pred_label)

  except Exception as exc:
    y_pred_all.append(INVALID_LABEL)
    failed_cases.append({
      "paragraph_id": paragraph_id,
      "reason": "exception",
      "ground_truth": gt_label,
      "error": str(exc),
    })

total_eval = len(y_true_all)
num_parseable = len(y_true_parseable)
num_failed = len(failed_cases)
parse_fail_rate = num_failed / total_eval if total_eval else 0.0

print("\n=== Evaluation Summary ===")
print(f"Total evaluated samples: {total_eval}")
print(f"Parseable predictions:  {num_parseable}")
print(f"Failed predictions:   {num_failed}")
print(f"Parse failure rate:   {parse_fail_rate:.4f}")

print("\n" + "=" * 70)
print("PARSEABLE-ONLY METRICS")
print("=" * 70)

if y_true_parseable and set(y_true_parseable) & set(LABELS):
  acc_p = accuracy_score(y_true_parseable, y_pred_parseable)
  macro_p_p, macro_r_p, macro_f1_p, _ = precision_recall_fscore_support(
    y_true_parseable,
    y_pred_parseable,
    labels=LABELS,
    average="macro",
    zero_division=0,
  )
  weighted_p_p, weighted_r_p, weighted_f1_p, _ = precision_recall_fscore_support(
    y_true_parseable,
    y_pred_parseable,
    labels=LABELS,
    average="weighted",
    zero_division=0,
  )

  print(f"Accuracy:      {acc_p:.4f}")
  print(f"Macro Precision:   {macro_p_p:.4f}")
  print(f"Macro Recall:    {macro_r_p:.4f}")
  print(f"Macro F1:      {macro_f1_p:.4f}")
  print(f"Weighted Precision: {weighted_p_p:.4f}")
  print(f"Weighted Recall:   {weighted_r_p:.4f}")
  print(f"Weighted F1:     {weighted_f1_p:.4f}")

  print("\n=== Classification Report (Parseable-only) ===")
  print(classification_report(y_true_parseable, y_pred_parseable, labels=LABELS, zero_division=0))

  cm_parseable = confusion_matrix(y_true_parseable, y_pred_parseable, labels=LABELS)
  display(pd.DataFrame(cm_parseable, index=LABELS, columns=LABELS))
else:
  print("No parseable predictions available.")

print("\n" + "=" * 70)
print("END-TO-END METRICS (failed outputs counted as wrong)")
print("=" * 70)

if y_true_all and set(y_true_all) & set(LABELS):
  acc_all = accuracy_score(y_true_all, y_pred_all)
  macro_p_all, macro_r_all, macro_f1_all, _ = precision_recall_fscore_support(
    y_true_all,
    y_pred_all,
    labels=LABELS,
    average="macro",
    zero_division=0,
  )
  weighted_p_all, weighted_r_all, weighted_f1_all, _ = precision_recall_fscore_support(
    y_true_all,
    y_pred_all,
    labels=LABELS,
    average="weighted",
    zero_division=0,
  )

  print(f"Accuracy:      {acc_all:.4f}")
  print(f"Macro Precision:   {macro_p_all:.4f}")
  print(f"Macro Recall:    {macro_r_all:.4f}")
  print(f"Macro F1:      {macro_f1_all:.4f}")
  print(f"Weighted Precision: {weighted_p_all:.4f}")
  print(f"Weighted Recall:   {weighted_r_all:.4f}")
  print(f"Weighted F1:     {weighted_f1_all:.4f}")

  print("\n=== Classification Report (End-to-end) ===")
  print(classification_report(y_true_all, y_pred_all, labels=LABELS, zero_division=0))

  cm_all_labels = LABELS + [INVALID_LABEL]
  cm_all = confusion_matrix(y_true_all, y_pred_all, labels=cm_all_labels)
  print("\n=== Confusion Matrix (End-to-end) ===")
  print("Rows = ground truth, Columns = prediction")
  display(pd.DataFrame(cm_all, index=cm_all_labels, columns=cm_all_labels))

  invalid_pred_count = sum(pred == INVALID_LABEL for pred in y_pred_all)
  print(f"\nInvalid prediction count: {invalid_pred_count}")
else:
  print("No end-to-end samples available.")

if failed_cases:
  failed_df = pd.DataFrame(failed_cases)
  print("\n=== Failed Cases by Reason ===")
  display(failed_df["reason"].value_counts().rename_axis("reason").reset_index(name="count"))
  print("\n=== Failed Cases (head 20) ===")
  display(failed_df.head(20))


# <a name="Save"></a>
# ### Saving, loading finetuned models
# To save the final model as LoRA adapters, either use Hugging Face's `push_to_hub` for an online save or `save_pretrained` for a local save.
# 
# **[NOTE]** This ONLY saves the LoRA adapters, and not the full model. To save to 16bit or GGUF, scroll down!

# In[ ]:


model.save_pretrained("qwen_lora") # Local saving
tokenizer.save_pretrained("qwen_lora")
# model.push_to_hub("your_name/qwen_lora", token = "YOUR_HF_TOKEN") # Online saving
# tokenizer.push_to_hub("your_name/qwen_lora", token = "YOUR_HF_TOKEN") # Online saving


# Now if you want to load the LoRA adapters we just saved for inference, set `False` to `True`:

# In[ ]:


FastVisionModel.for_inference(model) # Enable for inference!

sample = dataset[2]
messages = []
for message in sample["conversations"][:-1]:
  messages.append({
    "role": message["role"],
    "content": [{"type": "text", "text": message["content"]}]
  })

input_text = tokenizer.apply_chat_template(messages, add_generation_prompt = True)
inputs = tokenizer(
  text=input_text,
  add_special_tokens = False,
  return_tensors = "pt",
).to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer, skip_prompt = True)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128,
          use_cache = True, temperature = 0.1, min_p = 0.1)


# ### Saving to float16 for VLLM
# 
# We also support saving to `float16` directly. Select `merged_16bit` for float16. Use `push_to_hub_merged` to upload to your Hugging Face account! You can go to https://huggingface.co/settings/tokens for your personal tokens. See [our docs](https://unsloth.ai/docs/basics/inference-and-deployment) for more deployment options.

# In[ ]:


# Select ONLY 1 to save! (Both not needed!)

# Save locally to 16bit
if False: model.save_pretrained_merged("unsloth_finetune", tokenizer,)

# To export and save to your Hugging Face account
if False: model.push_to_hub_merged("YOUR_USERNAME/unsloth_finetune", tokenizer, token = "YOUR_HF_TOKEN")


# ### GGUF / llama.cpp Conversion
# To save to `GGUF` / `llama.cpp`, we support it natively now! We clone `llama.cpp` and we default save it to `q8_0`. We allow all methods like `q4_k_m`. Use `save_pretrained_gguf` for local saving and `push_to_hub_gguf` for uploading to HF.
# 
# Some supported quant methods (full list on our [docs page](https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf)):
# * `q8_0` - Fast conversion. High resource use, but generally acceptable.
# * `q4_k_m` - Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q4_K.
# * `q5_k_m` - Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q5_K.
# 
# [**NEW**] To finetune and auto export to Ollama, try our [Ollama notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3_(8B)-Ollama.ipynb)

# In[ ]:


# Save to 8bit Q8_0
if False: model.save_pretrained_gguf("qwen_finetune", tokenizer,)
# Remember to go to https://huggingface.co/settings/tokens for a token!
# And change hf to your username!
if False: model.push_to_hub_gguf("HF_USERNAME/qwen_finetune", tokenizer, token = "YOUR_HF_TOKEN")

# Save to 16bit GGUF
if False: model.save_pretrained_gguf("qwen_finetune", tokenizer, quantization_method = "f16")
if False: model.push_to_hub_gguf("HF_USERNAME/qwen_finetune", tokenizer, quantization_method = "f16", token = "YOUR_HF_TOKEN")

# Save to q4_k_m GGUF
if False: model.save_pretrained_gguf("qwen_finetune", tokenizer, quantization_method = "q4_k_m")
if False: model.push_to_hub_gguf("HF_USERNAME/qwen_finetune", tokenizer, quantization_method = "q4_k_m", token = "YOUR_HF_TOKEN")

# Save to multiple GGUF options - much faster if you want multiple!
if False:
  model.push_to_hub_gguf(
    "HF_USERNAME/qwen_finetune", # Change hf to your username!
    tokenizer,
    quantization_method = ["q4_k_m", "q8_0", "q5_k_m",],
    token = "YOUR_HF_TOKEN",
  )


# And we're done! If you have any questions on Unsloth, we have a [Discord](https://discord.gg/unsloth) channel! If you find any bugs or want to keep updated with the latest LLM stuff, or need help, join projects etc, feel free to join our Discord!
# 
# Some other resources:
# 1. Looking to use Unsloth locally? Read our [Installation Guide](https://unsloth.ai/docs/get-started/install) for details on installing Unsloth on Windows, Docker, AMD, Intel GPUs.
# 2. Learn how to do Reinforcement Learning with our [RL Guide and notebooks](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide).
# 3. Read our guides and notebooks for [Text-to-speech (TTS)](https://unsloth.ai/docs/basics/text-to-speech-tts-fine-tuning) and [vision](https://unsloth.ai/docs/basics/vision-fine-tuning) model support.
# 4. Explore our [LLM Tutorials Directory](https://unsloth.ai/docs/models/tutorials-how-to-fine-tune-and-run-llms) to find dedicated guides for each model.
# 5. Need help with Inference? Read our [Inference & Deployment page](https://unsloth.ai/docs/basics/inference-and-deployment) for details on using vLLM, llama.cpp, Ollama etc.
# 
# <div class="align-center">
#  <a href="https://unsloth.ai"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
#  <a href="https://discord.gg/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/Discord.png" width="145"></a>
#  <a href="https://unsloth.ai/docs/"><img src="https://github.com/unslothai/unsloth/blob/main/images/documentation%20green%20button.png?raw=true" width="125"></a>
# 
#  Join Discord if you need help + ⭐️ <i>Star us on <a href="https://github.com/unslothai/unsloth">Github</a> </i> ⭐️
# 
#  <b>This notebook and all Unsloth notebooks are licensed [LGPL-3.0](https://github.com/unslothai/notebooks?tab=LGPL-3.0-1-ov-file#readme)</b>
# </div>
