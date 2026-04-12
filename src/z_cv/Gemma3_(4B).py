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
#   !uv add unsloth # Do this in local & cloud setups
# else:
#   import torch; v = re.match(r'[\d]{1,}\.[\d]{1,}', str(torch.__version__)).group(0)
#   xformers = 'xformers==' + {'2.10':'0.0.34','2.9':'0.0.33.post1','2.8':'0.0.32.post2'}.get(v, "0.0.34")
#   !uv add sentencepiece protobuf "datasets==4.3.0" "huggingface_hub>=0.34.0" hf_transfer
#   !uv add unsloth_zoo bitsandbytes accelerate {xformers} peft trl triton unsloth
# !uv add transformers==4.56.2
# !uv add trl==0.22.2


# ### Unsloth
# 
# `FastModel` supports loading nearly any model now! This includes Vision and Text models!

# In[2]:


from unsloth import FastModel
import torch

fourbit_models = [
  # 4bit dynamic quants for superior accuracy and low memory use
  "unsloth/gemma-3-1b-it-unsloth-bnb-4bit",
  "unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
  "unsloth/gemma-3-12b-it-unsloth-bnb-4bit",
  "unsloth/gemma-3-27b-it-unsloth-bnb-4bit",

  # Other popular models!
  "unsloth/Llama-3.1-8B",
  "unsloth/Llama-3.2-3B",
  "unsloth/Llama-3.3-70B",
  "unsloth/mistral-7b-instruct-v0.3",
  "unsloth/Phi-4",
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastModel.from_pretrained(
  model_name = "unsloth/gemma-3-4b-it",
  max_seq_length = 2048, # Choose any for long context!
  load_in_4bit = True, # 4 bit quantization to reduce memory
  load_in_8bit = False, # [NEW!] A bit more accurate, uses 2x memory
  full_finetuning = False, # [NEW!] We have full finetuning now!
  # token = "YOUR_HF_TOKEN", # HF Token for gated models
)


# We now add LoRA adapters so we only need to update a small amount of parameters!

# In[3]:


model = FastModel.get_peft_model(
  model,
  finetune_vision_layers   = False, # Turn off for just text!
  finetune_language_layers  = True, # Should leave on!
  finetune_attention_modules = True, # Attention good for GRPO
  finetune_mlp_modules    = True, # Should leave on always!

  r = 16,      # Larger = higher accuracy, but might overfit
  lora_alpha = 16, # Recommended alpha == r at least
  lora_dropout = 0,
  bias = "none",
  random_state = 3407,
)


# <a name="Data"></a>
# ### Data Prep
# We now use the `Gemma-3` format for conversation style finetunes. We use [Maxime Labonne's FineTome-100k](https://huggingface.co/datasets/mlabonne/FineTome-100k) dataset in ShareGPT style. Gemma-3 renders multi turn conversations like below:
# 
# ```
# <bos><start_of_turn>user
# Hello!<end_of_turn>
# <start_of_turn>model
# Hey there!<end_of_turn>
# ```
# 
# We use our `get_chat_template` function to get the correct chat template. We support `zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, phi3, llama3, phi4, qwen2.5, gemma3` and more.

# In[4]:


from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(
  tokenizer,
  chat_template = "gemma-3",
)


# In[5]:


from datasets import load_dataset
# dataset = load_dataset("mlabonne/FineTome-100k", split = "train")
dataset_train = load_dataset("json", data_files="../../data/processed/step6_sft/train_sft_xml.json", split="train")
# dataset_val = load_dataset("json", data_files="../data/fine-tuning_data/test.json", split="train")


# We now use `standardize_data_formats` to try converting datasets to the correct format for finetuning purposes!

# In[6]:


from unsloth.chat_templates import standardize_data_formats
dataset = standardize_data_formats(dataset_train)
# dataset_val = standardize_data_formats(dataset_val)


# Let's see how row 100 looks like!

# In[7]:


dataset[10]
# dataset_val[10]


# We now have to apply the chat template for `Gemma-3` onto the conversations, and save it to `text`. We remove the `<bos>` token using removeprefix(`'<bos>'`) since we're finetuning. The Processor will add this token before training and the model expects only one.

# In[8]:


def formatting_prompts_func(examples):
  convos = examples["conversations"]
  texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False).removeprefix('<bos>') for convo in convos]
  return { "text" : texts, }

dataset = dataset.map(formatting_prompts_func, batched = True)


# Let's see how the chat template did! Notice there is no `<bos>` token as the processor tokenizer will be adding one.

# In[9]:


dataset[100]["text"]


# <a name="Train"></a>
# ### Train the model
# Now let's train our model. We do 60 steps to speed things up, but you can set `num_train_epochs=1` for a full run, and turn off `max_steps=None`.

# In[10]:


from trl import SFTTrainer, SFTConfig
trainer = SFTTrainer(
  model = model,
  tokenizer = tokenizer,
  train_dataset = dataset,
  eval_dataset = None, # Can set up evaluation!
  args = SFTConfig(
    dataset_text_field = "text",
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4, # Use GA to mimic batch size!
    warmup_steps = 5,
    num_train_epochs = 1, # Set this for 1 full training run.
    # max_steps =200,
    learning_rate = 2e-4, # Reduce to 2e-5 for long training runs
    logging_steps = 1,
    optim = "adamw_8bit",
    weight_decay = 0.001,
    lr_scheduler_type = "linear",
    seed = 3407,
    report_to = "none", # Use TrackIO/WandB etc
  ),
)


# We also use Unsloth's `train_on_completions` method to only train on the assistant outputs and ignore the loss on the user's inputs. This helps increase accuracy of finetunes!

# In[11]:


from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
  trainer,
  instruction_part = "<start_of_turn>user\n",
  response_part = "<start_of_turn>model\n",
)


# Let's verify masking the instruction part is done! Let's print the 100th row again. Notice how the sample only has a single `<bos>` as expected!

# In[12]:


tokenizer.decode(trainer.train_dataset[100]["input_ids"])


# Now let's print the masked out example - you should see only the answer is present:

# In[13]:


tokenizer.decode([tokenizer.pad_token_id if x == -100 else x for x in trainer.train_dataset[100]["labels"]]).replace(tokenizer.pad_token, " ")


# In[14]:


# @title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")


# Let's train the model! To resume a training run, set `trainer.train(resume_from_checkpoint = True)`

# In[15]:


trainer_stats = trainer.train()


# In[16]:


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
# Let's run the model via Unsloth native inference! According to the `Gemma-3` team, the recommended settings for inference are `temperature = 1.0, top_p = 0.95, top_k = 64`

# In[17]:


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


# In[18]:


# from unsloth.chat_templates import get_chat_template
# tokenizer = get_chat_template(
#   tokenizer,
#   chat_template = "gemma-3",
# )

# text = "Our results can fluctuate due to the effects of weather, natural disasters and seasonality.\n\nOur results of operations are impacted by severe weather, natural disasters and seasonality. Severe weather conditions and natural disasters (or other environmental events) can significantly disrupt service and create air traffic control problems. These events decrease revenue and can also increase costs. In addition, increases in the frequency, severity or duration of thunderstorms, hurricanes, typhoons or other severe weather events, including from changes in the global climate, could result in increases in delays and cancellations, turbulence-related injuries and fuel consumption to avoid such weather, any of which could result in loss of revenue and higher costs. In addition, demand for air travel is typically higher in the June and September quarters, particularly in our international markets, because there is more vacation travel during these periods than during the remainder of the year. The seasonal shifting of demand causes our financial results to vary on a seasonal basis. Because of fluctuations in our results from weather, natural disasters and seasonality, operating results for a historical period are not necessarily indicative of operating results for a future period and operating results for an interim period are not necessarily indicative of operating results for an entire year."

# messages = [
#    {
#     "role": "system",
#     "content": [{
#      "type" : "text",
#      "text" : SYSTEM_PROMPT
#     }]
#    },
#    {
#     "role": "user",
#     "content": [{
#      "type" : "text",
#      "text" : USER_PROMPT_TEMPLATE.replace("{text}", text)
#     }]
#    },
#   ]
# inputs = tokenizer.apply_chat_template(
#   messages,
#   add_generation_prompt = True, # Must add for generation
#   tokenize = True,
#   return_tensors = "pt",
#   return_dict = True,
# )
# outputs = model.generate(
#   **inputs.to("cuda"),
#   max_new_tokens = 4096, # Increase for longer outputs!
#   # Recommended Gemma-3 settings!
#   temperature = 1, top_p = 0.95, top_k = 64,
# )
# tokenizer.batch_decode(outputs)


#  You can also use a `TextStreamer` for continuous inference - so you can see the generation token by token, instead of waiting the whole time!

# In[19]:


# # messages = [{
# #   "role": "user",
# #   "content": [{"type" : "text", "text" : "Why is the sky blue?",}]
# # }]


# messages = [
#    {
#     "role": "system",
#     "content": [{
#      "type" : "text",
#      "text" : ""
#     }]
#    },
#    {
#     "role": "user",
#     "content": [{
#      "type" : "text",
#      "text" : " "
#     }]
#    },
#   ]
# inputs = tokenizer.apply_chat_template(
#   messages,
#   add_generation_prompt = True, # Must add for generation
#   tokenize = True,
#   return_tensors = "pt",
#   return_dict = True,
# )

# from transformers import TextStreamer
# _ = model.generate(
#   **inputs.to("cuda"),
#   max_new_tokens = 512, # Increase for longer outputs!
#   # Recommended Gemma-3 settings!
#   temperature = 1.0, top_p = 0.95, top_k = 64,
#   streamer = TextStreamer(tokenizer, skip_prompt = True),
# )


# <a name="Evaluation"></a>
# ### Model Evaluation
# Let's evaluate the model on the validation dataset and calculate metrics including F1 score, accuracy, precision, and recall.

# In[20]:


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


# In[ ]:


# import re
# from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
# from tqdm import tqdm

# # Function to extract predicted label from model output
# def extract_label(text):
#   """Extract the predicted label (0 or 1) from the model's output."""
#   # Look for "Answer: 0" or "Answer: 1" pattern
#   match = re.search(r'Answer:\s*([01])', text)
#   if match:
#     return int(match.group(1))

#   # Fallback: look for last occurrence of 0 or 1
#   matches = re.findall(r'\b([01])\b', text)
#   if matches:
#     return int(matches[-1])

#   return None

# # Function to extract ground truth label
# def extract_ground_truth(conversation):
#   """Extract the ground truth label from the assistant's response."""
#   for msg in conversation:
#     if msg['role'] == 'assistant':
#       return extract_label(msg['content'])
#   return None

# # Function to manually format prompt in Gemma-3 format
# def format_gemma3_prompt(conversation):
#   """Manually format conversation to Gemma-3 chat format."""
#   prompt_parts = []
#   for msg in conversation:
#     if msg['role'] == 'system':
#       # System message gets added to user message in Gemma-3
#       prompt_parts.append(f"<start_of_turn>user\n{msg['content']}\n\n")
#     elif msg['role'] == 'user':
#       # If there was a system message, append to it, otherwise start new
#       if prompt_parts and '<start_of_turn>user' in prompt_parts[-1]:
#         prompt_parts[-1] = prompt_parts[-1].rstrip() + msg['content'] + "<end_of_turn>\n"
#       else:
#         prompt_parts.append(f"<start_of_turn>user\n{msg['content']}<end_of_turn>\n")

#   # Add generation prompt
#   prompt_parts.append("<start_of_turn>model\n")
#   return '<bos>' + ''.join(prompt_parts)

# # Evaluate on validation set
# y_true = []
# y_pred = []
# failed_predictions = 0

# print(f"Evaluating on {len(dataset_val)} validation samples...\n")

# for idx in tqdm(range(len(dataset_val)), desc="Evaluating"):
#   sample = dataset_val[idx]
#   conversation = sample['conversations']

#   # Get ground truth
#   gt_label = extract_ground_truth(conversation)
#   if gt_label is None:
#     failed_predictions += 1
#     continue

#   try:
#     # Manually format the prompt
#     prompt = format_gemma3_prompt(conversation)

#     # Tokenize
#     inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=True)

#     # Generate prediction
#     outputs = model.generate(
#       **inputs.to("cuda"),
#       max_new_tokens=512,
#       temperature=1.0,
#       top_p=0.95,
#       top_k=64,
#     )

#     # Decode and extract prediction
#     decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
#     pred_label = extract_label(decoded)

#     if pred_label is None:
#       failed_predictions += 1
#       continue

#     y_true.append(gt_label)
#     y_pred.append(pred_label)

#   except Exception as e:
#     print(f"\nError processing sample {idx}: {e}")
#     failed_predictions += 1
#     continue

# print(f"\n{'='*60}")
# print("EVALUATION RESULTS")
# print(f"{'='*60}")
# print(f"Total samples: {len(dataset_val)}")
# print(f"Successfully evaluated: {len(y_true)}")
# print(f"Failed predictions: {failed_predictions}")
# print(f"{'='*60}\n")

# # Calculate metrics
# if len(y_true) > 0:
#   accuracy = accuracy_score(y_true, y_pred)
#   precision = precision_score(y_true, y_pred, zero_division=0)
#   recall = recall_score(y_true, y_pred, zero_division=0)
#   f1 = f1_score(y_true, y_pred, zero_division=0)

#   print("Overall Metrics:")
#   print(f" Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
#   print(f" Precision: {precision:.4f} ({precision*100:.2f}%)")
#   print(f" Recall:  {recall:.4f} ({recall*100:.2f}%)")
#   print(f" F1 Score: {f1:.4f} ({f1*100:.2f}%)\n")

#   # Confusion matrix
#   cm = confusion_matrix(y_true, y_pred)
#   print("Confusion Matrix:")
#   print("        Predicted")
#   print("        0   1")
#   print(f"Actual 0  [{cm[0][0]:4d} {cm[0][1]:4d}]")
#   print(f"Actual 1  [{cm[1][0]:4d} {cm[1][1]:4d}]\n")

#   # Detailed classification report
#   print("Detailed Classification Report:")
#   print(classification_report(y_true, y_pred, target_names=['Class 0', 'Class 1'], zero_division=0))
# else:
#   print("No successful predictions to evaluate!")


# <a name="Save"></a>
# ### Saving, loading finetuned models
# To save the final model as LoRA adapters, either use Hugging Face's `push_to_hub` for an online save or `save_pretrained` for a local save.
# 
# **[NOTE]** This ONLY saves the LoRA adapters, and not the full model. To save to 16bit or GGUF, scroll down!

# In[25]:


model.save_pretrained("gemma_3_lora") # Local saving
tokenizer.save_pretrained("gemma_3_lora")
# model.push_to_hub("HF_ACCOUNT/gemma_3_lora", token = "YOUR_HF_TOKEN") # Online saving
# tokenizer.push_to_hub("HF_ACCOUNT/gemma_3_lora", token = "YOUR_HF_TOKEN") # Online saving


# Now if you want to load the LoRA adapters we just saved for inference, set `False` to `True`:

# In[26]:


if True:
  from unsloth import FastModel
  model, tokenizer = FastModel.from_pretrained(
    model_name = "gemma_3_lora", # YOUR MODEL YOU USED FOR TRAINING
    max_seq_length = 2048,
    load_in_4bit = True,
  )

messages = [{
  "role": "user",
  "content": [{"type" : "text", "text" : "What is Gemma-3?",}]
}]
inputs = tokenizer.apply_chat_template(
  messages,
  add_generation_prompt = True, # Must add for generation
  tokenize = True,
  return_tensors = "pt",
  return_dict = True,
)

from transformers import TextStreamer
_ = model.generate(
  **inputs.to("cuda"),
  max_new_tokens = 64, # Increase for longer outputs!
  # Recommended Gemma-3 settings!
  temperature = 1.0, top_p = 0.95, top_k = 64,
  streamer = TextStreamer(tokenizer, skip_prompt = True),
)


# ### Saving to float16 for VLLM
# 
# We also support saving to `float16` directly for deployment! We save it in the folder `gemma-3-finetune`. Set `if False` to `if True` to let it run!

# In[ ]:


if False: # Change to True to save finetune!
  model.save_pretrained_merged("gemma-3-finetune", tokenizer)


# If you want to upload / push to your Hugging Face account, set `if False` to `if True` and add your Hugging Face token and upload location!

# In[ ]:


if False: # Change to True to upload finetune
  model.push_to_hub_merged(
    "HF_ACCOUNT/gemma-3-finetune", tokenizer,
    token = "YOUR_HF_TOKEN"
  )


# ### GGUF / llama.cpp Conversion
# To save to `GGUF` / `llama.cpp`, we support it natively now for all models! For now, you can convert easily to `Q8_0, F16 or BF16` precision. `Q4_K_M` for 4bit will come later!

# In[ ]:


if False: # Change to True to save to GGUF
  model.save_pretrained_gguf(
    "gemma_3_finetune",
    tokenizer,
    quantization_method = "Q8_0", # For now only Q8_0, BF16, F16 supported
  )


# Likewise, if you want to instead push to GGUF to your Hugging Face account, set `if False` to `if True` and add your Hugging Face token and upload location!

# In[ ]:


if False: # Change to True to upload GGUF
  model.push_to_hub_gguf(
    "HF_ACCOUNT/gemma_3_finetune",
    tokenizer,
    quantization_method = "Q8_0", # Only Q8_0, BF16, F16 supported
    token = "YOUR_HF_TOKEN",
  )


# Now, use the `gemma-3-finetune.gguf` file or `gemma-3-finetune-Q4_K_M.gguf` file in llama.cpp.
# 
# And we're done! If you have any questions on Unsloth, we have a [Discord](https://discord.gg/unsloth) channel! If you find any bugs or want to keep updated with the latest LLM stuff, or need help, join projects etc, feel free to join our Discord!
# 
# Some other resources:
# 1. Train your own reasoning model - Llama GRPO notebook [Free Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb)
# 2. Saving finetunes to Ollama. [Free notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3_(8B)-Ollama.ipynb)
# 3. Llama 3.2 Vision finetuning - Radiography use case. [Free Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_(11B)-Vision.ipynb)
# 4. See notebooks for DPO, ORPO, Continued pretraining, conversational finetuning and more on our [documentation](https://unsloth.ai/docs/get-started/unsloth-notebooks)!
# 
# <div class="align-center">
#  <a href="https://unsloth.ai"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
#  <a href="https://discord.gg/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/Discord.png" width="145"></a>
#  <a href="https://unsloth.ai/docs/"><img src="https://github.com/unslothai/unsloth/blob/main/images/documentation%20green%20button.png?raw=true" width="125"></a>
# 
#  Join Discord if you need help + ⭐️ <i>Star us on <a href="https://github.com/unslothai/unsloth">Github</a> </i> ⭐️
# </div>
# 
#  This notebook and all Unsloth notebooks are licensed [LGPL-3.0](https://github.com/unslothai/notebooks?tab=LGPL-3.0-1-ov-file#readme).
