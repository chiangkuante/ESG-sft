# ESG Risk Classification Prompt for LLM Benchmarking

## System Prompt

```
You are an ESG (Environmental, Social, and Governance) risk classification expert specialized in analyzing U.S. 10-K filings. Your task is to classify paragraphs from Item 1A (Risk Factors) sections into exactly one of the following 9 categories.

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
```

## User Prompt Template

```
Classify the following paragraph from a 10-K filing into one of the 9 ESG categories based on the definitions and examples below.

=== CATEGORY DEFINITIONS ===

[ENVIRONMENTAL]

1. Climate Change
Discussions about carbon emissions or climate change, including initiatives to increase carbon efficiency, environmental technologies, renewable energy, and the development or refurbishment of buildings with leading ecological design features.
Examples:
- Commitment to securing 100% of purchased electricity from renewable sources, reducing operational carbon footprint by 30%.
- Acquiring interest in a solar asset developer that combines clean electricity generation with carbon sequestration.
- Converting plants from coal to natural gas or steam.

2. Natural Capital
Discussions about water stress, biodiversity, land use, and raw materials sourcing. Includes efficient water processes, water recycling, alternative water sources, programs to protect biodiversity, address community land-use concerns, and policies to source materials with lower environmental impact.
Examples:
- Programs tackling the global water crisis, providing access to safe water in the developing world.
- Biodiversity action plans that maintain flora and fauna, adjusting maintenance work after finding protected species.
- Using paper certified from sustainably managed forests; remediating soil as part of redevelopment projects.

3. Pollution & Waste
Discussions about toxic emissions, packaging materials, and electronic waste. Includes pollution, contamination, emission of toxic substances, wastewater, product packaging content, end-of-life recycling or disposal, and removal of end-of-life electronic products.
Examples:
- Programs to reduce operational spills and clean up areas affected by oil spills.
- Reducing packaging through lightweighting and packaging reduction initiatives.
- Streamlined waste services for tenants; ensuring products go back to customers rather than landfills.

[SOCIAL]

4. Human Capital
Discussions about labor management, health and safety, human capital development and training, and supply chain labor standards. Includes workforce management, workflow disruptions, labor productivity, employee diversity, pay equality, H&S programs, training, development programs, employee engagement, and supply chain labor issues.
Examples:
- Formal channels guiding decisions on flextime, part-time, compressed work weeks, job sharing, and remote work.
- Technology systems tracking driver behaviors to increase accountability and reduce speeding.
- Targeted leadership development programs for colleagues at specific career stages.
- Efforts to source raw materials aligned with fair and equal treatment of workers.

5. Product Liability
Discussions about product safety and quality, privacy and data security, chemical safety, consumer financial protection, and health and demographic risk. Includes product recalls, product quality concerns, data security breaches, data privacy policies, use of chemicals of concern, transparency of financial products, and public health trends.
Examples:
- Products designed and tested to comply with all applicable safety regulations.
- Comprehensive cybersecurity programs to predict, protect, detect, respond to, and recover from cyberattacks.
- Phasing out chemicals suspected of causing medical reactions or harm.
- Resources to help customers understand, build, and improve credit.

6. Community Relations
Discussions about a firm's interaction with its local communities, including access to communications, finance, and healthcare. Includes opportunities in underserved markets, developing countries, underserved populations, and relevant philanthropic efforts.
Examples:
- Increasing local sourcing of barley from 70% to 86%, targeting 100% local sourcing.
- Commitments to invest in supporting smallholder farmers and enterprise development.
- Improving access for local people to healthcare and disease treatments.
- Employee volunteering for disaster relief, hunger, medical research, home building, or youth mentoring.

[GOVERNANCE]

7. Corporate Governance
Discussions about shareholders and ownership, board of directors, executive pay, and internal controls. Includes ownership structure, board independence, board skills and diversity, CEO and executive pay practices, performance incentives, audit matters, and internal audit.
Examples:
- Disclosure of beneficial owners of more than 5% of common stock.
- Board composed entirely of independent directors with diversity in gender, age, race, and professional experience.
- Linking executive remuneration to short-term targets to reduce Net Carbon Footprint.
- Ratification of appointment of independent registered public accounting firm.

8. Business Ethics & Values
Discussions about ethical components such as a firm's values and controversies. Includes ethical conduct of business, fraud, corruption, bribery, fiduciary responsibilities, conflicts of interest, misrepresentation, bias, negligence, political contributions, negative accounting events.
Examples:
- Guiding principles that every employee pledges to embrace and work by each day.
- Dedication to uncompromising integrity in all business relations.
- Settlements involving civil monetary penalties without admitting or denying claims.
- Challenges to transactions under antitrust laws by private parties or state attorneys general.

[NON-ESG]

9. Non-ESG
The paragraph does not primarily discuss any of the above ESG topics. Typical Non-ESG content includes: general financial performance, revenue/earnings discussions, market competition, product/service descriptions, operational logistics, legal boilerplate, accounting policies, and general business strategy unrelated to ESG.

=== IMPORTANT RULES ===
- Choose the SINGLE most relevant category. If a paragraph touches multiple ESG topics, select the PRIMARY one.
- Only classify as an ESG category if the paragraph PRIMARILY discusses that topic. Incidental mentions do not qualify.
- If uncertain between an ESG category and Non-ESG, lean toward Non-ESG.
- Respond with ONLY the category name, nothing else.

=== PARAGRAPH TO CLASSIFY ===
{text}
```

---

## Usage Notes

### API Call Parameters (for all LLMs)
- **Temperature**: 0 (deterministic output)
- **Max tokens**: 20 (category names are short)
- **Top-p**: 1.0

### Post-processing Rules
After receiving the LLM response, apply the following normalization:
1. Strip whitespace and newlines
2. Remove any numbering prefix (e.g., "1. ", "3) ")
3. Remove quotation marks or markdown formatting
4. Map common variations:
 - "Pollution and Waste" → "Pollution & Waste"
 - "Business Ethics and Values" → "Business Ethics & Values"
 - "Business Ethics" → "Business Ethics & Values"
 - "Non ESG" / "NonESG" / "N/A" → "Non-ESG"
 - "Community" → "Community Relations"
 - "Governance" / "Corp Governance" → "Corporate Governance"
5. If the response cannot be mapped to any valid category, mark as "Unclassified"

### Supported LLMs (from the paper)
| Model | API Provider | Model String |
|-------|-------------|-------------|
| ChatGPT 4o | OpenAI | `gpt-4o` |
| Claude 3.5 Sonnet | Anthropic | `claude-3-5-sonnet-20241022` |
| Gemini 2.5 Pro | Google | `gemini-2.5-pro` |
| Grok 3 | xAI | `grok-3` |
| DeepSeek 3.1 | DeepSeek | `deepseek-chat` |
