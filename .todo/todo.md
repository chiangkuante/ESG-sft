# ESG 10-K 分類實驗：完整流程

---



## 步驟 0：10-K Filing 下載與原始 HTML 保存

### 0.1 目的

本步驟的唯一目的是取得每份 10-K 的完整原始 HTML，作為後續所有處理的起點。不在此階段做任何文字抽取或段落切割，因為一旦先轉成純文字，粗體/斜體標題、表格邊界、清單結構等關鍵結構訊號就會消失，無法復原。保留原始 HTML 意味著後續調整切段規則時可以隨時重跑，不需要重新下載。

### 0.2 工具與流程

使用 `sec-downloader` 進行批次下載。流程分為兩步：

**第一步：取得 filing metadata。** 準備公司清單（S&P 500 的 ticker 或 CIK），對每家公司呼叫 `get_filing_metadatas(...)`，指定 form type 為 10-K，設定目標年份範圍（ 2021-2025）。每筆 metadata 包含 ticker、cik、company_name、filing_date、report_date、accession_number、form_type、primary_doc_url 等欄位。

**第二步：下載原始 HTML。** 對每筆 metadata，使用 `Downloader(...).download_filing(url=metadata.primary_doc_url)` 將 HTML 內容下載到記憶體，再寫入本地檔案系統。另一種方式是利用 `DownloadStorage` 包裝暫存檔案系統再讀入記憶體，適合大批次流程。

### 0.3 儲存結構

按 company/year/accession 建立資料夾，每份 filing 保留三份內容：

```
data/raw/
├── AAPL/
│ ├── 2021/
│ │ ├── 0000320193-20-000096/
│ │ │ ├── filing.html   # 原始 HTML
│ │ │ ├── metadata.json  # filing metadata
│ │ │ └── item1a.json   # 後續步驟 1 產出（此時尚未生成）
│ │ └── ...
│ └── ...
└── ...
```

### 0.4 規模估算

S&P 500 × 5 年（2021-2025）≈ 2,500 份 filing。每份 HTML 約 1-5 MB，總下載量約 10 GB。SEC EDGAR 有速率限制（每秒 10 次請求），批次下載需加入適當延遲。建議使用 User-Agent header 包含聯絡信箱(chiangkuante@gmail.com)以符合 SEC 的 fair access policy。

---

## 步驟 1：Heading-Aware Semantic Parsing 與段落切割

### 1.1 核心方法：sec-parser

不使用一般的 regex 切文字或 BeautifulSoup 簡單解析，而是使用 `sec-parser` 進行 heading-aware 的 semantic parsing。sec-parser 的設計目的是把 SEC EDGAR HTML 解析成對應視覺與資訊結構的 semantic tree，將原始 HTML tag 轉成更高層級的語義元素：TitleElement（標題）、TextElement（正文段落）、TableElement（表格）等。這不只是 HTML parser，而是把文件還原為接近人眼閱讀時看到的「文件結構」。

### 1.2 四層處理流程

**第一層：全文 Semantic Parsing**

將整份 filing HTML 透過 sec-parser 解析成 semantic elements 序列。每個元素會被自動辨識為 title、text、table 或其他類型，不需要自行發明標籤。

**第二層：定位 Item 1A 區段**

在 semantic elements 中找到 Item 1A (Risk Factors) 的起點和終點。採用 heading-aware + regex fallback 雙重判定：優先相信被辨識為 top-level title 的元素，其文字若匹配 "Item 1A" 或 "ITEM 1A" 即視為區段起點；終點以下一個 top-level item title（Item 1B 或 Item 2）為準。這種方式能應付絕大多數公司的格式差異。

**第三層：保留標題並組合段落**

這是與第一版最關鍵的差異。不把標題當成獨立樣本，而是把每個風險子標題暫存在 `current_heading` 變數中。當 parser 讀到後續的 TextElement 時，輸出文字組合為：

```
[current_heading]\n[paragraph_text]
```

每個段落樣本都以標題開頭，形成「標題＋正文」的完整語意單位。這樣分類模型看到的不只是孤立正文，而是有明確主題的完整風險因子。10-K 的風險因子結構通常是標題提供主題、正文提供細節，保留標題對 ESG 細類別的判別至關重要。

**第四層：段落邊界決定與合併/切分規則**

以 parser 輸出的 TextElement 為最小單位，加入合併與切分規則：

合併規則：若某段正文極短（< 100 字元），且與前一個 TextElement 同屬同一標題之下，則向前合併。這避免 HTML parser 因格式碎片化而產生的不完整小段落。

切分規則：若單一 TextElement 過長（> 2000 字元），在句子邊界上細切，切完後每個切片仍以 `current_heading` 作為開頭，確保每個段落都有標題上下文。

最終每個段落的目標長度：200-2000 字元，95% 在 512 token 以內（FinBERT 上限）。

### 1.3 表格與不相干文本移除

因為使用 semantic parsing，表格和雜訊清理可以比純文字方式更精準。清理分為五層：

**第一層：移除所有 TableElement。** sec-parser 已將 table 辨識為獨立語義元素，直接按元素類型移除，不需要猜測哪些 `<div>` 是表格。

**第二層：移除導覽與目錄性文字。** Table of Contents、頁首頁尾、章節索引、頁碼殘留。這些文字通常以零碎短文字塊出現，匹配目錄或頁碼樣式者（只包含 item 編號、頁碼、短標題、連續點點線）直接刪除。

**第三層：移除交叉引用與格式殘留。** "See Item 7…"、"refer to Note…"、"Table of Contents"、"continued"、"Page xx" 等。建立 stop-pattern 規則表，文字長度極短且匹配常見模式者標記為 non-content block 並刪除。

**第四層：移除數字密集的非敘述性文字。** 雖非 HTML 表格，但語意上接近表格殘留——數字比例過高、幾乎全是年份百分比、缺乏完整句子結構的文字。用統計規則過濾：數字與符號佔比 > 60%、平均詞長異常、連續大寫比例過高等。

**第五層：分離 Item 1A 的通用前言。** 部分公司在風險因子開始前有制式警語、safe harbor 聲明或前言敘述。這些位於第一個風險子標題之前、不屬於任何具體風險標題的通用前言，標記為 `section_intro` 另存，不進入主要分類語料。

### 1.4 輸出格式

每筆段落資料包含以下欄位：

```json
{
 "paragraph_id": "AAPL_2021_RISK_01",
 "ticker": "AAPL",
 "cik": "0000320193",
 "filing_date": "2021-10-30",
 "accession_number": "0000320193-20-000096",
 "risk_heading": "Global and regional economic conditions could materially adversely affect the Company.",
 "paragraph_text": "The Company has international operations with...",
 "combined_text": "Global and regional economic conditions could materially adversely affect the Company.\nThe Company has international operations with...",
 "char_count": 1234,
 "is_table_removed": true,
 "cleaning_flags": ["merged_short_fragment", "removed_page_number"]
}
```

`combined_text` 是後續標註、FinBERT 預測、LLM 分類統一使用的文字欄位，格式固定為「標題＋換行＋正文」。

### 1.5 品質驗證

對分割結果進行系統性驗證：統計每份文件的段落數量（預期 15-40 個風險因子）、段落長度分布（目標 200-2000 字元）、手動檢查 50 份跨不同公司和年份的文件確認無表格殘留。對異常文件（段落數 < 5 或 > 60）逐一檢查。統計表格移除率、短文字過濾率、合併率等清洗指標。

---

## 步驟 2：FinBERT 全量預測 + 隨機抽樣

### 2.1 FinBERT 全量預測

使用 `yiyanghkust/finbert-esg-9-categories` 對步驟 1 產出的全量段落池（約 160,000 段）進行預測。記錄每個段落的預測類別和信心分數。此步驟有兩個目的：一是為後續 Label Studio 提供預標註，加速人工標註；二是為步驟 4 的偽標籤生成提供篩選依據。

### 2.2 全量預測分布統計

統計 FinBERT 對全量段落的預測分布。預期結果：Non-ESG 佔 70-85%，各 ESG 類別佔 1-5%。識別哪些類別在全量資料中是低頻的，為步驟 4 提供依據。

### 2.3 隨機抽樣 500 筆

從全量段落池中進行分層隨機抽樣約 500 筆。分層條件為公司和年份，確保不會過度集中在少數公司或特定年份。每筆抽樣資料已帶有 FinBERT 的預測類別和信心分數。

### 2.4 資料切分

將 500 筆隨機抽樣分為兩部分：

**固定測試集（200 筆）**：完全保持原始隨機分布，永遠不參與任何訓練或超參數調整。這 200 筆反映了真實 10-K 資料中的類別比例（預期約 70-80% Non-ESG，每個 ESG 類別 2-5%）。這是最終報告模型在真實場景表現的唯一依據。

**原始訓練種子（300 筆）**：用於後續合併成完整訓練集。同樣保持原始隨機分布。

---

## 步驟 3：人工標註（含 FinBERT 預標註）

### 3.1 Label Studio 預標註設定

將 FinBERT 的預測結果作為預標註（pre-annotation）匯入 Label Studio。標註者打開每筆資料時，會看到 FinBERT 已經預選的類別，可以直接確認或修改。這能大幅加速標註——對於 FinBERT 預測正確的筆數（預估 70-80%），標註者只需快速確認；只有預測錯誤或不確定的筆數需要仔細判斷。

### 3.2 匯入格式

匯入 Label Studio 的 JSON 格式需要同時包含資料和預標註：

```json
[
 {
 "data": {
  "paragraph_id": "AAPL_2021_RISK_01",
  "combined_text": "Global and regional economic conditions...",
  "risk_heading": "Global and regional economic conditions could...",
  "risk_section": "Macroeconomic and Industry Risks",
  "ticker": "AAPL",
  "filing_date": "2021-10-30",
  "finbert_label": "Non-ESG",
  "finbert_confidence": 0.92
 },
 "predictions": [
  {
  "model_version": "finbert-esg-9-categories",
  "result": [
   {
   "from_name": "label",
   "to_name": "combined_text",
   "type": "choices",
   "value": {
    "choices": ["Non-ESG"]
   }
   }
  ],
  "score": 0.92
  }
 ]
 }
]
```

### 3.3 Labeling Config

```xml
<View>
 <Header value="ESG Risk Factor Classification" />
 <View style="display:flex; gap:8px; margin-bottom:8px;">
 <View style="background:#e8f4fd; padding:6px 12px; border-radius:4px;">
  <Text name="ticker_info" value="$ticker" />
 </View>
 <View style="background:#f0f0f0; padding:6px 12px; border-radius:4px;">
  <Text name="date_info" value="$filing_date" />
 </View>
 <View style="background:#fff3cd; padding:6px 12px; border-radius:4px;">
  <Text name="finbert_info" value="FinBERT: $finbert_label ($finbert_confidence)" />
 </View>
 </View>
 <Text name="combined_text" value="$combined_text" />
 <Choices name="label" toName="combined_text" choice="single">
 <Choice value="Climate Change" />
 <Choice value="Natural Capital" />
 <Choice value="Pollution &amp; Waste" />
 <Choice value="Human Capital" />
 <Choice value="Product Liability" />
 <Choice value="Community Relations" />
 <Choice value="Corporate Governance" />
 <Choice value="Business Ethics &amp; Values" />
 <Choice value="Non-ESG" />
 </Choices>
</View>
```

標註者會看到：頂部顯示 ticker、filing_date、FinBERT 預測結果和信心分數作為參考資訊。FinBERT 的預測會自動預選在選項中（來自 predictions），標註者可以直接確認或改選其他類別。

### 3.4 標註者設定

**標註者 p1（Project 16）：500 筆完整標註（單一標注者）**
匯入 500 筆（完整隨機集），含 FinBERT 預標註，由一位標注者完成全部標註。

標註產出檔案：`data/origin_data/10k_1A/project-16-at-2026-03-31-11-52-a8826fd3.csv`

**標註者 p2：500 筆完整標註（第二標注者）**
對同批 500 筆段落由第二位標注者獨立標註。

標註產出檔案：`data/origin_data/10k_1A/chiang_500.csv`

**額外平衡資料集：blance_500**
針對類別不平衡問題另行標註的平衡資料集。

資料檔案：`data/origin_data/10k_1A/blance_500.csv`

### 3.5 標註注意事項

預標註是輔助工具，不是答案。標註者必須依據自己對類別定義的理解做最終判斷。特別注意 FinBERT 信心分數低（< 0.5）的筆數——這些是模型不確定的案例，需要標註者更仔細地判斷。

標註完成後計算 FinBERT 預標註與最終人工標籤的一致率，以量化 FinBERT 的預測品質。FinBERT 與人工標註的一致率為 95.6%（500 筆中僅 22 筆不一致），其系統性弱點集中於將 ESG 類別誤判為 Non-ESG（22 筆中 16 筆屬此模式）。

### 3.6 多標註者配置（config.yaml）

透過 `config.yaml` 的 `step4_cv.annotator` 切換標註者，`step4_cv.balance` 控制平衡資料集：

```yaml
step4_cv:
  annotator: p1                     # p1 | p2
  annotators:
    p1:
      csv: data/origin_data/10k_1A/project-16-at-2026-03-31-11-52-a8826fd3.csv
    p2:
      csv: data/origin_data/10k_1A/chiang_500.csv
  balance:
    enabled: false
    include_in_cv: false
    csv: data/origin_data/10k_1A/blance_500.csv
```

所有中間產出、模型、結果以 annotator_name 為子目錄區分：

| annotator | balance.enabled | balance.include_in_cv | annotator_name |
|-----------|----------------|----------------------|----------------|
| p1        | false          | -                    | `p1`           |
| p2        | false          | -                    | `p2`           |
| p2        | true           | false                | `p2_balance`   |
| p2        | true           | true                 | `p2_combined`  |

目錄結構範例：
```
data/processed/step4_cv/{annotator_name}/
data/processed/step5_reasoning/{annotator_name}/
data/processed/step5_sft/{annotator_name}/
models/step5_cv/{annotator_name}/
results/step5_cv/{annotator_name}/
```

---

## 步驟 4：FinBERT 偽標籤 + 類別平衡合成（在每個 Fold 內執行）

> 此步驟不再獨立執行，而是整合到步驟 5 的每個 Fold 內。此處描述方法論，具體執行在步驟 5 中。

### 4.1 資料準備

**人工標註資料集**：500 筆（9 類別，來自 Project-16）
**未標註資料集**：約 160,000 筆（同源的 10-K Risk Factor 段落，已有 FinBERT 預測結果）

類別分布：Non-ESG 297 筆（59.4%）、Product Liability 59 筆、Corporate Governance 40 筆、Human Capital 37 筆、Business Ethics & Values 29 筆、Climate Change 25 筆、Pollution & Waste 5 筆、Community Relations 5 筆、Natural Capital 3 筆

### 4.2 FinBERT 偽標籤生成（Pseudo-Labeling）

**目的**：利用 FinBERT 全量預測結果中的高信心樣本擴充訓練資料。FinBERT 與人工標註的一致率為 95.6%，說明其高信心預測具有可靠的品質。

**信心門檻校準**：使用當前 fold 的訓練集（約 333 筆），分析 FinBERT 預測信心分數與人工標註的一致性，找出使 FinBERT 準確率達到可接受水準（>= 95%）的信心門檻。由於 FinBERT 的錯誤模式具有類別依賴性（主要將 ESG 誤判為 Non-ESG），建議計算 per-class 的信心門檻。若單一 fold 的訓練集中某些稀有類別樣本不足以可靠地估計門檻，則退回到使用全類別的統一門檻。

**篩選規則**：
- 從 160,000 筆 FinBERT 預測結果中篩選高信心樣本（排除已標註的 500 筆）
- 對 Non-ESG 類別施加數量上限（cap），建議限制在 800-1,000 筆，避免 Non-ESG 主導偽標籤集
- 其他 ESG 類別按門檻有多少取多少
- 最終目標為篩選出約 2,000-3,000 筆偽標籤資料

### 4.3 Temperature-Based Smoothed Sampling（類別平衡策略）

採用 Conneau et al.（2020）提出的溫度平滑取樣公式，控制各類別的目標訓練數量：

$$T_i = B \times \frac{n_i^\alpha}{\sum_j n_j^\alpha}$$

並施加下界約束：$T_i = \max(T_i,\ n_i)$，確保不對任何類別進行下採樣。

參數設定：
- $\alpha = 0.5$（平方根平衡，在保留原始分布資訊與平衡稀有類別之間取得折衷）
- $B = 4000$（總訓練預算，落在 LoRA 微調的有效範圍 500-5,000 筆內）

此公式的效果是將原始 83:1 的最大最小類別比壓縮至約 9:1，同時避免對稀有類別過度合成（合成比例控制在 90% 以內）。$\alpha$ 作為超參數，可在後續實驗中嘗試 0.3、0.5、0.7 三個值進行比較。

### 4.4 LLM API 合成資料生成

對步驟 4.3 中 $S_i > 0$ 的類別，使用 LLM API（如 GPT-4o、Claude 等）生成合成訓練資料。合成時僅以當前 fold 訓練集中的該類別真實樣本作為 few-shot 範例（seed），不使用驗證集中的任何樣本，以避免資料洩漏。

合成資料的 prompt 設計應包含：
1. 該類別的定義與邊界說明
2. 2-3 筆來自當前 fold 訓練集的真實範例
3. 明確要求生成的文本在語意上符合該類別，但用詞、句式、情境需有所變化
4. 輸出格式與原始資料一致

針對 FinBERT 的系統性弱點（ESG -> Non-ESG 的漏判），額外生成一批邊界案例：這些文本表面上看似 Non-ESG，但實際包含特定 ESG 訊號（特別是 Product Liability、Climate Change、Corporate Governance 這三個最容易被 FinBERT 漏判的類別）。

為控制合成品質，建議在每批合成後進行抽樣人工檢查（約 10-20%），刪除明顯偏離類別定義或品質不佳的樣本。

### 4.5 各類別目標數量估算

合併當前 fold 的訓練集（~333 筆）與偽標籤資料，統計各類別的現有數量 $n_i$。代入公式計算每個類別的目標數量 $T_i$。各類別需要合成的數量為 $S_i = \max(0,\ T_i - n_i)$。

以估計值為例：

| 類別 | 現有數量 | 目標數量 | 需合成 |
|---|---|---|---|
| Non-ESG | ~998 | ~998 | 0 |
| Product Liability | ~439 | ~658 | ~219 |
| Corporate Governance | ~327 | ~568 | ~241 |
| Human Capital | ~275 | ~521 | ~246 |
| Business Ethics & Values | ~219 | ~465 | ~246 |
| Climate Change | ~167 | ~406 | ~239 |
| Pollution & Waste | ~23 | ~151 | ~128 |
| Community Relations | ~18 | ~133 | ~115 |
| Natural Capital | ~12 | ~109 | ~97 |

（實際數量依每 fold 的切分與偽標籤篩選結果而定。）

### 4.6 最終訓練集組裝

合併以下三個來源：
1. 當前 fold 的人工標註訓練集（~333 筆）
2. 偽標籤資料（~2,000-3,000 筆）
3. LLM 合成資料（~1,500 筆）

最終訓練集約 4,000 筆。對訓練集進行隨機打亂（shuffle），確保各來源的資料充分混合。

---

## 步驟 5：Stratified 3-Fold Cross-Validation + LoRA 微調 + 評估

### 5.1 為什麼用 3-Fold 而非 5-Fold

採用分層 3-Fold 交叉驗證，確保每個 fold 的訓練集與驗證集中各類別的比例與整體一致。每次 fold 切分約為 333 筆訓練、167 筆驗證。選擇 3-Fold 而非 5-Fold 的原因是稀有類別（Pollution & Waste 5 筆、Community Relations 5 筆、Natural Capital 3 筆）在 5-Fold 下每個 fold 的驗證集中可能完全缺席，3-Fold 可確保每個稀有類別在驗證集中至少出現 1 筆。

### 5.2 生成 3 折切分

使用 scikit-learn 的 `StratifiedKFold(n_splits=3, shuffle=True, random_state=42)` 對 500 筆人工標註資料切分。固定 random_state 確保所有模型使用完全相同的 fold。

儲存 fold 定義（paragraph_id 的 train/test 分配），所有模型共用同一份 fold 定義。

### 5.3 參與評估的模型配置

**Group A — FinBERT Baseline（1 個配置）：**
FinBERT-ESG-9 原始預訓練模型（直接推論，不需要訓練）

**Group B — API LLM Baseline（3 個配置，直接推論，不訓練）：**

| 模型 | Provider | 說明 |
|-----|----------|------|
| gemini-3-flash-preview | Gemini | 直接對 3-Fold 驗證集推論 |
| gpt-5.4-mini-2026-03-17 | OpenAI | 直接對 3-Fold 驗證集推論 |
| claude-sonnet-4-6 | Anthropic | 直接對 3-Fold 驗證集推論 |

**API 限制（需納入實驗排程考量）**

- Claude:
  Requests per Minute 1K
  Input Tokens per Minute 450K
  Output Tokens per Minute 90K
- Gemini:
  RPM 1K
  TPM 2M
  RPD 10K
- GPT:
  Token limits 10,000,000 TPM
  Request and other limits 10,000 RPM
  Batch queue limits 1,000,000,000 TPD

**Group C — 本地小模型蒸餾版（4 個配置）：**

| 模型 | Hugging Face ID | 參數量 |
|-----|----------------|-------|
| Gemma 3 4B | google/gemma-3-4b-it | 4B |
| Qwen 3.5 4B | Qwen/Qwen3.5-4B | 4B |
| Llama 3.2 3B | meta-llama/Llama-3.2-3B-Instruct | 3B |
| Ministral 3B | mistralai/Ministral-3-3B-Instruct-2512 | 3B |

### 5.4 對每個 Fold（k = 1, 2, 3）依序執行以下步驟

#### 5.4.1 資料切分

使用 Stratified 3-Fold 將 500 筆人工標註資料切分為當前 fold 的訓練集（約 333 筆）與驗證集（約 167 筆）。驗證集在當前 fold 的所有後續步驟中完全隔離，不參與任何訓練資料的生成或篩選。

#### 5.4.2 校準 FinBERT 信心門檻

僅使用當前 fold 的 333 筆訓練集，分析 FinBERT 預測信心分數與人工標註的一致性，找出使 FinBERT 準確率達到可接受水準（>= 95%）的信心門檻。

由於 FinBERT 的錯誤模式具有類別依賴性（主要將 ESG 誤判為 Non-ESG），建議計算 per-class 的信心門檻。若單一 fold 的訓練集中某些稀有類別樣本不足以可靠地估計門檻，則退回到使用全類別的統一門檻。

#### 5.4.3 生成偽標籤資料

基於步驟 5.4.2 得到的信心門檻，從預處理階段的 160,000 筆 FinBERT 預測結果中篩選高信心樣本。為避免 Non-ESG 主導偽標籤集（因 Non-ESG 本身信心普遍較高），對 Non-ESG 類別施加數量上限（cap），建議限制在 800-1,000 筆，其他 ESG 類別則按門檻有多少取多少。最終目標為篩選出約 2,000-3,000 筆偽標籤資料，並記錄各類別的實際數量。

#### 5.4.4 計算各類別目標數量

合併當前 fold 的訓練集（333 筆）與偽標籤資料，統計各類別的現有數量 $n_i$。代入 Temperature-Based Smoothed Sampling 公式（$\alpha = 0.5$，$B = 4000$）計算每個類別的目標數量 $T_i$。各類別需要合成的數量為 $S_i = \max(0,\ T_i - n_i)$。

#### 5.4.5 LLM API 合成資料生成

對步驟 5.4.4 中 $S_i > 0$ 的類別，使用 LLM API 生成合成訓練資料。合成時僅以當前 fold 訓練集中的該類別真實樣本作為 few-shot 範例（seed），不使用驗證集中的任何樣本。

#### 5.4.6 LLM API 批量生成推理原因（Reasoning Generation）

對全部約 4,000 筆訓練資料（人工標註 + 偽標籤 + LLM 合成），使用 LLM API（如 GPT-4o-mini）批量生成 chain-of-thought 推理文字。這些推理是「教學素材」——讓小模型不僅學會「答案是什麼」，還學會「為什麼是這個答案」。

每筆呼叫包含：system prompt 定義角色為 ESG 分類專家、user prompt 包含九類 ESG 類別定義、段落文字（combined_text，包含標題）、以及正確的標籤。要求模型生成 3-5 句的推理，說明段落中的關鍵 ESG 概念、如何對應到該類別定義、為何排除其他類別。

**品質控管**：每筆自動驗證推理長度在 80-200 字之間、不含 markdown 格式。驗證失敗的自動重試最多 3 次。

#### 5.4.7 組裝最終訓練集並轉換為 SFT 格式

合併三個來源（人工標註訓練集 + 偽標籤 + LLM 合成），約 4,000 筆。轉換為 instruction-following 格式，每筆包含：
- **system**：分類專家角色定義
- **user**：分類 prompt + 類別定義 + 段落的 combined_text（不含答案）
- **assistant**：LLM 生成的推理 + 最終分類，使用 XML 格式包裝

assistant 的回覆格式範例：
```xml
<reasoning>This paragraph's key concepts are legal, regulatory, compliance, litigation, court orders, and costs of compliance. These concepts map to Non-ESG because they discuss legal and regulatory risk, litigation outcomes, and impacts on business operations and financial performance rather than environmental, social, or governance-specific topics defined in the ESG categories. There is no discussion of climate, natural resources, pollution, labor practices, product safety, community relations, board/governance structures, or ethical misconduct that would match the other ESG categories. Thus the content is primarily business and legal risk disclosure, not ESG-related risk.</reasoning>
<label>Non-ESG</label>
```

對訓練集進行隨機打亂（shuffle），確保各來源的資料充分混合。

#### 5.4.8 FinBERT 推論

對驗證子集直接推論。不需要訓練。

#### 5.4.9 本地小模型 LoRA 微調 + 推論

用該折訓練子集的 SFT 資料做 LoRA 微調。訓練資料的 assistant 回覆包含 `<reasoning>` 和 `<label>` 兩部分，模型同時學習推理過程和最終分類。微調完成後對驗證子集推論。**每個模型每折重新微調一次**，4 模型 x 3 折 = 12 次微調。

微調超參數（依照 Unsloth 官方筆記本）：
- LoRA: r=16, alpha=16, dropout=0, bias="none"
- Training: lr=2e-4, batch=2, grad_accum=4, epochs=1, warmup_steps=5
- Optimizer: adamw_8bit, weight_decay=0.001, lr_scheduler=linear
- random_state/seed: 3407, max_seq_length: 2048

實作腳本：`src/fine-tuning/run_cv_finetune.py`

#### 5.4.10 API LLM 直接對驗證集推論

除 FinBERT 與本地 LoRA 小模型外，額外讓 API LLM 直接對每個 fold 的驗證集推論，作為 closed-weight instruction-following baseline。這些模型不參與任何本地訓練，只使用統一的 ESG 分類 prompt 對驗證集做 zero-shot / instruction-following 分類。

評估模型包括：
- `gemini-3-flash-preview`
- `gpt-5.4-mini-2026-03-17`
- `claude-sonnet-4-6`

執行方式：
- 對每個 fold 的驗證集（約 165-168 筆）逐筆呼叫 API
- prompt 與本地模型推論使用相同的 ESG 類別定義與輸出格式
- 每個模型獨立輸出 `fold_k_results.json`、`fold_k_metrics.json`
- 三折跑完後額外輸出整體 `overall_folds.json`、`overall_summary.json`、`overall_summary.md`

實作腳本：`src/step5_cv/run_cv_api_llm.py`

### 5.5 推論結果後處理

本地小模型微調後的輸出包含推理（`<reasoning>`）和分類標籤（`<label>`）。標籤解析優先順序：先找 `<label>...</label>` XML 標記 -> 找不到則以 "classified as:" 後面的文字 -> 找不到則取最後一行 -> 標準化比對（處理 "Pollution and Waste" vs "Pollution & Waste" 等變體）-> 無法解析標記為 Unclassified。

### 5.6 評估指標

每個模型配置報告 3 折的平均值 +/- 標準差：

- Overall Accuracy
- Macro F1-score / Weighted F1-score
- 各類別 Precision / Recall / F1-score
- Cohen's Kappa vs Ground Truth
- Pillar-level F1：Environmental（CC + NC + PW）、Social（HC + PL + CR）、Governance（CG + BEV）、Non-ESG

對於稀有類別（Pollution & Waste、Community Relations、Natural Capital），因驗證集中樣本數極少（每 fold 僅 1-2 筆），其 per-class F1 的波動較大，應在報告中標註此限制。

---

## 實驗總量估算

| 類別 | 模型數 | x 設定 | x 3 折 | = 總推論次數 |
|-----|-------|--------|--------|------------|
| FinBERT | 1 | x 1 | x 3 | 3 輪 x 167 筆 = 501 次本地推論 |
| API LLM | 3 | x 1 | x 3 | 9 輪 x 167 筆 = 1,503 次 API 推論 |
| 本地小模型 | 4 | x 1 (微調) | x 3 | 12 輪 x 167 筆 = 2,004 次本地推論 |
| **微調次數** | | | | **12 次**（4 模型 x 3 折）|
| **偽標籤 + 合成** | | | | 3 折 x (信心門檻校準 + 偽標籤篩選 + LLM 合成) |

---

## 消融實驗

### 資料組成消融（p1 標註者）

進一步進行以下消融實驗以驗證各組件的貢獻：

- **無偽標籤**：僅用 333 筆人工標註 + LLM 合成，評估偽標籤的貢獻
- **無 LLM 合成**：僅用人工標註 + 偽標籤（不做類別平衡），評估合成資料的貢獻
- **不同 alpha 值**：比較 alpha = 0.3、0.5、0.7 的效果
- **不同基座模型**：比較 Gemma 3 4B、Qwen 3.5 4B、Llama 3.2 3B、Ministral 3B

### Reasoning 消融

- **label_only**：移除訓練資料中的 `<reasoning>` 標籤，assistant 回應格式簡化為 `Label: {label}`，測試 chain-of-thought 推理的貢獻
- 配置：`step5_cv.finetune.ablation.label_only: true`

### 第二標註者 (p2) 實驗

使用 `chiang_500.csv`（標註者 p2）進行三種實驗，皆不生成偽標籤與合成資料：

**實驗 1 (p2_balance)**：chiang_500 做 3-fold CV split，blance_500 全部加入 train 訓練
- config：`annotator: p2`, `balance.enabled: true`, `balance.include_in_cv: false`
- `pseudo_labels.enabled: false`, `synthetic_generation.enabled: false`
- blance_500 作為獨立資料來源加入每個 fold 的 train（source="balance"），不進入 val

**實驗 2 (p2)**：chiang_500 做 3-fold CV split，僅用 ~333 筆 train 訓練
- config：`annotator: p2`, `balance.enabled: false`
- `pseudo_labels.enabled: false`, `synthetic_generation.enabled: false`

**實驗 3 (p2_combined)**：chiang_500 + blance_500 合併後做 3-fold CV split
- config：`annotator: p2`, `balance.enabled: true`, `balance.include_in_cv: true`
- `pseudo_labels.enabled: false`, `synthetic_generation.enabled: false`
- 合併後的 1,000 筆資料一起做 StratifiedKFold split，train 和 val 都可能包含兩個來源的資料

---

## 資料流摘要

### p1 標註者（完整 pipeline）

```
[步驟 0] SEC EDGAR 下載 [x]
 S&P 500 x 2021-2025 ≈ 2,500 份 10-K HTML
    |
[步驟 1] sec-parser Semantic Parsing + 清洗 [x]
 heading-aware 段落切割，保留標題，移除表格/雜訊
 -> 全量段落池（~160,000 段）
    |
[步驟 2] FinBERT 全量預測 + 隨機抽樣 [x]
 FinBERT 預測全量段落（預標註用 + 偽標籤用）
 隨機抽樣 500 筆（已帶 FinBERT 預測）
    |
[步驟 3] Label Studio 人工標註（含 FinBERT 預標註）[x]
 p1 (Project 16): 500 筆（單一標注者）含預標註
 FinBERT 與人工標註一致率 95.6%
 -> 最終 500 筆人工標籤
    |
[步驟 4-5] 整合為 Stratified 3-Fold CV Pipeline (annotator_name=p1)
 |
 對每個 Fold (k=1,2,3)：
 |
 ├── 5.4.1 Stratified 3-Fold 切分 500 筆
 |   -> 訓練集 ~333 筆 + 驗證集 ~167 筆
 |
 ├── 5.4.2 校準 FinBERT 信心門檻
 |   僅用訓練集 ~333 筆校準
 |
 ├── 5.4.3 偽標籤生成 <── FinBERT 全量預測結果
 |   篩選高信心 ~2,000-3,000 筆
 |   Non-ESG cap 在 800-1,000 筆
 |
 ├── 5.4.4 Temperature-Based Smoothed Sampling
 |   計算各類別目標數量 (alpha=0.5, B=4000)
 |
 ├── 5.4.5 LLM API 合成資料
 |   補足低頻類別，約 ~1,500 筆
 |
 ├── 5.4.6 LLM API 批量生成推理原因
 |   對 ~4,000 筆生成 <reasoning> chain-of-thought
 |
 ├── 5.4.7 組裝 SFT 訓練集
 |   ~4,000 筆，格式: <reasoning>...</reasoning><label>...</label>
 |
 ├── 5.4.8 FinBERT 對驗證集推論
 |
 └── 5.4.9 本地小模型 LoRA 微調 + 推論
     訓練含 reasoning + label，4 模型 x 3 折 = 12 次微調
 |
 └── 5.4.10 API LLM 對驗證集直接推論
     3 個 closed-weight 模型 x 3 折 = 9 輪 API 推論
```

### p2 標註者實驗（三種配置）

```
[步驟 3] p2 標註者 (chiang_500.csv) + 平衡資料集 (blance_500.csv)

------- 實驗 1 (annotator_name=p2_balance) -------
chiang_500 做 3-fold split
 -> 訓練集 ~333 筆 + 驗證集 ~167 筆
 -> 訓練集 + blance_500 全量加入 train
 -> 生成 reasoning -> 組裝 SFT -> 微調 + 推論

------- 實驗 2 (annotator_name=p2) -------
chiang_500 做 3-fold split
 -> 訓練集 ~333 筆 + 驗證集 ~167 筆
 -> 僅用 ~333 筆 train
 -> 生成 reasoning -> 組裝 SFT -> 微調 + 推論

------- 實驗 3 (annotator_name=p2_combined) -------
chiang_500 + blance_500 合併（~1,000 筆）做 3-fold split
 -> 訓練集 ~667 筆 + 驗證集 ~333 筆
 -> 生成 reasoning -> 組裝 SFT -> 微調 + 推論

* 三種實驗皆不生成偽標籤與合成資料
* pseudo_labels.enabled: false, synthetic_generation.enabled: false
```

---

## 參考文獻

- He, H., Bai, Y., Garcia, E. A., & Li, S. (2008). ADASYN: Adaptive synthetic sampling approach for imbalanced learning. *IEEE IJCNN*, 1322-1328.
- Conneau, A., et al. (2020). Unsupervised Cross-lingual Representation Learning at Scale. *ACL 2020*.
- Hu, Q., et al. (2023). Synthetic Data as Validation. *arXiv:2310.16052*.
- Nakada, R., et al. (2024). Synthetic Oversampling: Theory and A Practical Approach Using LLMs to Address Data Imbalance. *arXiv:2406.03628*.
