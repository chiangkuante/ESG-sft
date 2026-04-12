# ESG 文本分類 LoRA 微調實驗設計

## 實驗目標

以 FinBERT-ESG 為教師模型，透過知識蒸餾策略訓練 LoRA 微調的 LLM（Gemma 3B），使其學習並超越 FinBERT-ESG 在 10-K Risk Factor 段落的 ESG 分類能力。FinBERT-ESG 與人工標註的一致率為 95.6%（500 筆中僅 22 筆不一致），其系統性弱點集中於將 ESG 類別誤判為 Non-ESG（22 筆中 16 筆屬此模式）。

## 資料概況

- **人工標註資料集**：500 筆（9 類別，來自 Project-16）
- **未標註資料集**：約 160,000 筆（同源的 10-K Risk Factor 段落）
- **類別分布**：Non-ESG 297 筆（59.4%）、Product Liability 59 筆、Corporate Governance 40 筆、Human Capital 37 筆、Business Ethics & Values 29 筆、Climate Change 25 筆、Pollution & Waste 5 筆、Community Relations 5 筆、Natural Capital 3 筆

## 類別平衡策略：Temperature-Based Smoothed Sampling

採用 Conneau et al.（2020）提出的溫度平滑取樣公式，控制各類別的目標訓練數量：

$$T_i = B \times \frac{n_i^\alpha}{\sum_j n_j^\alpha}$$

並施加下界約束：$T_i = \max(T_i,\ n_i)$，確保不對任何類別進行下採樣。

參數設定：

- $\alpha = 0.5$（平方根平衡，在保留原始分布資訊與平衡稀有類別之間取得折衷）
- $B = 4000$（總訓練預算，落在 LoRA 微調的有效範圍 500–5,000 筆內）

此公式的效果是將原始 83:1 的最大最小類別比壓縮至約 9:1，同時避免對稀有類別過度合成（合成比例控制在 90% 以內）。α 作為超參數，可在後續實驗中嘗試 0.3、0.5、0.7 三個值進行比較。

## 驗證架構：Stratified 3-Fold Cross-Validation

採用分層 3-Fold 交叉驗證，確保每個 fold 的訓練集與驗證集中各類別的比例與整體一致。每次 fold 切分約為 333 筆訓練、167 筆驗證。選擇 3-Fold 而非 5-Fold 的原因是稀有類別（Pollution & Waste 5 筆、Community Relations 5 筆、Natural Capital 3 筆）在 5-Fold 下每個 fold 的驗證集中可能完全缺席，3-Fold 可確保每個稀有類別在驗證集中至少出現 1 筆。

---

## 實驗步驟

### 預處理：FinBERT 全量推論（僅執行一次）

對 160,000 筆未標註資料執行 FinBERT-ESG 推論，取得每筆資料的預測類別（hard label）與 softmax 信心分數。將結果儲存為結構化檔案，後續各 fold 直接讀取篩選，無需重複推論。

### 對每個 Fold（k = 1, 2, 3）依序執行以下步驟：

#### 步驟一：資料切分

使用 Stratified 3-Fold 將 500 筆人工標註資料切分為當前 fold 的訓練集（約 333 筆）與驗證集（約 167 筆）。驗證集在當前 fold 的所有後續步驟中完全隔離，不參與任何訓練資料的生成或篩選。

#### 步驟二：校準 FinBERT 信心門檻

僅使用當前 fold 的 333 筆訓練集，分析 FinBERT 預測信心分數與人工標註的一致性，找出使 FinBERT 準確率達到可接受水準（例如 ≥ 95%）的信心門檻。

由於 FinBERT 的錯誤模式具有類別依賴性（主要將 ESG 誤判為 Non-ESG），建議計算 per-class 的信心門檻。若單一 fold 的訓練集中某些稀有類別樣本不足以可靠地估計門檻，則退回到使用全類別的統一門檻。

#### 步驟三：生成偽標籤資料

基於步驟二得到的信心門檻，從預處理階段的 160,000 筆 FinBERT 預測結果中篩選高信心樣本。為避免 Non-ESG 主導偽標籤集（因 Non-ESG 本身信心普遍較高），對 Non-ESG 類別施加數量上限（cap），建議限制在 800–1,000 筆，其他 ESG 類別則按門檻有多少取多少。最終目標為篩選出約 2,000–3,000 筆偽標籤資料，並記錄各類別的實際數量。

#### 步驟四：計算各類別目標數量

合併當前 fold 的訓練集（333 筆）與偽標籤資料，統計各類別的現有數量 $n_i$。代入 Temperature-Based Smoothed Sampling 公式（$\alpha = 0.5$，$B = 4000$）計算每個類別的目標數量 $T_i$。各類別需要合成的數量為 $S_i = \max(0,\ T_i - n_i)$。

以估計值為例，各類別的大致目標為：

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

#### 步驟五：LLM API 合成資料生成

對步驟四中 $S_i > 0$ 的類別，使用 LLM API（如 GPT-4o、Claude 等）生成合成訓練資料。合成時僅以當前 fold 訓練集中的該類別真實樣本作為 few-shot 範例（seed），不使用驗證集中的任何樣本，以避免資料洩漏。

合成資料的 prompt 設計應包含：

1. 該類別的定義與邊界說明
2. 2–3 筆來自當前 fold 訓練集的真實範例
3. 明確要求生成的文本在語意上符合該類別，但用詞、句式、情境需有所變化
4. 輸出格式與原始資料一致

針對 FinBERT 的系統性弱點（ESG → Non-ESG 的漏判），額外生成一批邊界案例：這些文本表面上看似 Non-ESG，但實際包含特定 ESG 訊號（特別是 Product Liability、Climate Change、Corporate Governance 這三個最容易被 FinBERT 漏判的類別）。

為控制合成品質，建議在每批合成後進行抽樣人工檢查（約 10–20%），刪除明顯偏離類別定義或品質不佳的樣本。

#### 步驟六：組裝最終訓練集

合併以下三個來源：

1. 當前 fold 的人工標註訓練集（~333 筆）
2. 偽標籤資料（~2,000–3,000 筆）
3. LLM 合成資料（~1,500 筆）

最終訓練集約 4,000 筆。對訓練集進行隨機打亂（shuffle），確保各來源的資料充分混合。

#### 步驟七：LoRA 微調訓練

使用 LoRA 對 Gemma 3B（或其他候選基座模型）進行微調。訓練資料格式為 instruction-following 格式，每筆包含系統指令（含 9 個類別的定義）、輸入文本（10-K Risk Factor 段落）、以及預期輸出（類別標籤）。

建議的初始超參數：

- LoRA rank：16
- LoRA alpha：32
- Learning rate：2e-4
- Batch size：8
- Epochs：3–5（搭配 early stopping，以驗證集 loss 為監控指標）
- Warmup ratio：0.1

#### 步驟八：驗證與評估

在當前 fold 的驗證集（~167 筆，純人工標註）上評估模型表現。計算以下指標：

- Per-class Precision、Recall、F1-Score
- Macro-F1（各類別 F1 的簡單平均，對稀有類別賦予同等權重）
- Weighted-F1（按類別數量加權，反映整體表現）

同時在相同的驗證集上計算 FinBERT-ESG 的對應指標，作為基線比較。

對於稀有類別（Pollution & Waste、Community Relations、Natural Capital），因驗證集中樣本數極少（每 fold 僅 1–2 筆），其 per-class F1 的波動較大，應在報告中標註此限制。

---

## 結果彙報

彙報 3 個 fold 的各項指標之平均值與標準差。主要比較維度為：

1. **LoRA LLM vs. FinBERT-ESG**：在相同驗證集上的 Macro-F1 與 Weighted-F1 對比
2. **Per-class 改善分析**：特別關注 FinBERT 系統性弱點類別（Product Liability、Climate Change、Corporate Governance 被誤判為 Non-ESG 的案例）是否獲得改善
3. **穩定性**：3 fold 之間的標準差，評估模型的穩健性

## 消融實驗

進一步進行以下消融實驗以驗證各組件的貢獻：

- **無偽標籤**：僅用 333 筆人工標註 + LLM 合成，評估偽標籤的貢獻
- **無 LLM 合成**：僅用人工標註 + 偽標籤（不做類別平衡），評估合成資料的貢獻
- **不同基座模型**：比較 Gemma 3B、Llama 3.2 3B 等候選模型

## 參考文獻

- He, H., Bai, Y., Garcia, E. A., & Li, S. (2008). ADASYN: Adaptive synthetic sampling approach for imbalanced learning. *IEEE IJCNN*, 1322–1328.
- Conneau, A., et al. (2020). Unsupervised Cross-lingual Representation Learning at Scale. *ACL 2020*.
- Hu, Q., et al. (2023). Synthetic Data as Validation. *arXiv:2310.16052*.
- Nakada, R., et al. (2024). Synthetic Oversampling: Theory and A Practical Approach Using LLMs to Address Data Imbalance. *arXiv:2406.03628*.
