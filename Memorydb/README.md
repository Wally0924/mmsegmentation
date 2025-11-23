# 🗺️ VLM-First 場景先驗記憶庫 - 完整建置說明文件

[toc]

## 1. 🚀 架構總覽 (Architecture Overview)

本記憶庫採用「**VLM 作為教師 (VLM-as-Teacher)**」的非對稱架構。

> **核心理念：**
> 我們利用 **VLM (LLaVA)** 和 **S-BERT** 的強大語意理解能力，作為離線的「**教師**」，來克服單純影像特徵在光影、天氣變化下的分群錯誤問題。
>
> 同時，我們利用 **DINOv2** 強健的影像特徵作為「**學生**」，用於構建最終的**快速查詢索引 (Key)**，確保系統在上線時能達到毫秒級的即時回應。

---

## 2. 🤖 核心模型與工具

1.  **DINOv2** (`facebook/dinov2-base`):
    * **用途:** 提取穩健的純影像特徵，作為即時查詢的基礎。
2.  **LLaVA-NeXT** (`llava-hf/llava-v1.6-mistral-7b-hf`):
    * **用途:** 產生不受天氣影響的語意摘要 JSON，提供分群所需的智能。
3.  **Sentence-BERT** (`all-mpnet-base-v2`):
    * **用途:** 將語意摘要轉換為數學向量。
4.  **K-Means** (`scikit-learn`):
    * **用途:** 在高維混合特徵空間中進行精準分群。
5.  **ChromaDB**:
    * **用途:** 向量資料庫，儲存最終成果。

---

## 3. 🛠️ 建置步驟詳解

請依序執行以下所有腳本。確保您處於已安裝好環境的 `memory_factory` Conda 環境中。

### 階段 1a：提取「快學生」影像特徵 (DINOv2)

* **執行腳本：** `01_extract_feature.py`
* **作法：** 使用 DINOv2 模型對所有影像提取特徵，並進行 L2 標準化。
* **產出：** `all_image_features.npy` (768 維)
* **關鍵參數：**
    * `MODEL_ID = "facebook/dinov2-base"`

### 階段 1b：提取「慢老師」語意摘要 (LLaVA)

* **執行腳本：** `01a_llm_semantic.py`
* **作法：** 強迫 LLaVA 忽略天氣與光影，為每一張影像生成一份結構化的 JSON 摘要。
* **產出：** `llava_summaries.json` (387 份 JSON 字串)
* **關鍵參數：**
    * `ENGLISH_JSON_PROMPT`: 包含 `road_layout`, `ocr_text_on_signs` 等欄位的嚴格 Prompt。
    * `BATCH_SIZE`: 根據顯卡記憶體調整 (例如 8)。

### 階段 1c：提取「慢老師」語意向量 (S-BERT)

* **執行腳本：** `01b_text_vector.py`
* **作法：** 將 JSON 摘要轉換為標準化字串，並使用 S-BERT 編碼為文字向量，最後進行 L2 標準化。
* **產出：** `all_text_features.npy` (768 維)

### 階段 1d：特徵融合 (Teacher Feature)

* **執行腳本：** `01c_merge_feature.py`
* **作法：** 將 DINOv2 影像向量與 S-BERT 文字向量拼接。
* **產出：** `all_joint_features.npy` (1536 維)

### 階段 2：VLM 指導分群 (Clustering)

* **執行腳本：** `02_run_clustering.py`
* **作法：** 在 1536 維的混合特徵空間上執行 K-Means，產生最準確的分群結果（標準答案）。
* **產出：**
    * `cluster_labels.npy`: 每張影像的群組 ID (0-35)。
    * `cluster_centers.npy`: **(注意)** 這裡產生的是 1536 維中心點，我們在下一階段會替換它。
* **關鍵參數：**
    * `INPUT_FEATURES_FILE = "all_joint_features.npy"`
    * `K_CLUSTERS`: 設定群組數量 (例如 36)。

### 階段 3a：計算快速查詢 Key (New)

* **執行腳本：** `02c_calculate_image_centers.py` (需新建)
* **作法：** 讀取 `cluster_labels.npy` (老師的答案) 和 `all_image_features.npy` (學生的特徵)，計算每個群組的**影像向量平均值**。
* **產出：** `image_only_cluster_centers.npy` (768 維，這將是最終的 **Key**)。

### 階段 3b：計算機率型 Value (Probability)

* **執行腳本：** `03_calculate_probabilities.py`
* **作法：** 統計每個群組中，LLaVA JSON 各個特徵出現的頻率，轉換為機率。
* **產出：** `semantic_summaries.json` (機率型 JSON，這將是最終的 **Value**)。

### 階段 4：建置最終記憶庫 (Database)

* **執行腳本：** `04_build_memory_db.py`
* **作法：** 將 Key 和 Value 存入 ChromaDB。
* **關鍵修改：**
    * `INPUT_CENTERS_FILE`: 指向 `image_only_cluster_centers.npy` (768 維)。
    * `INPUT_SUMMARIES_FILE`: 指向 `semantic_summaries.json`。
* **產出：** `memory_db_chroma/` 資料夾。

---

## 4. 🔬 驗證流程 (Verification)

建置完成後，請執行以下驗證。

### A. 視覺化驗證

* **執行腳本：** `05_visualize_clusters.py`
* **方法：** 檢查 `all_clusters_visualization/` 資料夾。確認同一群組內是否**同時包含**了白天、夜晚、晴天、雨天的影像。如果是，代表 VLM 指導分群成功。

### B. 線上查詢模擬

* **執行腳本：** (自訂測試腳本)
* **方法：**
    1. 讀取一張新的測試圖片。
    2. 執行 `01_extract_feature.py` 中的 DINOv2 邏輯產生 768 維向量。
    3. 用該向量查詢 `memory_db_chroma`。
    4. 確認回傳的 Value (機率 JSON) 是否準確描述了該場景。