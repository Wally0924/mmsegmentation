### 階段 1：(VLM) 生成全影像語意摘要

#### 腳本：`01a_llm_semantic.py`

* **🎯 目的 (Goal):**
    這是整個 VLM-First 架構中**最耗時、也最核心**的一步。
    它的任務**不是**分群，而是「特徵提取」。它會讀取您的 N 張（例如 387 張）影像，並強迫 LLaVA-NeXT **為每一張影像**生成一份「JSON 格式的語意摘要」。

* **💡 作法 (Method):**
    1.  此腳本會載入 `LLaVA-NeXT` 模型到GPU。
    2.  它會讀取 `image_filenames.json` 來獲取檔案清單。
    3.  它會以 `BATCH_SIZE`（例如 8）為單位，將影像和 `ENGLISH_JSON_PROMPT` **批次**送入 VLM。
    4.  VLM 會被 Prompt 強迫忽略光影、天氣（解決白天/夜晚問題），並專注於 OCR 和固定結構。
    5.  最後，它會儲存一個 `檔名` -> `JSON 字串` 的大型字典。

* **📥 輸入 (Inputs):**
    * `image_filenames.json` (來自您舊的 01_extract_feature)
    * `data/training_images/` (您的原始影像)

* **📤 輸出 (Outputs):**
    * `llava_summaries.json` (包含 387 筆 `檔名: JSON字串` 的字典)

* **⚠️ 關鍵參數與檢查點 (Parameters to Check):**

    * `BATCH_SIZE`:
        * **說明：** 這是優化的關鍵。`16` 可能太高導致 OOM (CUDA Out of Memory)，`8` 是一個安全且高效的設定。
        * **檢查：** 如果您遇到 OOM 錯誤，請**降低**此數值 (例如 `8` -> `4`)。

    * `ENGLISH_JSON_PROMPT`:
        * **說明：** **這是此腳本最重要的部分**。您必須在這裡定義 LLaVA 該如何「思考」。
        * **檢查：** 請確保 JSON 結構中包含了**高獨特性**的欄位（例如 `ocr_text_on_signs`），這能幫助解決「不同地點、相似場景」的混淆問題。

    * `(Method) Model Loading`:
        * **說明：** 為了最大化GPU效能，此腳本**不使用** `load_in_4bit=True`。
        * **檢查：** 它使用的是 `torch_dtype=torch.float16`，這是 FP16 全速推論。

---

### 階段 2：(S-BERT) 語意摘要轉為文字向量

#### 腳本：`01b_text_vector.py`

* **🎯 目的 (Goal):**
    此腳本的目的是將「階段 1」產生的 N 份「**JSON 文字摘要**」，轉換為 K-Means 演算法能看懂的「**數學向量**」。
    這是 VLM-First 架構的第二步，也是**確保分群精準度**的關鍵。

* **💡 作法 (Method):**
    1.  此腳本會載入一個專門用於「文字相似度」的模型：**Sentence-BERT (S-BERT)**。
    2.  它會讀取 `image_filenames.json` (確保順序) 和 `llava_summaries.json` (VLM 產生的 JSON 字串)。
    3.  它會將 JSON 字串（例如 `"{'road_layout': 'T-junction', ...}"`）**本身**作為「句子」送入 S-BERT。
    4.  S-BERT 會將這些「句子」編碼成 768 維的「文字向量」。
    5.  **核心優勢：** 兩個**一模一樣**的 JSON 字串，會產生**一模一樣**的向量。這就解決了「白天/夜晚」影像向量不同的問題。

* **📥 輸入 (Inputs):**
    * `llava_summaries.json` (來自 `01b_` 的 N 份 JSON 字串)
    * `image_filenames.json` (用來確保向量順序與原始影像一致)

* **📤 輸出 (Outputs):**
    * `all_text_features.npy` (形狀為 `[(number_images), 768]` 的**新**特徵向量)

* **⚠️ 關鍵參數與檢查點 (Parameters to Check):**

    * `MODEL_ID`:
        * **說明：** `all-mpnet-base-v2` 是 `S-BERT` 的一個高效能英文模型，專精於句子相似度。
        * **注意：** 我們在這裡**不使用 CLIP**，因為 `S-BERT` 在「純文字 vs 純文字」的比較上更為精確。
---

### 階段 3：(K-Means) 對文字向量进行分群

#### 腳本：`02_run_clustering.py`

* **🎯 目的 (Goal):**
    在「階段 2」中，我們獲得了 `all_text_features.npy` (基於 VLM 摘要的文字向量)。
    此腳本的任務是使用 `K-Means` 演算法，讀取這些「文字向量」，並將它們自動歸納為 `K` 個「地點群組」。

* **💡 作法 (Method):**
    1.  此腳本**需要被修改**，以讀取「階段 2」的新產出。
    2.  它會載入 `scikit-learn` 函式庫中的 `KMeans` 模型。
    3.  它會執行 `KMeans.fit()`，找出 `K` 個（例如 50 個）群組的中心點。
    4.  它會儲存兩份關鍵的「分群結果」檔案。

* **📥 輸入 (Inputs):**
    * `all_text_features.npy` (來自 `01c_` 的 N 個**文字向量**)

* **📤 輸出 (Outputs):**
    * `cluster_labels.npy`: (1D 陣列) 記錄 N 張影像中，**每一張**被分配到的群組 ID (例如 0 到 49)。
    * `cluster_centers.npy`: (K x 768 矩陣) 記錄 K 個群組的**「中心向量」**。**這將是您記憶庫的 Key (索引鍵)**。

* **⚠️ 關鍵參數與檢查點 (Parameters to Check):**

    * `INPUT_FEATURES_FILE`:
        * **說明：** **(!!! 必須修改 !!!)** 這是執行此腳本前**必須**手動修改的參數。
        * **檢查：** 您必須將此變數從舊的 `all_features.npy` (影像向量)
        * **改為：** `INPUT_FEATURES_FILE = "all_text_features.npy"` (文字向量)

    * `K_CLUSTERS`:
        * **說明：** **(!!! 必須設定 !!!)** 這是 K-Means 的核心參數，代表您希望將資料分為多少個「獨特地點」。
        * **檢查：** 您必須手動設定一個整數 (例如 `50`)。您可以透過「手肘法」或「領域知識」來決定 K 值。這個 K 值將決定您記憶庫的大小。

    * `random_state`:
        * **說明：** 為了讓 K-Means 的結果可被重現，我們將其設為一個固定值 (例如 `42`)。
        * **檢查：** 請保持此設定，以確保您每次重新執行時，`cluster_labels.npy` 的內容都完全相同。

# 🗺️ VLM 記憶庫建置說明 (4/6)

---

### 階段 4：(LLaVA) 生成最終群組摘要 (Value)

#### 腳本：`03_generate_summaries.py`

* **🎯 目的 (Goal):**
    在「階段 3」中，我們得到了 `K` 個群組。但是，我們資料庫的 `Key` (`cluster_centers.npy`) 是基於「JSON 字串」產生的。
    我們**不一定**想把那些冗長的 JSON 字串當作記憶庫的 `Value` (值)。
    
    此腳本的任務是：為這 `K` 個群組，各產生**一份**乾淨的、**人類可讀的「單句摘要」**。這將成為我們資料庫的 **Value (值)**。

* **💡 作法 (Method):**
    > **釐清：為什麼 LLaVA 要執行第二次？**
    > * **階段 1 (`01b_`) 的 LLaVA：** 任務是「**特徵提取**」。輸入 1 張圖，輸出 1 份 JSON。**目的是為了分群**。 (執行 N=387 次)
    > * **階段 4 (`03_`) 的 LLaVA：** 任務是「**總結歸納**」。輸入 5 張圖，輸出 1 句話。**目的是為了資料庫 Value**。(執行 K=50 次)

    1.  此腳本會讀取 `cluster_labels.npy` (來自階段 3) 和 `image_filenames.json`。
    2.  它會遍歷 `K` 個群組 (例如 0 到 49)。
    3.  對於**每一個**群組，它會隨機挑選 `NUM_IMAGES_PER_CLUSTER` (例如 5) 張影像。
    4.  它會將這 5 張影像，連同一個**新的「總結型」Prompt** (`GOLDEN_PROMPT`)，**批次**送入 LLaVA。
    5.  LLaVA 會根據這 5 張圖，歸納出一句「單句摘要」。

* **📥 輸入 (Inputs):**
    * `cluster_labels.npy` (來自 `02_`，基於**文字向量**的分群結果)
    * `image_filenames.json` (來自舊 `01_`)
    * `data/training_images/` (原始影像)

* **📤 輸出 (Outputs):**
    * `semantic_summaries.json` (包含 K 筆 `ID: "單句摘要"` 的字典)。**這將是您記憶庫的 Value (值)**。

* **⚠️ 關鍵參數與檢查點 (Parameters to Check):**

    * `K_CLUSTERS`:
        * **說明：** **(!!! 必須設定 !!!)** 必須與 `02_run_clustering.py` 中的 `K_CLUSTERS` **完全一致** (例如 `50`)。

    * `NUM_IMAGES_PER_CLUSTER`:
        * **說明：** 決定要用多少張影像來「總結」一個地點。`5` 是一個很好的平衡點。

    * `GOLDEN_PROMPT` (或 `PROMPT`):
        * **說明：** **(!!! 關鍵 !!!)** 這裡的 Prompt **不應該**是「JSON 格式」的 Prompt。
        * **檢查：** 這裡的 Prompt 應該是「總結型」的，例如：`Summarize this location in a single, descriptive sentence.` (請用一句話總結這個地點)。

    * `(Logic) Parsing`:
        * **說明：** 腳本的 5.6 節（解碼）是關鍵。
        * **檢查：** 確保它使用了 `response_full.split("[/INST]")[-1].strip()`，以便從 LLaVA 的回覆中正確剖析出乾淨的「單句摘要」。

# 🗺️ VLM 記憶庫建置說明 (5/6)

---

### 階段 5：(ChromaDB) 建置最終記憶庫

#### 腳本：`04_build_memory_db.py`

* **🎯 目的 (Goal):**
    在「階段 3」我們得到了 `cluster_centers.npy` (K 個**Key** / 向量)。
    在「階段 4」我們得到了 `semantic_summaries.json` (K 個**Value** / 摘要)。
    
    此腳本的任務是：初始化一個 `ChromaDB` (向量資料庫)，並將這 K 組 **Key-Value 對** 永久儲存到資料庫中。

* **💡 作法 (Method):**
    1.  此腳本會載入 `cluster_centers.npy` 和 `semantic_summaries.json`。
    2.  它會驗證 `K` 值的數量是否一致。
    3.  它會初始化一個 `chromadb.PersistentClient`，這會將資料庫儲存在硬碟的資料夾中。
    4.  它會建立一個 `Collection` (資料表)，並**指定使用「餘弦相似度 (cosine similarity)」**，這對於 CLIP/S-BERT 向量是最佳的度量方式。
    5.  它會將 K 筆資料「批次」存入資料庫中。

* **📥 輸入 (Inputs):**
    * `cluster_centers.npy` (來自 `02_`，基於**文字向量**的 K 個中心點 [**Keys**])
    * `semantic_summaries.json` (來自 `03_`，K 句人類可讀的摘要 [**Values**])

* **📤 輸出 (Outputs):**
    * `memory_db_chroma/` (資料夾)：**此即為您的最終記憶庫成品**。您未來的「即時」系統將會讀取這個資料夾。

* **⚠️ 關鍵參數與檢查點 (Parameters to Check):**

    * `INPUT_CENTERS_FILE`:
        * **檢查：** 必須設為 `cluster_centers.npy`。

    * `INPUT_SUMMARIES_FILE`:
        * **檢查：** 必須設為 `semantic_summaries.json`。

    * `CHROMA_DB_PATH`:
        * **說明：** 您的最終資料庫資料夾的名稱。預設為 `memory_db_chroma`。

    * `COLLECTION_NAME`:
        * **說明：** 您的「資料表」名稱。您未來的查詢程式**必須**使用這個名稱才能讀取資料。
        * **檢查：** 預設為 `vlm_memory`。

    * `(Method) Metadata`:
        * **說明：** 在 `client.get_or_create_collection` 中，我們傳入了 `metadata={"hnsw:space": "cosine"}`。
        * **檢查：** 這是確保資料庫使用「餘弦相似度」的關鍵設定，請確保它存在。

# 🗺️ VLM 記憶庫建置說明 (6/6)

---

### 階段 6：(Verification) 驗證與視覺化

* **🎯 目的 (Goal):**
    在建置完成後，我們**必須**進行人工和自動化驗證，以確保：
    1.  K-Means 分群（階段 3）的結果是否合理？（VLM-First 是否真的解決了白天/夜晚問題？）
    2.  資料庫（階段 5）的儲存是否 100% 正確？

---

#### A. 驗證 K-Means 分群結果 (人工全覽)

* **執行腳本：** `05_visualize_clusters.py`
* **作法 (Method):**
    1.  此腳本會讀取 `cluster_labels.npy`（來自階段 3）和 `image_filenames.json`。
    2.  它會建立一個名為 `all_clusters_visualization/` 的總資料夾。
    3.  它會遍歷 387 張影像，並將**每一張**都**複製**到 K 個（例如 `cluster_00/` 到 `cluster_49/`）對應的子資料夾中。
* **驗證方法 (Verification):**
    * 執行 `python 05_visualize_clusters.py`。
    * **手動打開 `all_clusters_visualization/` 資料夾。**
    * **(!!! 關鍵驗證 !!!)** 隨機點開幾個資料夾（例如 `cluster_23/`），並檢查：
        * 裡面的圖片是否真的是同一個地點？
        * **裡面是否同時包含了「白天」和「夜晚」的影像？**
    * 如果「是」，則代表您的 VLM-First 架構**大獲成功**。

---

#### B. 驗證資料庫查詢 (自動抽查)

* **執行腳本：** `test_memory_db.py`
* **作法 (Method):**
    1.  此腳本會同時讀取 `memory_db_chroma/` 資料庫和我們的「標準答案」檔案（`cluster_centers.npy` 和 `semantic_summaries.json`）。
    2.  它會從 `cluster_centers.npy` 中取出您指定的 Key（例如第 8 號向量）。
    3.  用這個 Key 去**查詢** `ChromaDB`，並要求資料庫回傳「最接近的 1 筆」結果。
* **驗證方法 (Verification):**
    * 執行 `python test_memory_db.py 8` (抽查第 8 號群組)。
    * **檢查終端機輸出：**
        * 您必須看到 `✅ ID 驗證成功！`
        * 您必須看到 `✅ 向量驗證成功！`
        * 您必須看到 `距離:` 是一個極小的數字（例如 `0.0` 或 `1.19e-07`），這代表資料庫 100% 準確地找到了向量本身。

---

#### C. (輔助) 查詢資料庫 Collection 名稱

* **執行腳本：** `list_collections.py`
* **作法 (Method):**
    * 如果您未來忘記了 `COLLECTION_NAME`（資料表名稱）叫什麼。
    * 執行 `python list_collections.py`。
* **輸出 (Output):**
    * 它會印出 `memory_db_chroma/` 資料庫中所有的 Collection 名稱（例如 `vlm_memory`）及其資料筆數。