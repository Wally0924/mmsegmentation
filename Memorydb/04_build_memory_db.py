import chromadb
import numpy as np
import json
import os
import shutil

# --- 1. 設定 ---
INPUT_CENTERS_FILE = "cluster_centers.npy"    # 記憶庫的 "Key" (K x 768 維向量)
INPUT_SUMMARIES_FILE = "semantic_summaries.json"  # 記憶庫的 "Value" (K 句摘要)

# 這是我們「最終產出」的資料庫資料夾名稱
CHROMA_DB_PATH = "memory_db_chroma"

# 向量資料庫中 "Collection" (類似資料表) 的名稱
COLLECTION_NAME = "vlm_memory"

# --- 2. 準備工作 ---
print(f"--- 階段四：建置並儲存向量記憶庫 ---")

# 安全檢查：如果舊的資料庫資料夾已存在，先刪除它
# 這樣能確保我們每次都是從「全新」的狀態開始建置
if os.path.exists(CHROMA_DB_PATH):
    print(f"偵測到舊的資料庫 '{CHROMA_DB_PATH}'，正在將其刪除...")
    shutil.rmtree(CHROMA_DB_PATH)

# --- 3. 載入 Keys 和 Values ---
print(f"正在載入 Keys ({INPUT_CENTERS_FILE})...")
try:
    cluster_centers = np.load(INPUT_CENTERS_FILE)
except FileNotFoundError:
    print(f"錯誤：找不到 {INPUT_CENTERS_FILE}！")
    print("請先執行 02_run_clustering.py 腳本。")
    exit()

print(f"正在載入 Values ({INPUT_SUMMARIES_FILE})...")
try:
    with open(INPUT_SUMMARIES_FILE, 'r', encoding='utf-8') as f:
        summaries = json.load(f)
except FileNotFoundError:
    print(f"錯誤：找不到 {INPUT_SUMMARIES_FILE}！")
    print("請先執行 03_generate_summaries.py 腳本。")
    exit()

# 獲取 K 值 (群組總數)，並驗證 Keys 和 Values 數量是否一致
K_CLUSTERS = cluster_centers.shape[0]
if K_CLUSTERS != len(summaries):
    print(f"錯誤：Key 數量 ({K_CLUSTERS}) 與 Value 數量 ({len(summaries)}) 不匹配！")
    print("請檢查 02 和 03 腳本中的 K_CLUSTERS 設定是否一致。")
    exit()

print(f"成功載入 {K_CLUSTERS} 筆 Key-Value 對。")

# --- 4. 初始化並填入 ChromaDB ---
print(f"正在初始化向量資料庫 (儲存於 '{CHROMA_DB_PATH}')...")

# PersistentClient 會將資料庫「永久儲存」到硬碟上的資料夾
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

# 建立一個 Collection (資料表)
# 我們還指定了 metadata={"hnsw:space": "cosine"}
# 這告訴 ChromaDB 我們要使用「餘弦相似度 (cosine similarity)」來計算向量距離
# 這是 CLIP 向量最推薦的距離度量
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)

print(f"正在將 {K_CLUSTERS} 筆資料填入資料庫...")

# 為了讓 ChromaDB 能批次處理，我們需要準備好三個列表：
embeddings_list = []  # 儲存所有 Key (向量)
documents_list = []   # 儲存所有 Value (摘要文字)
ids_list = []         # 儲存每個資料點的唯一 ID (我們用群組 ID)

for cluster_id in range(K_CLUSTERS):
    key_vector = cluster_centers[cluster_id].tolist()
    value_text = summaries[str(cluster_id)]
    
    # ChromaDB 需要一個唯一的字串 ID
    item_id = f"cluster_{cluster_id}"
    
    embeddings_list.append(key_vector)
    documents_list.append(value_text)
    ids_list.append(item_id)

# 使用 .add() 批次將所有資料一次性加入資料庫，效能最好
collection.add(
    embeddings=embeddings_list,
    documents=documents_list,
    ids=ids_list
)

# --- 5. 完成 ---
print(f"\n資料庫填充完畢！")
db_count = collection.count()
print(f"驗證：資料庫 '{COLLECTION_NAME}' 中現有 {db_count} 筆資料。")