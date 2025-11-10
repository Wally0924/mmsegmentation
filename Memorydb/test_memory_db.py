import chromadb
import numpy as np
import json
import sys
import os

# --- 1. 設定 ---
CHROMA_DB_PATH = "memory_db_chroma"        # 您的資料庫資料夾
COLLECTION_NAME = "vlm_memory"           # 您的 Collection (資料表) 名稱

# 我們需要原始檔案來交叉比對 (Ground Truth)
INPUT_CENTERS_FILE = "cluster_centers.npy"    # "標準答案" Keys
INPUT_SUMMARIES_FILE = "semantic_summaries.json"  # "標準答案" Values

# --- 2. 獲取要測試的 Cluster ID ---
if len(sys.argv) != 2:
    print("\n[錯誤] 請提供一個要測試的 Cluster ID (0 到 K-1 之間的數字)。")
    print("   用法: python test_memory_db.py <cluster_id>")
    print("   範例: python test_memory_db.py 5")
    sys.exit(1)

try:
    CLUSTER_ID_TO_TEST = int(sys.argv[1])
except ValueError:
    print("\n[錯誤] Cluster ID 必須是一個數字。")
    sys.exit(1)

print(f"--- 正在測試從記憶庫中「檢索」 Cluster {CLUSTER_ID_TO_TEST} ---")

# --- 3. 載入資料庫和「標準答案」---
try:
    # 載入 ChromaDB
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)
    
    # 載入「標準答案」
    cluster_centers = np.load(INPUT_CENTERS_FILE)
    with open(INPUT_SUMMARIES_FILE, 'r', encoding='utf-8') as f:
        summaries = json.load(f)
        
except Exception as e:
    print(f"\n[錯誤] 載入資料庫或檔案失敗: {e}")
    print(f"請確保 '{CHROMA_DB_PATH}' 資料夾存在，且 npy/json 檔案也都在。")
    sys.exit(1)

# --- 4. 執行「冒煙測試」 (Smoke Test) ---
db_count = collection.count()
k_count = cluster_centers.shape[0]

print(f"資料庫總數驗證：資料庫回報 {db_count} 筆資料，K 值為 {k_count}。")
if db_count != k_count:
    print(f"[警告] 數量不匹配！請檢查 04 腳本是否出錯。")
if CLUSTER_ID_TO_TEST >= db_count:
    print(f"[錯誤] 您要測試的 ID {CLUSTER_ID_TO_TEST} 超出了範圍 (0-{db_count-1})。")
    sys.exit(1)

# --- 5. 準備「查詢向量」(Query Vector) ---
# 我們直接從「標準答案」中取出第 N 個向量
# 這就是我們用來查詢的「Key」
query_vector = cluster_centers[CLUSTER_ID_TO_TEST].tolist()

# 找出「標準答案」的 Value (摘要文字)
expected_id_str = str(CLUSTER_ID_TO_TEST)
expected_text = summaries[expected_id_str]
expected_db_id = f"cluster_{expected_id_str}" # 04 腳本中儲存的 ID 格式

print(f"\n【標準答案】 (來自 .json 檔案):")
print(f"  ID: {expected_id_str}")
print(f"  摘要: {expected_text}")

# --- 6. 執行資料庫查詢 ---
print("\n... 正在用「標準答案 Key」查詢資料庫 ...")

# 關鍵步驟：
# query_embeddings: 我們要查詢的向量
# n_results=1:      請回傳「最接近的 1 筆」結果
results = collection.query(
    query_embeddings=[query_vector],
    n_results=1
)

# --- 7. 驗證結果 ---
try:
    # 'results' 是一個複雜的字典，我們把它解開
    returned_id = results['ids'][0][0]
    returned_text = results['documents'][0][0]
    returned_distance = results['distances'][0][0]
    
    print("\n【資料庫回傳】 (來自 ChromaDB):")
    print(f"  ID: {returned_id}")
    print(f"  摘要: {returned_text}")
    print(f"  距離: {returned_distance:.8f}") # 顯示到小數點後 8 位

    print("\n--- 最終驗證 ---")
    
    # 驗證 1: ID 是否匹配?
    if returned_id == expected_db_id:
        print(f"✅ ID 驗證成功！ (預期: {expected_db_id}, 收到: {returned_id})")
    else:
        print(f"❌ ID 驗證失敗！ (預期: {expected_db_id}, 收到: {returned_id})")

    # 驗證 2: 距離是否為 0?
    # (因為是用向量本身去查，距離應該要*幾乎*為 0)
    if returned_distance < 0.0001:
        print(f"✅ 向量驗證成功！ (距離 {returned_distance} 接近 0)")
    else:
        print(f"❌ 向量驗證失敗！ (距離 {returned_distance} 不為 0)")
        
    # 驗證 3: 摘要文字是否匹配? (非必要，但建議)
    if returned_text == expected_text:
        print(f"✅ 文本驗證成功！")
    else:
        print(f"❌ 文本驗證失敗！ (資料庫中的摘要與 .json 檔案不符)")

except Exception as e:
    print(f"\n[錯誤] 解析查詢結果失敗: {e}")
    print("資料庫可能是空的或查詢失敗。")