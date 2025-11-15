import chromadb
import numpy as np
import json
import os
import shutil

# --- 1. 設定 ---
INPUT_CENTERS_FILE = "cluster_centers.npy"
INPUT_SUMMARIES_FILE = "semantic_summaries.json"

CHROMA_DB_PATH = "memory_db_chroma"

COLLECTION_NAME = "vlm_memory"

# --- 2. 準備工作 ---
print(f"--- Phase 4: Building and Populating Vector Database ---")

if os.path.exists(CHROMA_DB_PATH):
    print(f"Detected old database '{CHROMA_DB_PATH}', deleting it...")
    shutil.rmtree(CHROMA_DB_PATH)

# --- 3. 載入 Keys 和 Values ---
print(f"Loading Keys ({INPUT_CENTERS_FILE})...")
try:
    cluster_centers = np.load(INPUT_CENTERS_FILE)
except FileNotFoundError:
    print(f"ERROR: File '{INPUT_CENTERS_FILE}' not found!")
    print("Please run script 02_run_clustering.py first.")
    exit()

print(f"Loading Values ({INPUT_SUMMARIES_FILE})...")
try:
    with open(INPUT_SUMMARIES_FILE, 'r', encoding='utf-8') as f:
        summaries = json.load(f)
except FileNotFoundError:
    print(f"ERROR: File '{INPUT_SUMMARIES_FILE}' not found!")
    print("Please run script 03_calculate_probabilities.py first.")
    exit()

K_CLUSTERS = cluster_centers.shape[0]
if K_CLUSTERS != len(summaries):
    print(f"ERROR: Key count ({K_CLUSTERS}) does not match Value count ({len(summaries)})!")
    print("Please check K_CLUSTERS setting in scripts 02 and 03.")
    exit()

print(f"Successfully loaded {K_CLUSTERS} Key-Value pairs.")

# --- 4. 初始化並填入 ChromaDB ---
print(f"Initializing Vector Database (at '{CHROMA_DB_PATH}')...")

client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)

print(f"Adding {K_CLUSTERS} items to database...")

embeddings_list = []
documents_list = []
ids_list = []

for cluster_id in range(K_CLUSTERS):
    key_vector = cluster_centers[cluster_id].tolist()
    # (註：這裡的 value_text 現在是「機率 JSON 字串」)
    value_text = summaries[str(cluster_id)] 
    
    item_id = f"cluster_{cluster_id}"
    
    embeddings_list.append(key_vector)
    documents_list.append(value_text)
    ids_list.append(item_id)

collection.add(
    embeddings=embeddings_list,
    documents=documents_list,
    ids=ids_list
)

# --- 5. 完成 ---
print(f"\nDatabase population complete!")
db_count = collection.count()
print(f"Verification: Database '{COLLECTION_NAME}' now contains {db_count} items.")
print(f"Your final artifact (the database) is stored in the '{CHROMA_DB_PATH}/' directory.")