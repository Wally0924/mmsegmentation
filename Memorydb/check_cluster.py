import numpy as np
import json
import os
import shutil
import sys

# --- 1. 設定 ---
# (這些檔案應該都存在於同一個資料夾)
IMAGE_FOLDER = "data/training_images/"
INPUT_LABELS_FILE = "cluster_labels.npy"
INPUT_FILENAMES_FILE = "image_filenames.json"
INPUT_SUMMARIES_FILE = "semantic_summaries.json"

# --- 2. 檢查並獲取要抽查的 Cluster ID ---
if len(sys.argv) != 2:
    print("\n[錯誤] 請提供一個要抽查的 Cluster ID。")
    print("   用法: python check_cluster.py <cluster_id>")
    print("   範例: python check_cluster.py 5")
    sys.exit(1)

try:
    CLUSTER_ID_TO_CHECK = int(sys.argv[1])
except ValueError:
    print("\n[錯誤] Cluster ID 必須是一個數字。")
    print("   範例: python check_cluster.py 5")
    sys.exit(1)

print(f"--- 正在抽查 Cluster {CLUSTER_ID_TO_CHECK} ---")

# --- 3. 載入所有必要的檔案 ---
try:
    with open(INPUT_SUMMARIES_FILE, 'r', encoding='utf-8') as f:
        summaries = json.load(f)
    with open(INPUT_FILENAMES_FILE, 'r') as f:
        filenames = json.load(f)
    labels = np.load(INPUT_LABELS_FILE)
except FileNotFoundError as e:
    print(f"\n[錯誤] 找不到必要的檔案: {e.filename}")
    print("請確保所有 .json 和 .npy 檔案都在此資料夾中。")
    sys.exit(1)

# --- 4. 顯示「語意摘要 (Value)」---
summary_key = str(CLUSTER_ID_TO_CHECK)
if summary_key not in summaries:
    print(f"[錯誤] 在 {INPUT_SUMMARIES_FILE} 中找不到 Cluster ID '{summary_key}'。")
    print(f"請確認您的 K 值是否正確。")
    sys.exit(1)

print("\n【LLaVA 摘要 (Value)】:")
print("="*30)
print(summaries[summary_key])
print("="*30)

# --- 5. 找出所有屬於此群組的圖片 ---
indices = np.where(labels == CLUSTER_ID_TO_CHECK)[0]

if len(indices) == 0:
    print("\n[結果] 這個群組沒有分配到任何影像。")
    sys.exit(0)

print(f"\n【K-Means 結果】: 此群組共包含 {len(indices)} 張影像。")

# --- 6. 建立驗證資料夾並複製圖片 ---
VERIFY_FOLDER = f"verification_cluster_{CLUSTER_ID_TO_CHECK}"
if os.path.exists(VERIFY_FOLDER):
    # 如果資料夾已存在，先刪除舊的
    shutil.rmtree(VERIFY_FOLDER)
os.makedirs(VERIFY_FOLDER)

print(f"正在將這 {len(indices)} 張影像複製到 '{VERIFY_FOLDER}/' ...")

cluster_filenames = []
for idx in indices:
    try:
        filename = filenames[idx]
        cluster_filenames.append(filename)
        
        # 組合來源路徑和目標路徑
        src_path = os.path.join(IMAGE_FOLDER, filename)
        dst_path = os.path.join(VERIFY_FOLDER, filename)
        
        # 複製檔案
        shutil.copy(src_path, dst_path)
    except Exception as e:
        print(f"警告：複製檔案 {filename} 失敗, {e}")

print("\n--- 驗證腳本完成 ---")
print(f"請打開新的資料夾 '{VERIFY_FOLDER}/'，")
print(f"並「人工比對」裡面的 {len(cluster_filenames)} 張圖片是否與上面的「LLaVA 摘要」相符。")

