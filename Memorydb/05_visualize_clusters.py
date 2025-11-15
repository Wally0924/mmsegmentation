import numpy as np
import json
import os
import shutil
from tqdm import tqdm
import warnings

# --- 1. 設定 ---
IMAGE_FOLDER = "data/training_images/"        # 您的原始影像來源資料夾
INPUT_LABELS_FILE = "cluster_labels.npy"      # 階段二的產出 (分群標籤)
INPUT_FILENAMES_FILE = "image_filenames.json" # 階段一的產出 (檔名索引)

# 這是「總輸出資料夾」的名稱
# 所有 K 個子資料夾都會被建立在 *裡面*
OUTPUT_BASE_DIR = "all_clusters_visualization"

# --- 2. 準備工作 ---
print(f"--- 階段五：視覺化所有 K-Means 群集 ---")

# 載入資料
try:
    labels = np.load(INPUT_LABELS_FILE)
    with open(INPUT_FILENAMES_FILE, 'r') as f:
        filenames = json.load(f)
except FileNotFoundError as e:
    print(f"[錯誤] 找不到必要的檔案: {e.filename}")
    print("請確保 cluster_labels.npy 和 image_filenames.json 都在此資料夾中。")
    exit()

# 檢查資料一致性
if len(labels) != len(filenames):
    print(f"[錯誤] 資料不匹配！標籤數量 ({len(labels)}) 與檔名數量 ({len(filenames)}) 不同。")
    exit()

# 從標籤中自動推算出 K 值 (例如：如果最大 ID 是 49，K 就是 50)
K_CLUSTERS = int(np.max(labels)) + 1
num_images = len(filenames)

print(f"成功載入 {num_images} 張影像的資料。")
print(f"偵測到 K = {K_CLUSTERS} 個群組。")

# --- 3. 建立 K 個空的子資料夾 ---
# 為了確保乾淨，如果總資料夾已存在，先刪除
if os.path.exists(OUTPUT_BASE_DIR):
    print(f"偵測到舊的資料夾 '{OUTPUT_BASE_DIR}'，正在將其刪除...")
    shutil.rmtree(OUTPUT_BASE_DIR)

print(f"正在建立新的總資料夾: '{OUTPUT_BASE_DIR}/'")
os.makedirs(OUTPUT_BASE_DIR)

print(f"正在 {OUTPUT_BASE_DIR}/ 中建立 {K_CLUSTERS} 個子資料夾...")
for i in range(K_CLUSTERS):
    # f"cluster_{i:02d}" 會格式化成 "cluster_00", "cluster_01"...
    # 這樣有助於在檔案總管中正確排序
    cluster_dir = os.path.join(OUTPUT_BASE_DIR, f"cluster_{i:02d}")
    os.makedirs(cluster_dir)

# --- 4. 遍歷所有影像並複製 (核心步驟) ---
print(f"正在將 {num_images} 張影像複製並分類到對應的資料夾...")

# 使用 tqdm 顯示進度條
for i in tqdm(range(num_images), desc="分類影像中"):
    try:
        # 獲取第 i 張圖的檔名和它被分配到的群組 ID
        filename = filenames[i]
        cluster_id = labels[i]
        
        # 1. 組合「來源路徑」
        src_path = os.path.join(IMAGE_FOLDER, filename)
        
        # 2. 組合「目標路徑」
        dst_path = os.path.join(OUTPUT_BASE_DIR, f"cluster_{cluster_id:02d}", filename)
        
        # 3. 執行複製
        shutil.copy(src_path, dst_path)
        
    except FileNotFoundError:
        print(f"\n[警告] 找不到來源影像: {src_path}，跳過此檔案。")
    except Exception as e:
        print(f"\n[警告] 複製檔案 {filename} 時出錯: {e}，跳過此檔案。")

# --- 5. 完成 ---
print("\n---")
print("✅ 階段五：視覺化群集 - 完成！ ---")
print(f"所有 {num_images} 張影像都已被複製並分類到：")
print(f"'{OUTPUT_BASE_DIR}/' 資料夾中的 {K_CLUSTERS} 個子資料夾內。")
print("\n請您現在手動打開該資料夾，瀏覽每個子資料夾以驗證分群結果。")