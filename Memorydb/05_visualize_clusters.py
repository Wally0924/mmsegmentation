import numpy as np
import json
import os
import shutil
from tqdm import tqdm
import warnings

# --- 1. 設定 ---
IMAGE_FOLDER = "data/training_images/"        # 您的原始影像根目錄
INPUT_LABELS_FILE = "cluster_labels.npy"      # 階段二的產出 (分群標籤)
INPUT_FILENAMES_FILE = "image_filenames.json" # 階段一的產出 (含相對路徑的檔名索引)

# 輸出資料夾
OUTPUT_BASE_DIR = "all_clusters_visualization"

# --- 2. 準備工作 ---
print(f"--- 階段五：視覺化所有 K-Means 群集 ---")

# 載入資料
try:
    labels = np.load(INPUT_LABELS_FILE)
    with open(INPUT_FILENAMES_FILE, 'r') as f:
        filenames = json.load(f) # 這裡載入的是相對路徑，如 "sunny/morning/001.jpg"
except FileNotFoundError as e:
    print(f"[錯誤] 找不到必要的檔案: {e.filename}")
    print("請確保 cluster_labels.npy 和 image_filenames.json 都在此資料夾中。")
    exit()

# 檢查資料一致性
if len(labels) != len(filenames):
    print(f"[錯誤] 資料不匹配！標籤數量 ({len(labels)}) 與檔名數量 ({len(filenames)}) 不同。")
    exit()

K_CLUSTERS = int(np.max(labels)) + 1
num_images = len(filenames)

print(f"成功載入 {num_images} 張影像的資料。")
print(f"偵測到 K = {K_CLUSTERS} 個群組。")

# --- 3. 建立 K 個空的子資料夾 ---
if os.path.exists(OUTPUT_BASE_DIR):
    print(f"偵測到舊的資料夾 '{OUTPUT_BASE_DIR}'，正在將其刪除以確保乾淨...")
    shutil.rmtree(OUTPUT_BASE_DIR)

print(f"正在建立新的總資料夾: '{OUTPUT_BASE_DIR}/'")
os.makedirs(OUTPUT_BASE_DIR)

print(f"正在建立子資料夾...")
for i in range(K_CLUSTERS):
    cluster_dir = os.path.join(OUTPUT_BASE_DIR, f"cluster_{i:02d}")
    os.makedirs(cluster_dir)

# --- 4. 遍歷所有影像並複製 (核心修改) ---
print(f"正在將 {num_images} 張影像複製並分類...")

success_count = 0
fail_count = 0

for i in tqdm(range(num_images), desc="分類影像中"):
    try:
        # filename 是相對路徑，例如 "sunny/morning/001.jpg"
        rel_filename = filenames[i]
        cluster_id = labels[i]
        
        # 1. 組合正確的「來源路徑」
        # 使用 os.path.join 自動處理系統路徑分隔符號
        # 如果 JSON 裡是用 / 但系統是 Windows，這裡可能需要 replace
        clean_rel_filename = rel_filename.replace("/", os.sep).replace("\\", os.sep)
        src_path = os.path.join(IMAGE_FOLDER, clean_rel_filename)
        
        # 2. 組合「目標路徑」 (關鍵修改：扁平化)
        # 我們不希望在目標資料夾建立複雜的樹狀結構，所以將路徑攤平
        # "sunny/morning/001.jpg" -> "sunny_morning_001.jpg"
        flat_filename = rel_filename.replace("/", "_").replace("\\", "_")
        
        # 為了避免檔名太長或有奇怪符號，這是個安全的做法
        dst_path = os.path.join(OUTPUT_BASE_DIR, f"cluster_{cluster_id:02d}", flat_filename)
        
        # 3. 執行複製
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
            success_count += 1
        else:
            # 嘗試 debug：有時候是因為路徑拼接問題
            tqdm.write(f"[警告] 找不到來源影像: {src_path}")
            fail_count += 1
        
    except Exception as e:
        tqdm.write(f"[錯誤] 複製 {rel_filename} 時失敗: {e}")
        fail_count += 1

# --- 5. 完成 ---
print("\n---")
print("✅ 階段五：視覺化群集 - 完成！")
print(f"成功複製: {success_count} 張, 失敗: {fail_count} 張")
print(f"結果已儲存至 '{OUTPUT_BASE_DIR}/'。")
print("提示：檔名已包含原始資料夾資訊 (例如 'sunny_morning_001.jpg')，方便您識別來源。")