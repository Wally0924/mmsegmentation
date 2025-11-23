import numpy as np
import os

# --- 設定 (Settings) ---
# (輸入 1) 聰明的老師分好的「標準答案」 (來自 02_run_clustering.py)
INPUT_LABELS_FILE = "cluster_labels.npy"

# (輸入 2) 快速的學生「影像特徵」 (來自 01_extract_feature.py)
# 這是 DINOv2 的 768 維向量
INPUT_IMAGE_FEATURES_FILE = "all_image_features.npy"

# (輸出) 我們要產生的「快速 Key」 (供 04_build_memory_db.py 使用)
OUTPUT_CENTERS_FILE = "image_only_cluster_centers.npy"

print(f"--- 階段 2b：計算純影像中心點 (Fast Key) ---")

# --- 1. 載入資料 ---
print("正在載入資料...")
if not os.path.exists(INPUT_LABELS_FILE) or not os.path.exists(INPUT_IMAGE_FEATURES_FILE):
    print("錯誤：找不到檔案。請確認階段 1a 和 2 已完成。")
    exit()

try:
    labels = np.load(INPUT_LABELS_FILE)
    img_features = np.load(INPUT_IMAGE_FEATURES_FILE)
except Exception as e:
    print(f"載入錯誤: {e}")
    exit()

# 檢查資料一致性
if labels.shape[0] != img_features.shape[0]:
    print(f"錯誤：標籤數量 ({labels.shape[0]}) 與影像特徵數量 ({img_features.shape[0]}) 不符！")
    exit()

# 推算 K 值
# 假設 cluster ID 是連續的 0 ~ K-1
K_CLUSTERS = int(np.max(labels)) + 1
feature_dim = img_features.shape[1] # 預期 768

print(f"偵測到 K={K_CLUSTERS}, 特徵維度={feature_dim}")

# --- 2. 計算平均中心點 ---
print("正在計算每個群組的平均向量...")

# 初始化一個空的矩陣來存 K 個中心點
new_centers = np.zeros((K_CLUSTERS, feature_dim), dtype=np.float32)

for k in range(K_CLUSTERS):
    # 找出所有被分到第 k 群的影像索引
    indices = np.where(labels == k)[0]
    
    # (防呆) 如果某一群是空的
    if len(indices) == 0:
        print(f"警告：群組 {k} 是空的！")
        continue
        
    # 取出這些影像的 768 維向量
    cluster_vectors = img_features[indices]
    
    # 計算平均值 (Mean)
    # 這代表了這個地點「視覺上」的平均長相
    mean_vector = np.mean(cluster_vectors, axis=0)
    
    # (重要) L2 標準化 (Normalization)
    # 確保 Key 是單位向量，這對於 Cosine Similarity 搜尋至關重要
    norm = np.linalg.norm(mean_vector)
    if norm > 0:
        mean_vector = mean_vector / norm
    
    new_centers[k] = mean_vector
    # print(f"群組 {k:02d}: 平均了 {len(indices)} 張影像")

# --- 3. 儲存結果 ---
print(f"正在儲存快速 Key 到 {OUTPUT_CENTERS_FILE}...")
np.save(OUTPUT_CENTERS_FILE, new_centers)

print("階段 2b (計算 Key) - 完成！")
print(f"您的快速 Key 已儲存在: {OUTPUT_CENTERS_FILE}")
