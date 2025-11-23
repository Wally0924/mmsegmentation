import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import os

# 忽略 sklearn 關於記憶體洩漏的良性警告
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.cluster._kmeans")

# --- 設定 (Settings) ---
INPUT_FEATURES_FILE = "all_joint_features.npy"   # 輸入：融合特徵 (Teacher)
OUTPUT_LABELS_FILE = "cluster_labels.npy"        # 輸出：最終分群標籤 (合併後)

# 1. 初始 K 值 (建議設大一點，讓 K-Means 先切細)
INITIAL_K = 7

# 2. 合併門檻 (相似度高於此值則合併)
# 建議：0.92 ~ 0.96。數值越高代表「非常非常像」才合併。
MERGE_THRESHOLD = 0.90

print(f"--- 階段 2：VLM 指導分群 (Initial K={INITIAL_K}, Merge Threshold={MERGE_THRESHOLD}) ---")

# --- 1. 載入特徵 ---
if not os.path.exists(INPUT_FEATURES_FILE):
    print(f"錯誤：找不到 {INPUT_FEATURES_FILE}。請先執行 01c。")
    exit()

print(f"正在載入特徵: {INPUT_FEATURES_FILE}...")
features = np.load(INPUT_FEATURES_FILE)

if features.shape[0] < INITIAL_K:
    print(f"錯誤：K 值 ({INITIAL_K}) 大於資料總數 ({features.shape[0]})。")
    exit()

# --- 2. 執行 K-Means (初始分群) ---
print("正在執行初始 K-Means...")
kmeans = KMeans(
    n_clusters=INITIAL_K,
    init="k-means++",
    n_init='auto',
    max_iter=100,
    random_state=42 
)
kmeans.fit(features)
labels = kmeans.labels_

print(f"初始分群完成，共 {INITIAL_K} 個群組。")

# --- 3. 後處理：自動合併相似群組 ---
print("\n--- 開始後處理：合併相似群組 ---")

while True:
    # A. 計算當前所有群組的中心點
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    if n_clusters < 2:
        break

    centroids = []
    valid_ids = []
    
    for label in unique_labels:
        # 找出該群組的所有向量
        mask = (labels == label)
        group_features = features[mask]
        
        # 計算平均中心
        mean_vec = np.mean(group_features, axis=0)
        # L2 標準化 (為了計算 Cosine Similarity)
        mean_vec = mean_vec / (np.linalg.norm(mean_vec) + 1e-9)
        
        centroids.append(mean_vec)
        valid_ids.append(label)
    
    centroids = np.array(centroids)
    
    # B. 計算相似度矩陣
    sim_matrix = cosine_similarity(centroids)
    np.fill_diagonal(sim_matrix, -1) # 忽略自己
    
    # C. 找出最相似的一對
    max_sim = np.max(sim_matrix)
    
    # 如果最高相似度低於門檻，停止合併
    if max_sim < MERGE_THRESHOLD:
        break
        
    # D. 執行合併
    idx_a, idx_b = np.unravel_index(np.argmax(sim_matrix), sim_matrix.shape)
    label_a = valid_ids[idx_a]
    label_b = valid_ids[idx_b]
    
    # 為了整潔，將 ID 較大的併入 ID 較小的
    target_id = min(label_a, label_b)
    source_id = max(label_a, label_b)
    
    print(f"  >> 合併: 群組 {source_id} -> 群組 {target_id} (相似度: {max_sim:.4f})")
    
    # 更新標籤
    labels[labels == source_id] = target_id

# --- 4. 重整 ID (Remap 0..N-1) ---
print("正在重整群組編號...")
unique_final = np.unique(labels)
final_labels = np.zeros_like(labels)

for new_id, old_id in enumerate(unique_final):
    final_labels[labels == old_id] = new_id

FINAL_K = len(unique_final)

# --- 5. 儲存結果 ---
print(f"\n合併完成！群組數量從 {INITIAL_K} 縮減為 {FINAL_K}。")
print(f"正在儲存最終標籤到 {OUTPUT_LABELS_FILE}...")
np.save(OUTPUT_LABELS_FILE, final_labels)

# --- 6. 摘要 ---
print("\n--- 分群摘要 ---")
counts = np.bincount(final_labels)
for i in range(FINAL_K):
    print(f"群組 {i:02d}: {counts[i]} 張影像")

print("\n階段 2 (分群+合併) - 完成！")