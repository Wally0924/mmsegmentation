import numpy as np
from sklearn.cluster import KMeans
import warnings

# 忽略 scikit-learn 關於記憶體洩漏的良性警告
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.cluster._kmeans")

# --- 1. 設定 ---
INPUT_FEATURES_FILE = "all_joint_features.npy"       # 階段一的特徵檔案
OUTPUT_LABELS_FILE = "cluster_labels.npy"      # 輸出：每張圖片的群組 ID
OUTPUT_CENTERS_FILE = "cluster_centers.npy"    # 輸出：K 個群組的中心向量 (記憶庫的 Key)

K_CLUSTERS = 25

# --------------------

print(f"--- 階段二：執行 K-Means 群集 ---")
print(f"目標群組數量 K = {K_CLUSTERS}")

# --- 2. 載入特徵向量 ---
print(f"正在載入特徵檔案: {INPUT_FEATURES_FILE}...")
try:
    features = np.load(INPUT_FEATURES_FILE)
except FileNotFoundError:
    print(f"錯誤：找不到 {INPUT_FEATURES_FILE} 檔案！")
    print("請先執行 01_extract_features.py 腳本。")
    exit()

if features.shape[0] < K_CLUSTERS:
    print(f"錯誤：K 值 ({K_CLUSTERS}) 大於總影像數量 ({features.shape[0]})。")
    print("請降低 K_CLUSTERS 的值。")
    exit()

print(f"成功載入 {features.shape[0]} 個特徵 (維度: {features.shape[1]})。")

# --- 3. 執行 K-Means 演算法 ---
print(f"正在對 {features.shape[0]} 個向量執行 K-Means (K={K_CLUSTERS})...")

kmeans = KMeans(
    n_clusters=K_CLUSTERS,  # 您設定的 K 值
    init="k-means++",       # 聰明的初始化方法，能加速收斂
    n_init='auto',          # 'auto' 是 scikit-learn 1.4 版後推薦的預設值
    max_iter=300,           # 每次執行的最大迭代次數
    random_state=42         # 設為固定數字 (42) 確保每次執行的結果都一樣
)

# .fit() 會執行所有計算
kmeans.fit(features)

print("K-Means 計算完成！")

# --- 4. 儲存結果 ---
# kmeans.labels_：一個 1D 陣列 (長度 387)，
#                 其值為 0 到 K-1，代表每張圖片被分到的群組 ID
labels = kmeans.labels_

# kmeans.cluster_centers_：一個 2D 陣列 (形狀 K x 768)，
#                           代表 K 個群組的「中心向量」
centers = kmeans.cluster_centers_

print(f"正在儲存群組標籤 (Labels) 到 {OUTPUT_LABELS_FILE}...")
np.save(OUTPUT_LABELS_FILE, labels)

print(f"正在儲存群組中心 (Centers) 到 {OUTPUT_CENTERS_FILE}...")
np.save(OUTPUT_CENTERS_FILE, centers)

# --- 5. (額外) 顯示群集摘要 ---
# 這非常有用，可以讓您知道每個群組分到了多少張圖片
print("\n--- 群集結果摘要 ---")
# np.bincount 會計算 0, 1, 2... 各出現了幾次
cluster_counts = np.bincount(labels)

for i in range(K_CLUSTERS):
    print(f"群組 {i:02d} (地點 {i:02d}): 分配到 {cluster_counts[i]} 張影像")

print("\n---")
print("階段二：K-Means 群集 - 完成！")
print(f"您的群組標籤已儲存在: {OUTPUT_LABELS_FILE}")
print(f"您的群組中心向量已儲存在: {OUTPUT_CENTERS_FILE}")