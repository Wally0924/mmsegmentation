import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt # 我們需要畫圖

# 忽略 scikit-learn 關於記憶體洩漏的良性警告
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.cluster._kmeans")

# --- 1. Settings ---
INPUT_FEATURES_FILE = "all_joint_features.npy"

# --- 2. 設定您要測試的 K 值範圍 ---
K_RANGE_START = 10
K_RANGE_END = 80
K_RANGE_STEP = 1 

# --------------------
print(f"--- 02b: Finding Best K for {INPUT_FEATURES_FILE} ---")

# --- 3. Load Features ---
print(f"Loading features from {INPUT_FEATURES_FILE}...")
try:
    features = np.load(INPUT_FEATURES_FILE)
except FileNotFoundError:
    print(f"ERROR: File '{INPUT_FEATURES_FILE}' not found!")
    print("Please run script 01c_extract_text_features.py first.")
    exit()

print(f"Features loaded. Shape: {features.shape}")

# --- 4. Loop through K values and calculate scores ---
k_values = list(range(K_RANGE_START, K_RANGE_END, K_RANGE_STEP))
inertia_scores = [] # 儲存「手肘法」的分數
silhouette_scores = [] # 儲存「輪廓係數」的分數

print(f"Calculating Inertia and Silhouette scores for K in {k_values}...")
print("NOTE: Silhouette score calculation can be VERY SLOW.")

for k in tqdm(k_values, desc="Testing K values"):
    kmeans = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init='auto',
        max_iter=100,
        random_state=42 # 保持一致性
    )
    
    # 執行 K-Means
    labels = kmeans.fit_predict(features)
    
    # 1. 計算 Inertia (手肘法)
    inertia = kmeans.inertia_
    inertia_scores.append(inertia)
    
    # 2. 計算 Silhouette Score (輪廓係數)
    score = silhouette_score(features, labels, metric='cosine') # 使用餘弦距離
    silhouette_scores.append(score)
    
    tqdm.write(f"K = {k:02d} | Inertia: {inertia:.2f} | Silhouette Score: {score:.4f}")

# --- 5. Print Best K ---
best_k_index = np.argmax(silhouette_scores) # 找出最高分的「索引」
best_k = k_values[best_k_index]             # 透過索引找出 K 值
best_score = silhouette_scores[best_k_index]

print("\n---")
print("--- Calculation Complete ---")
print(f"📈 Best K based on Silhouette Score: K = {best_k} (Score: {best_score:.4f})")
print("---")

# --- 6. Plot the results ---
print("Generating plots (elbow_plot.png, silhouette_plot.png)...")

# 繪製手肘法
plt.figure(figsize=(10, 5))
plt.plot(k_values, inertia_scores, 'bo-')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Inertia (Sum of squared distances)')
plt.title('Elbow Method for Optimal K')
plt.grid(True)
plt.savefig('elbow_plot.png')

# 繪製輪廓係數
plt.figure(figsize=(10, 5))
plt.plot(k_values, silhouette_scores, 'ro-')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Optimal K')
plt.grid(True)
plt.savefig('silhouette_plot.png')

print("Plots saved. Please check 'elbow_plot.png' and 'silhouette_plot.png'.")
print(f"RECOMMENDATION: Use K = {best_k} in your 02_run_clustering.py script.")