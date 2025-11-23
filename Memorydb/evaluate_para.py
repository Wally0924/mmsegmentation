import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm
import warnings

# 忽略 scikit-learn 的記憶體警告
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.cluster._kmeans")

# --- 1. 設定 (Configuration) ---
IMAGE_FEATURES_FILE = "all_image_features.npy"
TEXT_FEATURES_FILE = "all_text_features.npy"

# 權重測試範圍：從 0.0 到 1.0，間隔 0.1
# 例如: [0.0, 0.1, ..., 0.9, 1.0]
WEIGHT_step = 0.1
WEIGHT_RANGE = np.arange(0.0, 1.01, WEIGHT_step)

# K 值測試範圍：為了節省時間，建議步長設大一點，或者範圍縮小
# 我們要看的是「特徵的好壞」，通常取一個合理的 K 範圍即可
K_MIN = 10
K_MAX = 50
K_STEP = 5
K_VALUES = range(K_MIN, K_MAX + 1, K_STEP)

# 隨機種子 (確保結果可重現)
RANDOM_STATE = 42

def load_features(img_path: str, txt_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    載入原始的圖像與文字特徵。
    """
    if not os.path.exists(img_path) or not os.path.exists(txt_path):
        raise FileNotFoundError("找不到特徵檔案，請檢查路徑。")
    
    print(f"Loading features...")
    img = np.load(img_path)
    txt = np.load(txt_path)
    print(f"Image Features: {img.shape}, Text Features: {txt.shape}")
    
    # 確保樣本數一致
    assert img.shape[0] == txt.shape[0], "圖像與文字特徵的樣本數量不一致！"
    
    return img, txt

def get_weighted_features(img: np.ndarray, txt: np.ndarray, w_img: float, w_txt: float) -> np.ndarray:
    """
    在記憶體中動態生成加權拼接特徵。
    """
    # 1. 應用權重
    weighted_img = img * w_img
    weighted_txt = txt * w_txt
    
    # 2. 拼接 (Concatenate)
    joint_feats = np.concatenate([weighted_img, weighted_txt], axis=1)
    
    return joint_feats

def evaluate_single_configuration(features: np.ndarray, k_values: list) -> float:
    """
    給定一組特徵，測試不同的 K 值，回傳該特徵下「最高的」Silhouette Score。
    這代表了這組特徵在最佳狀態下的表現能力。
    """
    best_score_for_this_weight = -1
    
    # 針對這組特徵，跑幾個 K 看看哪個最好
    # 這裡我們不使用 tqdm 顯示內層迴圈，以免輸出太亂，只在外部顯示進度
    for k in k_values:
        kmeans = KMeans(
            n_clusters=k, 
            init="k-means++", 
            n_init='auto', 
            max_iter=100, 
            random_state=RANDOM_STATE
        )
        labels = kmeans.fit_predict(features)
        
        # 計算輪廓係數 (分數越高越好: -1 到 1)
        # metric='cosine' 適用於高維向量
        score = silhouette_score(features, labels, metric='cosine')
        
        if score > best_score_for_this_weight:
            best_score_for_this_weight = score
            
    return best_score_for_this_weight

def main():
    print(f"--- 02_evaluate_weights.py: 尋找最佳特徵融合權重 ---")
    
    # 1. 載入資料
    try:
        img_feats, txt_feats = load_features(IMAGE_FEATURES_FILE, TEXT_FEATURES_FILE)
    except Exception as e:
        print(e)
        return

    results = [] # 儲存結果: (img_weight, txt_weight, best_score)
    
    print(f"\n開始網格搜索 (Grid Search)...")
    print(f"測試權重數量: {len(WEIGHT_RANGE)}")
    print(f"每個權重測試 K 值範圍: {list(K_VALUES)}")
    
    # 2. 遍歷權重
    pbar = tqdm(WEIGHT_RANGE, desc="Evaluating Weights")
    
    for w_img in pbar:
        w_txt = 1.0 - w_img # 確保相加為 1 (也可以不為1，但在拼接邏輯下，相對比例最重要)
        
        # A. 生成特徵
        joint_feats = get_weighted_features(img_feats, txt_feats, w_img, w_txt)
        
        # B. 評估這組特徵的好壞 (找出它的最佳 K 對應的分數)
        max_score = evaluate_single_configuration(joint_feats, K_VALUES)
        
        results.append({
            "img_weight": w_img,
            "txt_weight": w_txt,
            "score": max_score
        })
        
        # 更新進度條資訊
        pbar.set_postfix({"ImgW": f"{w_img:.1f}", "BestScore": f"{max_score:.4f}"})

    # 3. 找出最佳結果
    # 根據 score 排序
    results.sort(key=lambda x: x["score"], reverse=True)
    best_result = results[0]
    
    print("\n" + "="*40)
    print("🏆 最佳權重組合結果")
    print("="*40)
    print(f"最佳圖像權重 (Image Weight): {best_result['img_weight']:.2f}")
    print(f"最佳文字權重 (Text Weight) : {best_result['txt_weight']:.2f}")
    print(f"最高輪廓係數 (Silhouette)  : {best_result['score']:.4f}")
    print("-" * 40)
    print("Top 3 組合:")
    for i, res in enumerate(results[:3]):
        print(f"{i+1}. Img: {res['img_weight']:.1f}, Txt: {res['txt_weight']:.1f} -> Score: {res['score']:.4f}")

    # 4. 繪圖 (視覺化證明)
    x_weights = [r['img_weight'] for r in sorted(results, key=lambda x: x['img_weight'])]
    y_scores = [r['score'] for r in sorted(results, key=lambda x: x['img_weight'])]

    plt.figure(figsize=(10, 6))
    plt.plot(x_weights, y_scores, marker='o', linestyle='-', color='b', linewidth=2, markersize=8)
    
    # 標記最高點
    plt.plot(best_result['img_weight'], best_result['score'], 'r*', markersize=15, label='Best Weight')
    
    plt.title('Feature Fusion Weights vs. Clustering Quality\n(Higher Silhouette Score is Better)')
    plt.xlabel('Image Feature Weight (0.0 = Text Only, 1.0 = Image Only)')
    plt.ylabel('Best Silhouette Score (across tested K)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plot_filename = "weight_evaluation_plot.png"
    plt.savefig(plot_filename)
    print(f"\n📊 結果圖表已儲存至: {plot_filename}")
    print("請查看此圖表以決定要在 01c_merge_feature.py 中使用的權重。")

if __name__ == "__main__":
    main()