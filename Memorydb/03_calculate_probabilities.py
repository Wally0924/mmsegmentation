import numpy as np
import json
import os
from collections import Counter
from tqdm import tqdm
import warnings

# --- 1. 設定 ---
INPUT_LABELS_FILE = "cluster_labels.npy"
INPUT_FILENAMES_FILE = "image_filenames.json"
INPUT_SUMMARIES_FILE = "llava_summaries.json"
OUTPUT_SUMMARIES_FILE = "semantic_summaries.json"

# 機率門檻：只有出現機率高於此值的特徵才會被保留
PROBABILITY_THRESHOLD = 0.4

# [新增] 正規化設定：定義要移除的形容詞前綴
# 這些詞通常描述「大小」或「程度」，容易導致相同的物件被統計成不同的標籤
REMOVE_PREFIXES = [
    "massive_", "large_", "tall_", "small_", "long_", "short_", 
    "huge_", "giant_", "tiny_", "wide_", "narrow_", "prominent_"
]

warnings.filterwarnings("ignore", message="invalid escape sequence")

def normalize_object_label(label: str) -> str:
    """
    將物件標籤正規化，以合併相似的統計項目。
    功能：
    1. 轉小寫 (Lowercasing)
    2. 移除冗餘前綴 (Prefix Stripping)
    
    範例: 
    - "Massive_Beige_Building" -> "beige_building"
    - "tall_street_lamp" -> "street_lamp"
    """
    if not label:
        return ""
        
    label = label.lower().strip()
    
    # 循環移除前綴，直到沒有匹配為止 
    # (可以處理多重修飾，例如 "massive_tall_building" -> "building")
    while True:
        changed = False
        for prefix in REMOVE_PREFIXES:
            if label.startswith(prefix):
                label = label[len(prefix):]
                changed = True
                break # 重新開始檢查，因為 label 開頭變了
        if not changed:
            break
            
    return label

def main():
    """
    主要執行函式 (適配 Narrative + Inventory 架構 + 字串正規化)：
    1. 讀取 `visual_inventory` 中的所有物件清單。
    2. 執行 `normalize_object_label` 清洗標籤。
    3. 統計每個物件在該群組的出現頻率，並過濾低於門檻的雜訊。
    """
    print(f"--- Phase 3: Calculating Probabilities (Normalized Inventory Mode) ---")

    # --- 2. 載入來源檔案 ---
    print(f"Loading source files...")
    if not os.path.exists(INPUT_LABELS_FILE):
        print(f"ERROR: {INPUT_LABELS_FILE} not found.")
        exit()

    try:
        labels = np.load(INPUT_LABELS_FILE)
        with open(INPUT_FILENAMES_FILE, 'r') as f:
            filenames = json.load(f)
        with open(INPUT_SUMMARIES_FILE, 'r', encoding='utf-8') as f:
            summaries_dict = json.load(f)
    except Exception as e:
        print(f"ERROR loading files: {e}")
        exit()

    K_CLUSTERS = int(np.max(labels)) + 1
    print(f"Loaded {len(filenames)} files, grouped into {K_CLUSTERS} clusters.")

    # --- 3. 主迴圈：計算每個群組的機率摘要 ---
    final_summaries = {} 

    for cluster_id in tqdm(range(K_CLUSTERS), desc="Calculating"):
        
        # 找出屬於目前群組的所有影像索引
        indices = np.where(labels == cluster_id)[0]
        total_images = len(indices)

        if total_images == 0:
            final_summaries[str(cluster_id)] = json.dumps({"error": "Empty Cluster"})
            continue
        
        # 我們只會有一個主要的計數器類別 "obj"
        field_counters = {"obj": Counter()}
        
        for idx in indices:
            filename = filenames[idx]
            if filename not in summaries_dict:
                continue
                
            try:
                # 寬鬆模式解析 JSON
                data = json.loads(summaries_dict[filename], strict=False)
            except json.JSONDecodeError:
                continue
            
            # 鎖定 visual_inventory 欄位
            inventory = data.get("visual_inventory", {})
            if not inventory:
                continue

            # 將所有子分類的物件攤平並正規化
            all_objects_in_image = []
            
            # 遍歷 inventory 中的所有 list
            for category_list in inventory.values():
                if isinstance(category_list, list):
                    for item in category_list:
                        if item != "N/A":
                            # [關鍵修改] 應用正規化
                            clean_item = normalize_object_label(str(item))
                            if clean_item: # 確保不是空字串
                                all_objects_in_image.append(clean_item)
            
            # 更新計數器 (使用 set 來避免單張圖重複計算同一個物件多次，視需求而定)
            # 這裡維持 list，代表如果一張圖有兩個 "tree"，就真的算兩次 (密度概念)
            # 如果您希望一張圖只貢獻一次機率，可以使用 set(all_objects_in_image)
            field_counters["obj"].update(all_objects_in_image)

        # --- 3.3 計算機率 (Strict Filter) ---
        cluster_probs = {}

        if "obj" in field_counters:
            counter = field_counters["obj"]
            
            for val, count in counter.items():
                prob = round(count / total_images, 2)
                
                # 嚴格過濾：只保留機率 >= 0.4 的特徵
                if prob >= PROBABILITY_THRESHOLD:
                    full_key = f"obj:{val}"
                    cluster_probs[full_key] = prob

        # 儲存結果
        final_summaries[str(cluster_id)] = json.dumps(cluster_probs, ensure_ascii=False)

    # --- 4. 存檔 ---
    print(f"\nSaving results to {OUTPUT_SUMMARIES_FILE}...")
    with open(OUTPUT_SUMMARIES_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_summaries, f, indent=4, ensure_ascii=False)

    print("Phase 3 Complete.")
    print(f"Logic: Normalized objects, Threshold >= {PROBABILITY_THRESHOLD}")
    print(f"Removed prefixes: {REMOVE_PREFIXES}")

if __name__ == "__main__":
    main()