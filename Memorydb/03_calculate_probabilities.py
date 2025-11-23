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

# 機率門檻
PROBABILITY_THRESHOLD = 0.5

# [新設定] 強制保留名單
# 這些欄位通常描述性很強，變異大，很難超過 0.5。
# 我們強制保留該欄位中出現最多次的 1 個，以免資料庫失去這些關鍵細節。
FORCE_KEEP_TOP_1_FIELDS = ["primary_landmark", "distinctive_features", "surrounding_structure"]

warnings.filterwarnings("ignore", message="invalid escape sequence")

def main():
    print(f"--- Phase 3: Calculating Probabilities (Field-Aware Version) ---")

    # --- 2. Load Source Files ---
    print(f"Loading source files...")
    if not os.path.exists(INPUT_LABELS_FILE):
        print(f"ERROR: {INPUT_LABELS_FILE} not found.")
        exit()

    # 使用 with open 確保資源釋放
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
    print(f"Loaded {len(filenames)} files, {K_CLUSTERS} clusters.")

    # --- 3. Main Loop ---
    final_summaries = {} 

    for cluster_id in tqdm(range(K_CLUSTERS), desc="Calculating"):
        
        indices = np.where(labels == cluster_id)[0]
        total_images = len(indices)

        if total_images == 0:
            final_summaries[str(cluster_id)] = json.dumps({"error": "Empty Cluster"})
            continue
        
        # [修改] 使用「字典套 Counter」來分欄位統計
        # 結構: { "road_layout": Counter(), "ocr_text": Counter(), ... }
        field_counters = {} 
        
        for idx in indices:
            filename = filenames[idx]
            if filename not in summaries_dict:
                continue
                
            try:
                data = json.loads(summaries_dict[filename], strict=False)
            except json.JSONDecodeError:
                continue
            
            for key, value in data.items():
                if value == "N/A" or value is None:
                    continue
                
                # 初始化該欄位的 Counter
                if key not in field_counters:
                    field_counters[key] = Counter()

                # (您的正確邏輯) 列表拆解
                if isinstance(value, list):
                    for item in value:
                        if item != "N/A":
                            # 這裡只存值，不存 key 前綴，最後再組裝
                            field_counters[key].update([str(item)])
                else:
                    field_counters[key].update([str(value)])

        # --- 3.3 計算機率與補位 ---
        cluster_probs = {}

        for key, counter in field_counters.items():
            if not counter:
                continue

            # A. 先篩選出大於門檻的
            candidates = {}
            for val, count in counter.items():
                prob = round(count / total_images, 2)
                if prob >= PROBABILITY_THRESHOLD:
                    candidates[val] = prob
            
            # B. [關鍵邏輯] 補位機制
            # 如果篩選後是空的，但這個欄位很重要 (在強制名單內)
            # 就把出現最多次的那個抓回來
            if not candidates and key in FORCE_KEEP_TOP_1_FIELDS:
                most_common_val, count = counter.most_common(1)[0]
                prob = round(count / total_images, 2)
                candidates[most_common_val] = prob # 強制加入
            
            # C. 將結果加入最終字典 (組裝 key:value)
            for val, prob in candidates.items():
                # 為了讓資料庫搜尋方便，我們把 key 拼回去
                # 例如: "road_layout:T-junction"
                full_key = f"{key}:{val}"
                cluster_probs[full_key] = prob

        # 儲存
        final_summaries[str(cluster_id)] = json.dumps(cluster_probs, ensure_ascii=False)

    # --- 4. Save ---
    print(f"\nSaving results to {OUTPUT_SUMMARIES_FILE}...")
    with open(OUTPUT_SUMMARIES_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_summaries, f, indent=4, ensure_ascii=False)

    print("Phase 3 Complete.")

if __name__ == "__main__":
    main()