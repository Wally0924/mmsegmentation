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

warnings.filterwarnings("ignore", message="invalid escape sequence")

def main():
    """
    主要執行函式：
    1. 載入分群標籤與 VLM 摘要。
    2. 針對每個群組，統計各欄位特徵的出現頻率。
    3. 只保留出現機率大於等於 PROBABILITY_THRESHOLD (0.5) 的特徵。
    """
    print(f"--- Phase 3: Calculating Probabilities (Strict Threshold Version) ---")

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
    print(f"Loaded {len(filenames)} files, {K_CLUSTERS} clusters.")

    # --- 3. 主迴圈：計算每個群組的機率摘要 ---
    final_summaries = {} 

    for cluster_id in tqdm(range(K_CLUSTERS), desc="Calculating"):
        
        # 找出屬於目前群組的所有影像索引
        indices = np.where(labels == cluster_id)[0]
        total_images = len(indices)

        if total_images == 0:
            final_summaries[str(cluster_id)] = json.dumps({"error": "Empty Cluster"})
            continue
        
        # 使用「字典套 Counter」來分欄位統計
        # 結構: { "road_layout": Counter(), "ocr_text": Counter(), ... }
        field_counters = {} 
        
        for idx in indices:
            filename = filenames[idx]
            if filename not in summaries_dict:
                continue
                
            try:
                # 寬鬆模式解析 JSON
                data = json.loads(summaries_dict[filename], strict=False)
            except json.JSONDecodeError:
                continue
            
            for key, value in data.items():
                if value == "N/A" or value is None:
                    continue
                
                # 初始化該欄位的 Counter
                if key not in field_counters:
                    field_counters[key] = Counter()

                # 列表拆解邏輯 (正確處理如 ["crosswalk", "line"] 的情況)
                if isinstance(value, list):
                    for item in value:
                        if item != "N/A":
                            field_counters[key].update([str(item)])
                else:
                    field_counters[key].update([str(value)])

        # --- 3.3 計算機率 (Strict Filter) ---
        cluster_probs = {}

        for key, counter in field_counters.items():
            if not counter:
                continue

            # 直接遍歷所有統計值
            for val, count in counter.items():
                prob = round(count / total_images, 2)
                
                # [核心邏輯] 嚴格過濾：只保留機率 >= 0.5 的特徵
                # 不再進行任何補位 (Fallback) 操作
                if prob >= PROBABILITY_THRESHOLD:
                    # 拼裝 Key 名稱，例如: "road_layout:straight_road"
                    full_key = f"{key}:{val}"
                    cluster_probs[full_key] = prob

        # 儲存結果 (轉為 JSON 字串)
        final_summaries[str(cluster_id)] = json.dumps(cluster_probs, ensure_ascii=False)

    # --- 4. 存檔 ---
    print(f"\nSaving results to {OUTPUT_SUMMARIES_FILE}...")
    with open(OUTPUT_SUMMARIES_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_summaries, f, indent=4, ensure_ascii=False)

    print("Phase 3 Complete.")
    print(f"Logic: Only features with probability >= {PROBABILITY_THRESHOLD} are kept.")

if __name__ == "__main__":
    main()