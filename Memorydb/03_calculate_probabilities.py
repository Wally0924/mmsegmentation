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
# 建議設為 0.4 或 0.5，代表該群組中至少有 40-50% 的影像都包含此物件
PROBABILITY_THRESHOLD = 0.4

warnings.filterwarnings("ignore", message="invalid escape sequence")

def main():
    """
    主要執行函式 (適配 Narrative + Inventory 架構)：
    1. 讀取 `visual_inventory` 中的所有物件清單。
    2. 將不同分類 (structures, nature...) 攤平為統一的 'obj' 集合。
    3. 統計每個物件在該群組的出現頻率，並過濾低於門檻的雜訊。
    """
    print(f"--- Phase 3: Calculating Probabilities (Inventory Extraction Mode) ---")

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
        
        # 我們只會有一個主要的計數器類別 "obj"，用來存放所有視覺物件
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
            
            # [核心修改]：鎖定 visual_inventory 欄位
            # 新的 Prompt 結構是 { "scene_narrative": "...", "visual_inventory": { ... } }
            inventory = data.get("visual_inventory", {})
            
            if not inventory:
                continue

            # 將所有子分類 (structures, nature, road_components...) 的物件攤平
            all_objects_in_image = []
            
            # 遍歷 inventory 中的所有 list (例如 "structures": [...], "nature": [...])
            for category_list in inventory.values():
                if isinstance(category_list, list):
                    for item in category_list:
                        if item != "N/A":
                            all_objects_in_image.append(str(item))
            
            # 更新計數器 (一次性加入這張圖看到的所有物件)
            field_counters["obj"].update(all_objects_in_image)

        # --- 3.3 計算機率 (Strict Filter) ---
        cluster_probs = {}

        # 這裡我們只處理 "obj" 這個類別
        if "obj" in field_counters:
            counter = field_counters["obj"]
            
            for val, count in counter.items():
                prob = round(count / total_images, 2)
                
                # 嚴格過濾：只保留機率 >= 0.4 的特徵
                if prob >= PROBABILITY_THRESHOLD:
                    # 輸出格式為 "obj:物件名稱"
                    # 例如: "obj:traffic_light": 0.95
                    full_key = f"obj:{val}"
                    cluster_probs[full_key] = prob

        # 儲存結果 (轉為 JSON 字串)
        # 如果該群組沒有任何物件通過門檻，這會是一個空的 JSON "{}"
        final_summaries[str(cluster_id)] = json.dumps(cluster_probs, ensure_ascii=False)

    # --- 4. 存檔 ---
    print(f"\nSaving results to {OUTPUT_SUMMARIES_FILE}...")
    with open(OUTPUT_SUMMARIES_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_summaries, f, indent=4, ensure_ascii=False)

    print("Phase 3 Complete.")
    print(f"Extracted objects from 'visual_inventory'. Threshold: >= {PROBABILITY_THRESHOLD}")

if __name__ == "__main__":
    main()