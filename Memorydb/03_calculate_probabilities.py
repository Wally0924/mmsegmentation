import numpy as np
import json
import os
from collections import Counter
from tqdm import tqdm
import warnings

# --- 1. Settings ---
K_CLUSTERS = 30 

# (輸入) VLM-First 階段的產出
INPUT_LABELS_FILE = "cluster_labels.npy"      # 來自 02_ (基於文字向量) 的分群標籤
INPUT_FILENAMES_FILE = "image_filenames.json" # 來自 01_ 的原始檔名索引
INPUT_SUMMARIES_FILE = "llava_summaries.json" # 來自 01b_ (VLM 摘要)

# (輸出) 最終的 "Value"
OUTPUT_SUMMARIES_FILE = "semantic_summaries.json" 

# (可選) 忽略 JSON 剖析失敗的警告
warnings.filterwarnings("ignore", message="invalid escape sequence")

# --- 2. Load Source Files ---
print(f"Loading source files...")
try:
    labels = np.load(INPUT_LABELS_FILE)             
    filenames = json.load(open(INPUT_FILENAMES_FILE, 'r'))
    summaries_dict = json.load(open(INPUT_SUMMARIES_FILE, 'r', encoding='utf-8'))
except FileNotFoundError as e:
    print(f"ERROR: File not found: {e.filename}")
    print("Please run scripts 01b, 01c, and 02 first.")
    exit()

print("All files loaded successfully.")

# --- 3. Main Loop: Calculate Probabilities for K Clusters ---
print(f"\n--- New Phase 3: Calculating Probabilities for {K_CLUSTERS} clusters ---")

# final_summaries: 最終的 "Value" 字典 { "0": "{...prob...}", "1": "{...prob...}" }
final_summaries = {} 

for cluster_id in tqdm(range(K_CLUSTERS), desc="Calculating Probabilities"):
    
    # 3.1 找出屬於這個群組的所有圖片的「索引」
    indices = np.where(labels == cluster_id)[0]
    
    # 該群組的影像總數
    total_images_in_cluster = len(indices)

    if total_images_in_cluster == 0:
        tqdm.write(f"Cluster {cluster_id}: No images assigned. Skipping.")
        final_summaries[str(cluster_id)] = '{"error": "N/A (No images in cluster)"}'
        continue
    
    # 3.2 (關鍵) 統計此群組中「所有」特徵
    # 我們使用 Counter 來統計 "key:value" 對
    # 範例: {"road_layout:T-junction": 18, "ocr_text:7-ELEVEN": 15, ...}
    feature_counter = Counter()
    
    for idx in indices:
        filename = filenames[idx]
        if filename not in summaries_dict:
            continue
            
        json_str = summaries_dict[filename]
        
        try:
            # (A) 剖析 LLaVA 產生的 JSON 字串
            # 範例: '{"road_layout": "T-junction", ...}' -> {"road_layout": "T-junction", ...}
            # 我們使用 'strict=False' 來容錯 (例如 LLaVA 產生的 JSON 中有換行)
            data = json.loads(json_str, strict=False) 
        except json.JSONDecodeError:
            tqdm.write(f"WARNING: Cluster {cluster_id}, file {filename} - JSON decode error. Skipping.")
            continue
            
        # (B) 遍歷 JSON 中的 Key-Value 對
        for key, value in data.items():
            # (C) 我們只統計「有意義」的 (非 N/A) 特徵
            if value != "N/A" and value is not None:
                
                # (D) 將 Key 和 Value 組合為一個「獨特特徵」
                # 範例: "road_layout" + "T-junction" -> "road_layout:T-junction"
                # 範例: "ocr_text_on_signs" + "7-ELEVEN" -> "ocr_text_on_signs:7-ELEVEN"
                feature_key = f"{key}:{value}"
                
                # (E) 累加計數
                feature_counter.update([feature_key])

    if not feature_counter:
        tqdm.write(f"Cluster {cluster_id}: No valid features found. Skipping.")
        final_summaries[str(cluster_id)] = '{"error": "N/A (No valid features found)"}'
        continue
        
    # 3.3 (關鍵) 將「計數」轉換為「機率」
    # 範例: "road_layout:T-junction" 出現了 18 次，總共 20 張圖
    # 機率 = 18 / 20 = 0.9
    probabilities_dict = {}
    for feature, count in feature_counter.items():
        probability = count / total_images_in_cluster
        probabilities_dict[feature] = round(probability, 2) # 四捨五入到小數點後 2 位
    
    # 3.4 儲存結果
    # 我們將這個「機率字典」轉換為「JSON 字串」，這就是我們的 Value
    final_value_string = json.dumps(probabilities_dict, ensure_ascii=False)
    final_summaries[str(cluster_id)] = final_value_string
    # tqdm.write(f"Cluster {cluster_id:02d} Probabilities: {final_value_string}")

# --- 4. Save All Summaries ---
print("\n---")
print(f"Saving {len(final_summaries)} probability summaries to {OUTPUT_SUMMARIES_FILE}...")

with open(OUTPUT_SUMMARIES_FILE, 'w', encoding='utf-8') as f:
    json.dump(final_summaries, f, indent=4, ensure_ascii=False)

print("--- New Phase 3 (Probability Calculation): Complete ---")
print(f"Your database 'Values' ({OUTPUT_SUMMARIES_FILE}) are now Probability JSONs.")