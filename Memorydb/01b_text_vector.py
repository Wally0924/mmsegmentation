import json
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os

# --- 1. Settings ---
# "all-mpnet-base-v2" 是 S-BERT 中最強的英文模型之一
# 它產生的向量維度是 768
MODEL_ID = "all-mpnet-base-v2"

INPUT_SUMMARIES_FILE = "llava_summaries.json"
INPUT_FILENAMES_FILE = "image_filenames.json"
OUTPUT_FEATURES_FILE = "all_text_features.npy" 

# --- 2. Load Sentence-BERT Model ---
print(f"Loading Sentence-BERT Model: {MODEL_ID}...")
print("This will download the model on the first run...")
# 您可以在這裡指定 device="cuda" (如果您希望用 GPU 加速)
# 但 S-BERT 在 CPU 上也已經非常快
model = SentenceTransformer(MODEL_ID, device="cuda")
print("S-BERT Model loaded.")

# --- 3. Load Source Files ---
print(f"Loading source files...")
try:
    # 載入檔名列表 (用來保證順序)
    # 範例: ["00117393.jpg", ..., "00334795.jpg"]
    with open(INPUT_FILENAMES_FILE, 'r') as f:
        filenames_list = json.load(f)
    
    # 載入 VLM 摘要 (字典)
    # 範例: { "00117393.jpg": "{...json...}", ... }
    with open(INPUT_SUMMARIES_FILE, 'r', encoding='utf-8') as f:
        summaries_dict = json.load(f)
        
except FileNotFoundError as e:
    print(f"ERROR: File not found: {e.filename}")
    print("Please ensure 'llava_summaries.json' and 'image_filenames.json' exist.")
    exit()

print(f"Loaded {len(filenames_list)} filenames and {len(summaries_dict)} summaries.")
if len(filenames_list) != len(summaries_dict):
    print("WARNING: Mismatch in file counts. Proceeding with filenames list.")

# --- 4. Prepare Sentences in Correct Order ---

json_strings_to_encode = []

for filename in filenames_list:
    if filename in summaries_dict:
        raw_json_str = summaries_dict[filename]
        try:
            data = json.loads(raw_json_str)
        except json.JSONDecodeError:
            # 如果碰到壞掉的 JSON，就退回用原始字串
            json_strings_to_encode.append(raw_json_str)
            continue

        # 根據你的 schema 組成固定順序的字串
        parts = []
        
        # 1. 放入完整的敘述 (這對 S-BERT 分群效果最好)
        # Narrative 提供了空間關係和整體氛圍，這對區分 "博物館前" 和 "工業區" 至關重要
        narrative = data.get('scene_narrative', '')
        if narrative:
            parts.append(narrative)
            
        # 2. 補充物件清單 (增強關鍵字權重)
        # 我們把清單扁平化加入，加強 S-BERT 對特定物件的敏感度
        inventory = data.get('visual_inventory', {})
        all_objects = []
        
        # 將所有子類別的物件合併
        for category in ["structures", "road_components", "street_furniture", "nature"]:
            items = inventory.get(category, [])
            if items:
                all_objects.extend(items)
        
        if all_objects:
            # 格式: "Objects: museum_building, stone_column..."
            # 這能確保如果 narrative 漏寫了某個小東西，這裡能補上
            parts.append("Objects: " + ", ".join(all_objects))

        # 最終字串範例:
        # "The scene depicts... Objects: museum_building, stone_column..."
        canonical_str = " ".join(parts) 
        json_strings_to_encode.append(canonical_str)
    else:
        print(f"WARNING: Filename {filename} not found in summaries dict! Appending empty string.")
        json_strings_to_encode.append("")


# --- 5. Encode Text Vectors (Core Step) ---
print(f"Encoding {len(json_strings_to_encode)} text summaries into vectors...")
print(f"Using BATCH_SIZE = 32 (default for S-BERT)")

# S-BERT (all-mpnet-base-v2) 產生的向量維度是 768
all_text_features_np = model.encode(
    json_strings_to_encode, 
    show_progress_bar=True, # 顯示一個 tqdm 進度條
    batch_size=32           # S-BERT 內建批次處理
)

# L2 normalize 文字向量
# 這對後續的 Cosine Similarity 計算很重要
norms = np.linalg.norm(all_text_features_np, axis=1, keepdims=True)
all_text_features_np = all_text_features_np / np.clip(norms, 1e-12, None)


# --- 6. Save Results ---
print(f"Encoding complete. Vector shape: {all_text_features_np.shape}")
print(f"Saving text-based features to {OUTPUT_FEATURES_FILE}...")
np.save(OUTPUT_FEATURES_FILE, all_text_features_np)

print("\n---")
print("--- New Phase 2: Complete ---")
print(f"Your *new* feature file ({OUTPUT_FEATURES_FILE}) is ready.")
print("This file now contains text-based vectors, ready for clustering.")