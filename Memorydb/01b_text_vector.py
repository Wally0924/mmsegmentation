import json
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os

# --- 1. Settings ---
# 我們將使用 Sentence-BERT (S-BERT) 來編碼「文字」
# "all-mpnet-base-v2" 是 S-BERT 中最強的英文模型之一
# 它產生的向量維度是 768
MODEL_ID = "all-mpnet-base-v2"

# (輸入) 階段 1b 產生的 VLM 摘要 (JSON 字典)
INPUT_SUMMARIES_FILE = "llava_summaries.json"
# (輸入) 原始的檔名列表，用來「確保順序」
INPUT_FILENAMES_FILE = "image_filenames.json"

# (輸出) 新的、基於文字的特徵向量
# 這將 *取代* 舊的 all_features.npy
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
# 這是關鍵步驟：我們必須確保 all_text_features.npy[i]
# 100% 對應 image_filenames.json[i]

json_strings_to_encode = []
for filename in filenames_list:
    if filename in summaries_dict:
        # 從字典中找出 VLM 生成的 JSON 字串
        json_str = summaries_dict[filename]
        
        # (重要) 我們直接將「整個 JSON 字串」作為要編碼的文字
        # 這樣 "road_layout" 和 "N/A" 這些 "key" 和 "value"
        # 都會被 S-BERT 納入考量，提供更豐富的語意
        json_strings_to_encode.append(json_str)
    else:
        # 這種情況不應該發生，但作為保險
        print(f"WARNING: Filename {filename} not found in summaries dict! Appending empty string.")
        json_strings_to_encode.append("") # 塞一個空字串來保持順序

# --- 5. Encode Text Vectors (Core Step) ---
print(f"Encoding {len(json_strings_to_encode)} text summaries into vectors...")
print(f"Using BATCH_SIZE = 32 (default for S-BERT)")

# model.encode() 會將所有文字字串的 list 轉換為一個 Numpy 矩陣
# S-BERT (all-mpnet-base-v2) 產生的向量維度是 768
all_text_features_np = model.encode(
    json_strings_to_encode, 
    show_progress_bar=True, # 顯示一個 tqdm 進度條
    batch_size=32           # S-BERT 內建批次處理
)

# --- 6. Save Results ---
print(f"Encoding complete. Vector shape: {all_text_features_np.shape}")
print(f"Saving text-based features to {OUTPUT_FEATURES_FILE}...")
np.save(OUTPUT_FEATURES_FILE, all_text_features_np)

print("\n---")
print("--- New Phase 2: Complete ---")
print(f"Your *new* feature file ({OUTPUT_FEATURES_FILE}) is ready.")
print("This file now contains text-based vectors, ready for clustering.")