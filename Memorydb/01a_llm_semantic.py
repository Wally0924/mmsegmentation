import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image
import os
import numpy as np
import json
from tqdm import tqdm
import warnings
import math

# --- 1. Settings ---
BATCH_SIZE = 8 

IMAGE_FOLDER = "data/training_images/"        # 您的影像資料夾
INPUT_FILENAMES_FILE = "image_filenames.json" # 檔名索引
OUTPUT_SUMMARIES_FILE = "llava_summaries.json"  # 新的 VLM 摘要輸出

# LLaVA-NeXT VLM
MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf"

ENGLISH_JSON_PROMPT = '''
You are a specialized scene analysis AI for a synthetic virtual city.
Your task is to identify the unique "Zone Fingerprint" of this location based on architectural style and layout.

[STRICT RULES]
1. IGNORE all moving objects (cars, pedestrians). 
2. IGNORE lighting/weather/time. Focus on the permanent "texture" of the city.
3. DO NOT READ TEXT. Ignore all signboards and banners. Focus on the structure holding them.
4. OUTPUT FORMAT: Single JSON object only.

[JSON SKELETON]
{
  "zone_archetype": "...",
  "architectural_style": "...",
  "road_layout": "...",
  "sky_visibility": "...",
  "distinctive_structure": "...",
  "left_side_building": "...",
  "right_side_building": "...",
  "vegetation_signature": "...",
  "road_markings": []
}

[FIELD GUIDELINES]

- "zone_archetype":
  Classify the general vibe of this area:
  ["industrial_district", "classical_cultural_center", "resort_boulevard", 
   "dense_downtown_canyon", "residential_suburb", "construction_zone", "highway_overpass_area"]

- "architectural_style":
  The dominant material and design style:
  ["red_brick_industrial", "beaux_arts_stone_classical", "modern_glass_curtain", 
   "beige_stucco_resort", "mixed_urban_facades", "concrete_brutalist"]

- "road_layout":
  ["crossroad", "t_junction", "straight_avenue", "curved_boulevard", 
   "narrow_alley", "wide_intersection_with_island"]

- "sky_visibility":
  How much sky is visible? This distinguishes downtown canyons from open resorts.
  ["narrow_strip_visible", "open_sky_wide", "blocked_by_overhead_structure", "partially_obstructed_by_trees"]

- "distinctive_structure":
  Look for ONE unique identifier that separates this place from others:
  ["pedestrian_skybridge_connecting_buildings", "monumental_columned_portico", 
   "grand_stone_staircase", "dense_cluster_of_skyscrapers", "balcony_lined_apartments", 
   "large_billboard_frame", "N/A"]

- "left_side_building" & "right_side_building":
  Describe the immediate buildings:
  ["factory_warehouse_brick", "classical_museum_wing", "glass_skyscraper", 
   "stucco_hotel_complex", "parking_garage", "low_rise_shops", "construction_site"]

- "vegetation_signature":
  The type of trees is a key location marker in this city:
  ["palm_tree_rows", "large_canopy_deciduous_trees", "sparse_street_saplings", 
   "dense_tropical_bushes", "no_vegetation"]

- "road_markings":
  ["crosswalk_ladder_style", "double_yellow_lines", "white_lane_dividers", 
   "hatched_junction_box", "stop_line", "none"]

Remember: Output valid JSON only. Use the exact tokens provided above where possible.
'''

# --- 2. Setup Device ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu":
    print("WARNING: No GPU (cuda) detected. VLM inference will be extremely slow on CPU.")
else:
    print(f"Device: {DEVICE}")

# --- 3. Load VLM Model ---
print(f"Loading VLM Model: {MODEL_ID}...")
processor = LlavaNextProcessor.from_pretrained(MODEL_ID)
model = LlavaNextForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,    # 使用 FP16 (16-bit)
    low_cpu_mem_usage=True,     # 節省 CPU 記憶體
    device_map="auto"           # "auto" 將自動把 14GB 的模型載入 VRAM
)
model.eval()
print("VLM Model loaded successfully.")

# --- 4. Load Image Filename List ---
# 根據INPUT_FILENAMES_FILE載入所有影像檔名
print(f"Loading image filenames from {INPUT_FILENAMES_FILE}...")
try:
    with open(INPUT_FILENAMES_FILE, 'r') as f:
        filenames = json.load(f) # 載入 [ "001.jpg", "002.jpg", ... ]
except FileNotFoundError:
    print(f"ERROR: {INPUT_FILENAMES_FILE} not found!")
    exit()

num_images = len(filenames)
num_batches = math.ceil(num_images / BATCH_SIZE)
print(f"Found {num_images} images to process in {num_batches} batches (Batch Size = {BATCH_SIZE}).")

# --- 5. Main Batch-Processing Loop ---
print(f"\n--- New Phase 1: Generating {num_images} summaries ---")

# `summaries_dict` 將是我們的輸出: { "檔名": "JSON 字串", ... }
summaries_dict = {}

# 準備 LLaVA 的對話模板
content_template = [
    {"type": "text", "text": ENGLISH_JSON_PROMPT},
    {"type": "image"} # 單一影像佔位符
]
conversation_template = [{"role": "user", "content": content_template}]
prompt_template = processor.apply_chat_template(conversation_template, add_generation_prompt=True)


with torch.no_grad():
    # 建立批次迴圈
    for i in tqdm(range(0, num_images, BATCH_SIZE), desc="Processing Batches"):
        
        # 5.1 準備這個批次的資料
        batch_filenames = filenames[i : i + BATCH_SIZE]
        image_batch = []
        valid_filenames_batch = [] # 儲存這個批次中「讀取成功」的檔名

        for filename in batch_filenames:
            image_path = os.path.join(IMAGE_FOLDER, filename)
            try:
                img = Image.open(image_path).convert("RGB")
                image_batch.append(img)
                valid_filenames_batch.append(filename)
            except Exception as e:
                tqdm.write(f"WARNING: Skipping unreadable image {filename}, Error: {e}")
                summaries_dict[filename] = f'{{"error": "image read error: {e}"}}'
        
        if not image_batch:
            tqdm.write("Skipping empty batch (all images failed to read).")
            continue

        # 5.2 建立批次 Prompts
        # 建立一個 list，包含 N 個相同的 prompt
        prompts_batch = [prompt_template] * len(image_batch)
            
        # 5.3 批次處理影像和 Prompts
        inputs = processor(
            text=prompts_batch,  # 傳入 N 個 prompt
            images=image_batch,  # 傳入 N 張影像
            return_tensors="pt",
            padding=True # 批次處理必須打開 Padding
        ).to(DEVICE)

        # 5.4 LLaVA 執行「批次推論」
        outputs = model.generate(**inputs, max_new_tokens=256)
        
        # 5.5 批次解碼
        # 使用 batch_decode 來一次解碼所有 N 個輸出
        decoded_responses = processor.batch_decode(outputs, skip_special_tokens=True)
        
        # 5.6 剖析並儲存這個批次的結果
        for filename, full_response in zip(valid_filenames_batch, decoded_responses):
            try:
                # 依據 `[/INST]` 標記來分割，找出 LLaVA 的回答
                response_json_str = full_response.split("[/INST]")[-1].strip()
                
                if response_json_str.startswith("```json"):
                    response_json_str = response_json_str[7:-3].strip()
                elif response_json_str.startswith("{"):
                    pass
                else:
                    raise Exception(f"Not a JSON: {response_json_str[:20]}")

                # 儲存這個乾淨的 "JSON 字串"
                summaries_dict[filename] = response_json_str

            except Exception as e:
                tqdm.write(f"WARNING: Failed to parse VLM output for {filename}, Error: {e}")
                tqdm.write(f"         Full Output: {full_response}")
                summaries_dict[filename] = f'{{"error": "VLM output parse error: {e}"}}'

# --- 6. Save All Summaries ---
print("\n---")
print(f"Saving {len(summaries_dict)} summaries to {OUTPUT_SUMMARIES_FILE}...")

with open(OUTPUT_SUMMARIES_FILE, 'w', encoding='utf-8') as f:
    json.dump(summaries_dict, f, indent=4, ensure_ascii=False)

print("--- New Phase 1: Complete ---")