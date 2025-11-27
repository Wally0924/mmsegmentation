import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image
import os
import numpy as np
import json
from tqdm import tqdm
import warnings
import math
import re
# --- 1. Settings ---
BATCH_SIZE = 8 

IMAGE_FOLDER = "data/training_images/"        # 您的影像資料夾
INPUT_FILENAMES_FILE = "image_filenames.json" # 檔名索引
OUTPUT_SUMMARIES_FILE = "llava_summaries.json"  # 新的 VLM 摘要輸出

# LLaVA-NeXT VLM
MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf"

# ENGLISH_JSON_PROMPT = '''
# You are a highly detailed visual perception AI.
# Your task is to analyze the PERMANENT structural features of this virtual city image and provide output in two specific parts: a descriptive narrative and a precise object inventory.

# [STRICT RULES]
# 1. IGNORE moving objects (cars, pedestrians). Treat the scene as static.
# 2. IGNORE transient lighting, weather, and time of day. Focus on the "true color" (albedo) and permanent structure.
# 3. NO OCR. Focus on the physical objects (e.g., "rectangular banner"), not the text on them.
# 4. OUTPUT FORMAT: Single JSON object.

# [JSON SKELETON]
# {
#   "scene_narrative": "...",
#   "visual_inventory": {
#     "structures": [],
#     "road_components": [],
#     "street_furniture": [],
#     "nature": []
#   }
# }

# [INSTRUCTIONS]

# 1. "scene_narrative" (For Scene Clustering):
#    Write a comprehensive, natural language paragraph (3-5 sentences) describing the scene.
#    - Focus on PERMANENT features. Do not describe shadows, puddles, or sun glare.
#    - Start with the global layout (e.g., "This is a T-junction facing a grand classical museum...").
#    - Describe the spatial relationship of major structures (e.g., "On the left is a tree, to the right is a brick building...").
#    - Mention the architectural style and atmosphere.
#    - Example: "The scene depicts a wide T-intersection paved with asphalt. Dominating the view is a massive beige stone museum with classical columns and a grand staircase. To the left, large deciduous trees line the sidewalk. The road features prominent white ladder-style crosswalks."

# 2. "visual_inventory" (For Semantic Segmentation):
#    List specific, physical objects visible in the image. Be EXHAUSTIVE.
   
#    - "structures": Large fixed objects and building parts.
#      (e.g., ["museum_building", "stone_column", "grand_staircase", "brick_wall", "glass_facade", 
#             "skybridge", "warehouse", "balcony", "construction_scaffolding", "portico"])
     
#    - "road_components": Drivable surfaces and markings.
#      (e.g., ["asphalt_road", "concrete_sidewalk", "white_crosswalk", "double_yellow_line", 
#             "stop_line", "bike_lane_marking", "paved_tiles", "curb_stone"])
     
#    - "street_furniture": Small fixed objects on the sidewalk.
#      (e.g., ["traffic_light_vertical", "traffic_light_horizontal", "street_lamp", "banner_pole", 
#             "vertical_banner", "trash_can", "fire_hydrant", "bench", "bus_stop_shelter", "bollard"])
     
#    - "nature": Vegetation and sky elements.
#      (e.g., ["deciduous_tree_leafy", "deciduous_tree_bare", "palm_tree", "bush", 
#             "grass_patch", "potted_plant", "open_sky"])

# Remember: The narrative helps understand the "Whole", the inventory identifies the "Parts" for segmentation.
# '''

ENGLISH_JSON_PROMPT = '''
You are a semantic segmentation assistant.
Your goal is to list ONLY the objects visible in the image to help a segmentation model.

[STRICT RULES]
1. IGNORE moving objects (cars, pedestrians).
2. IGNORE weather/time effects.
3. NO OCR.
4. OUTPUT FORMAT: Single JSON object.

[REFERENCE VOCABULARY - SELECT FROM HERE]
(Do NOT copy the whole list. Pick only what you see.)

* STRUCTURES:
  [museum_building, stone_column, grand_staircase, brick_wall, glass_facade, skybridge, warehouse, balcony, construction_scaffolding, portico, red_awning, beige_stone_building]

* ROAD_COMPONENTS:
  [asphalt_road, concrete_sidewalk, white_crosswalk, double_yellow_line, stop_line, bike_lane_marking, paved_tiles, curb_stone]

* STREET_FURNITURE:
  [traffic_light_vertical, traffic_light_horizontal, street_lamp, banner_pole, vertical_banner, trash_can, fire_hydrant, bench, bus_stop_shelter, bollard]

* NATURE:
  [deciduous_tree_leafy, deciduous_tree_bare, palm_tree, bush, grass_patch, potted_plant, open_sky]

[TASK]
1. Analyze the image.
2. Write a "scene_narrative" (3-5 sentences).
3. Create a "visual_inventory" by selecting APPLICABLE items from the Reference Vocabulary.
   - If you see a tree, select "deciduous_tree_leafy".
   - If you DO NOT see a warehouse, DO NOT write "warehouse".
   - It is better to return a short, accurate list than a long, wrong one.

[JSON SKELETON]
{
  "scene_narrative": "...",
  "visual_inventory": {
    "structures": [],
    "road_components": [],
    "street_furniture": [],
    "nature": []
  }
}
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
        # 修改 generate 部分
        outputs = model.generate(
            **inputs, 
            max_new_tokens=1024,      # 確保長度足夠
            do_sample=False,          # [新增] 關閉採樣，使用貪婪解碼 (Greedy Search)
            # 或者使用: do_sample=True, temperature=0.2, top_p=0.9
            repetition_penalty=1.1    # [新增] 稍微懲罰重複內容，防止它一直複製
        )
        
        # 5.5 批次解碼
        # 使用 batch_decode 來一次解碼所有 N 個輸出
        decoded_responses = processor.batch_decode(outputs, skip_special_tokens=True)
        
        # 5.6 剖析並儲存這個批次的結果
        for filename, full_response in zip(valid_filenames_batch, decoded_responses):
            try:
                # 1. 提取 LLaVA 回答部分
                raw_output = full_response.split("[/INST]")[-1].strip()
                
                # 2. 使用 Regex 尋找 JSON 物件 (比 hardcode 切片更穩健)
                # 尋找第一個 { 和最後一個 } 之間的內容
                match = re.search(r'(\{.*\})', raw_output, re.DOTALL)
                if match:
                    json_str = match.group(1)
                else:
                    # 如果找不到大括號，嘗試直接解析整個字串
                    json_str = raw_output

                # 3. 嘗試解析 JSON
                data = json.loads(json_str)

                # --- [防呆機制] 檢查是否發生「清單複製」 ---
                # 如果某個欄位的物件超過 8 個，極有可能是模型在抄襲 Prompt
                inventory = data.get("visual_inventory", {})
                is_hallucinating = False
                
                for cat in ["structures", "road_components", "street_furniture", "nature"]:
                    items = inventory.get(cat, [])
                    if isinstance(items, list) and len(items) > 8:
                        tqdm.write(f"WARNING: Detected hallucination in {filename} ({cat} has {len(items)} items). Pruning.")
                        inventory[cat] = [] # 清空該欄位，避免污染資料庫
                        is_hallucinating = True
                
                if is_hallucinating:
                    data["visual_inventory"] = inventory
                    # 重新轉回字串
                    json_str = json.dumps(data, ensure_ascii=False)

                # 儲存
                summaries_dict[filename] = json_str

            except Exception as e:
                tqdm.write(f"WARNING: Failed to parse output for {filename}: {e}")
                # 選擇性：印出錯誤的輸出以便除錯
                # tqdm.write(f"Raw: {raw_output[:100]}...") 
                summaries_dict[filename] = f'{{"error": "parse_error"}}'

# --- 6. Save All Summaries ---
print("\n---")
print(f"Saving {len(summaries_dict)} summaries to {OUTPUT_SUMMARIES_FILE}...")

with open(OUTPUT_SUMMARIES_FILE, 'w', encoding='utf-8') as f:
    json.dump(summaries_dict, f, indent=4, ensure_ascii=False)

print("--- New Phase 1: Complete ---")