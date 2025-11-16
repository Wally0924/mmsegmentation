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
You are a precise, objective scene analysis AI.
Your task is to analyze only the permanent structural features of this road scene image and output one JSON object.

[STRICT RULES]
1. Treat all vehicles, pedestrians, cyclists, animals, and moving objects as invisible, and do not mention them anywhere.
2. Do not mention time of day, weather, sunlight, shadows, headlights, or reflections.
3. Focus only on fixed, immovable structures such as roads, buildings, barriers, trees, and permanent signs.
4. For very similar images of the same location, use exactly the same wording for each JSON field whenever possible.

[OUTPUT FORMAT]
You must output exactly one JSON object and nothing else (no explanations, no markdown, no comments).
Use the following JSON skeleton, keeping the same keys and order:

{
  "primary_landmark": "",
  "road_layout": "",
  "road_markings": [],
  "left_structure_type": "",
  "right_structure_type": "",
  "vegetation_type": "",
  "key_street_furniture": [],
  "ocr_text_on_signs": []
}

Now fill each field following these rules:

- "primary_landmark":
  The single most prominent fixed landmark in view as a short noun phrase,
  for example: "classical_museum_building", "idea_factory_building",
  "underpass_bridge", or "N/A".

- "road_layout":
  Choose one from the following only:
  "crossroad", "t_junction", "roundabout", "underpass",
  "overpass", "straight_road", "curve_road",
  "tunnel_entrance", "N/A".

- "road_markings":
  A list of permanent road markings as snake_case tokens,
  such as ["crosswalk", "double_yellow_line", "bike_lane"],
  or ["N/A"] if none are visible.

- "left_structure_type":
  The type of the immediate left structure next to the road edge, chosen from:
  "high_rise", "shop", "apartment", "brick_building",
  "sound_barrier", "trees", "open_field", "N/A".

- "right_structure_type:
  The type of the immediate right structure next to the road edge,
  using the same vocabulary as "left_structure_type".

- "vegetation_type":
  The dominant fixed vegetation type, chosen from:
  "palm_trees", "deciduous_trees", "conifer_trees",
  "bushes", "grass", "N/A".

- "key_street_furniture":
  A list of fixed sidewalk objects from the set:
  "bus_stop", "traffic_light", "guardrail",
  "fire_hydrant", "bench", "street_lamp",
  "utility_box", or "N/A" if none.

- "ocr_text_on_signs":
  A list of uppercase text strings read from permanent signs,
  for example: "MUSEUM", "IDEA FACTORY", "STOP",
  or ["N/A"] if no readable text is visible.

Remember:
Return only the JSON object, with double quotes around all keys
and string values.
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