import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image
import os
import numpy as np
import json
from tqdm import tqdm
import warnings
import random

# --- 1. 設定 ---
K_CLUSTERS = 50

# 每個群組要抽取幾張圖片來生成摘要
NUM_IMAGES_PER_CLUSTER = 5

IMAGE_FOLDER = "data/training_images/"          # 訓練影像資料夾
INPUT_LABELS_FILE = "cluster_labels.npy"      # 分群結果
INPUT_FILENAMES_FILE = "image_filenames.json" # 圖像檔名對應檔案
OUTPUT_SUMMARIES_FILE = "semantic_summaries.json" # 輸出：語意摘要檔案

# LLaVA-NeXT 模型 (VLM)
# llava-v1.6-mistral-7b-hf 是效果好、速度快、VRAM 需求相對小的選擇
MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf"

PROMPT = (
    "These images are all from the exact same location. "
    "Your task is to ignore all transient objects like cars, pedestrians, weather, and time of day. "
    "Focus *only* on the permanent, structural features of the location (e.g., road layout, buildings, landmarks, permanent signs). "
    "Summarize this location in a single, descriptive sentence."
)

# --- 2. 檢查設備 (GPU) ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu":
    print("警告：未偵測到 GPU (cuda)。VLM 推論在 CPU 上會非常非常慢。")

# --- 3. 載入 LLaVA 模型和處理器 ---
print(f"正在載入 VLM 模型: {MODEL_ID}...")

processor = LlavaNextProcessor.from_pretrained(MODEL_ID)
model = LlavaNextForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,    # 使用 16 位元浮點數 (FP16)
    low_cpu_mem_usage=True,     # 節省 CPU 記憶體
    load_in_4bit=True,          # 量化：以 4-bit 格式載入模型
    device_map="auto"           # 自動將模型分配到 GPU
)
model.eval() # 設為推論模式
print("VLM 模型載入完成。")

# --- 4. 載入 階段一 和 階段二 的資料 ---
print("正在載入群集資料...")
try:
    labels = np.load(INPUT_LABELS_FILE)             # (387,)
    with open(INPUT_FILENAMES_FILE, 'r') as f:
        filenames = json.load(f)                    # list, 387
except FileNotFoundError:
    print("錯誤：找不到 cluster_labels.npy 或 image_filenames.json！")
    print("請先執行 01 和 02 腳本。")
    exit()

# 建立一個完整的路徑列表
image_path_list = [os.path.join(IMAGE_FOLDER, f) for f in filenames]

# --- 5. 遍歷群組 (K) 並生成摘要 (核心步驟) ---
print(f"\n--- 階段三：開始為 {K_CLUSTERS} 個群組生成語意摘要 ---")
summaries = {} # 用來儲存 K 個摘要的字典

# 使用 tqdm 顯示進度條
for cluster_id in tqdm(range(K_CLUSTERS), desc="生成摘要中"):
    # 5.1 找出屬於這個群組的所有圖片
    # np.where 會回傳 `labels` 陣列中，值等於 `cluster_id` 的所有「索引」
    indices = np.where(labels == cluster_id)[0]

    if len(indices) == 0:
        print(f"\n群組 {cluster_id}: 沒有分配到任何影像，跳過。")
        summaries[str(cluster_id)] = "N/A (No images in cluster)"
        continue
    
    # 5.2 從索引中隨機挑選 N 張圖片
    num_to_sample = min(NUM_IMAGES_PER_CLUSTER, len(indices))
    selected_indices = random.sample(list(indices), num_to_sample)
    
    # 5.3 載入這 N 張圖片
    image_batch = []
    for idx in selected_indices:
        try:
            img = Image.open(image_path_list[idx]).convert("RGB")
            image_batch.append(img)
        except Exception as e:
            print(f"警告：無法讀取影像 {image_path_list[idx]}, {e}")
            
    if not image_batch:
        print(f"\n群組 {cluster_id}: 影像讀取失敗，跳過。")
        summaries[str(cluster_id)] = "N/A (Image read error)"
        continue

    # 5.4 建立 VLM Prompt (多張圖 + 一段文字)
    # LLaVA 1.6+ 支援 [ {"type": "text", ...}, {"type": "image"}, {"type": "image"} ] 格式
    content = [{"type": "text", "text": PROMPT}]
    for _ in image_batch:
        content.append({"type": "image"}) # 加入 N 個影像佔位符

    conversation = [{"role": "user", "content": content}]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    
    # 5.5 執行推論
    with torch.no_grad():
        inputs = processor(
            text=prompt, 
            images=image_batch,  # 傳入 N 張圖片的 list
            return_tensors="pt"
        ).to(DEVICE)

        output = model.generate(**inputs, max_new_tokens=150)
    
    # 5.6 解碼並儲存結果
    response_full = processor.decode(output[0], skip_special_tokens=True)
    # response_full 會是 "USER: ... ASSISTANT: 這是摘要"
    # 我們只取 "ASSISTANT:" 後面的部分
    try:
        response_text = response_full.split("[/INST]")[-1].strip()
    except:
        response_text = response_full.strip() # 保險起見

    summaries[str(cluster_id)] = response_text
    
    # (可選) 在進度條中顯示最新摘要
    tqdm.write(f"群組 {cluster_id:02d} 摘要: {response_text}")

# --- 6. 儲存所有摘要到 JSON 檔案 ---
print("\n---")
print(f"正在將 {len(summaries)} 筆摘要儲存到 {OUTPUT_SUMMARIES_FILE}...")

# ensure_ascii=False 確保中文可以正確儲存，不會變成 \uXXXX
with open(OUTPUT_SUMMARIES_FILE, 'w', encoding='utf-8') as f:
    json.dump(summaries, f, indent=4, ensure_ascii=False)

print("階段三：生成語意摘要 - 完成！")
print("您記憶庫的『Value』(語意摘要) 已準備就緒。")