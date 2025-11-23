import torch
from transformers import AutoProcessor, AutoModel
from PIL import Image
import os
import glob
import numpy as np
from tqdm import tqdm
import json
import warnings

# --- 1. 設定 (您可以在此修改) ---
IMAGE_FOLDER = "data/training_images/"      # 根目錄 (包含所有子資料夾)
OUTPUT_FEATURES_FILE = "all_image_features.npy"     
OUTPUT_FILENAMES_FILE = "image_filenames.json"  

# MODEL_ID = "openai/clip-vit-large-patch14"
MODEL_ID = "facebook/dinov2-base" 

# 忽略 PIL 的一些警告
warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)

# --- 2. 檢查設備 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"正在使用的設備: {DEVICE}")

# --- 3. 載入模型和處理器 ---
print(f"正在載入模型: {MODEL_ID}...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModel.from_pretrained(MODEL_ID).to(DEVICE).eval()
print("模型載入完成。")

# --- 4. 尋找所有影像 (關鍵修改) ---
# 支援巢狀目錄結構 (例如: data/training_images/sunny/noon/001.jpg)
image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
image_paths = []

print(f"正在 {IMAGE_FOLDER} 及其子資料夾中搜尋影像...")

for ext in image_extensions:
    # 使用 "**" 搭配 recursive=True 來進行遞迴搜尋
    # os.path.join(IMAGE_FOLDER, "**", ext) 會變成 "data/training_images/**/*.jpg"
    search_pattern = os.path.join(IMAGE_FOLDER, "**", ext)
    found_files = glob.glob(search_pattern, recursive=True)
    image_paths.extend(found_files)

image_paths = sorted(image_paths) # 排序確保順序一致
print(f"總共找到 {len(image_paths)} 張影像。")

if not image_paths:
    print(f"錯誤：在 {IMAGE_FOLDER} 中找不到任何影像。請檢查路徑或副檔名。")
    exit()

# --- 5. 處理影像並提取特徵 ---
all_features = []
all_filenames = []

with torch.no_grad():
    for path in tqdm(image_paths, desc="提取語意指紋"):
        try:
            image = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"\n警告：跳過無法讀取的影像 {path}, 錯誤: {e}")
            continue
            
        # 1. 處理影像
        inputs = processor(images=image, return_tensors="pt").to(DEVICE)
        
        # 2. 獲取特徵 (DINOv2)
        image_features = model(**inputs).last_hidden_state.mean(dim=1)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # 3. 儲存特徵
        all_features.append(image_features.cpu().numpy().flatten())
        
        # 4. 儲存檔名 (關鍵修改：使用相對路徑)
        # 這樣才能保存 "weather/time/filename.jpg" 的結構資訊
        # 如果只存檔名，不同資料夾下的同名檔案會無法區分
        relative_path = os.path.relpath(path, IMAGE_FOLDER)
        all_filenames.append(relative_path)

# --- 6. 儲存結果到檔案 ---
if not all_features:
    print("錯誤：未能提取任何特徵。")
    exit()

all_features_np = np.array(all_features)

print(f"\n成功提取 {all_features_np.shape[0]} 個特徵。")
print(f"特徵向量維度: {all_features_np.shape[1]}")

print(f"正在儲存特徵向量到 {OUTPUT_FEATURES_FILE}...")
np.save(OUTPUT_FEATURES_FILE, all_features_np)

print(f"正在儲存檔名列表到 {OUTPUT_FILENAMES_FILE}...")
# Windows 系統路徑分隔符號可能是反斜線，為了 JSON 兼容性，建議統一轉為正斜線 (可選)
all_filenames_normalized = [f.replace("\\", "/") for f in all_filenames]
with open(OUTPUT_FILENAMES_FILE, 'w') as f:
    json.dump(all_filenames_normalized, f, indent=4)

print("---")
print("階段一：提取語意指紋 - 完成！")