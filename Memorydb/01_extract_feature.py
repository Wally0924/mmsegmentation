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
IMAGE_FOLDER = "data/training_images/"      # 包含您所有影像的資料夾
OUTPUT_FEATURES_FILE = "all_image_features.npy"     # 儲存所有特徵向量的檔案
OUTPUT_FILENAMES_FILE = "image_filenames.json"  # 儲存對應檔名的檔案

# 推薦的 Image Encoder (CLIP)
MODEL_ID = "openai/clip-vit-large-patch14"

# 忽略 PIL 的一些警告
warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)

# --- 2. 檢查設備 (GPU 或 CPU) ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"正在使用的設備: {DEVICE}")

# --- 3. 載入模型和處理器 ---
print(f"正在載入模型: {MODEL_ID}...")
# AutoProcessor 會處理所有影像預處理 (調整大小、標準化等)
processor = AutoProcessor.from_pretrained(MODEL_ID)
# AutoModel 會載入 CLIP 模型
model = AutoModel.from_pretrained(MODEL_ID).to(DEVICE).eval() # .eval() 設為推論模式
print("模型載入完成。")

# --- 4. 尋找所有影像 ---
image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
image_paths = []
for ext in image_extensions:
    image_paths.extend(glob.glob(os.path.join(IMAGE_FOLDER, ext)))

image_paths = sorted(image_paths) # 排序以確保順序一致
print(f"在 {IMAGE_FOLDER} 中找到 {len(image_paths)} 張影像。")

if not image_paths:
    print(f"錯誤：在 {IMAGE_FOLDER} 中找不到任何影像。請檢查路徑。")
    exit()

# --- 5. 處理影像並提取特徵 (核心步驟) ---
all_features = []
all_filenames = []

with torch.no_grad():
    for path in tqdm(image_paths, desc="提取語意指紋 (Semantic Fingerprints)"):
        try:
            image = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"\n警告：跳過無法讀取的影像 {path}, 錯誤: {e}")
            continue
            
        # 1. 處理影像：使用 processor 將 PIL Image 轉為模型需要的 Tensor
        inputs = processor(images=image, return_tensors="pt").to(DEVICE)
        
        # 2. 獲取特徵：
        # model.get_image_features() 是 CLIP 專門用來提取影像特徵的函式
        image_features = model.get_image_features(**inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        # 3. 儲存結果：
        #    - .cpu() 將資料從 GPU 移回 CPU
        #    - .numpy() 將 PyTorch Tensor 轉為 Numpy array
        #    - .flatten() 將 (1, 768) 轉為 (768,) 的一維向量
        all_features.append(image_features.cpu().numpy().flatten())
        all_filenames.append(os.path.basename(path))

# --- 6. 儲存結果到檔案 ---
if not all_features:
    print("錯誤：未能提取任何特徵。")
    exit()

# 將 Python list 轉換為一個大型的 2D Numpy array
# 它的形狀會是 (影像數量, 特徵維度) (例如: 1000, 768)
all_features_np = np.array(all_features)

print(f"\n成功提取 {all_features_np.shape[0]} 個特徵。")
print(f"特徵向量維度: {all_features_np.shape[1]}")

# 儲存 Numpy 向量檔案
print(f"正在儲存特徵向量到 {OUTPUT_FEATURES_FILE}...")
np.save(OUTPUT_FEATURES_FILE, all_features_np)

# 儲存檔名對應表
print(f"正在儲存檔名列表到 {OUTPUT_FILENAMES_FILE}...")
with open(OUTPUT_FILENAMES_FILE, 'w') as f:
    json.dump(all_filenames, f, indent=4)

print("---")
print("階段一：提取語意指紋 - 完成！")
print(f"您的特徵向量已儲存在: {OUTPUT_FEATURES_FILE}")
print(f"您的檔名列表已儲存在: {OUTPUT_FILENAMES_FILE}")