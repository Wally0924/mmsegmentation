import numpy as np
import json

IMAGE_FEATURES_FILE = "all_image_features.npy"        # 01_extract_feature.py 輸出的圖像向量
TEXT_FEATURES_FILE = "all_text_features.npy"    # 01b_text_vector.py 輸出的文字向量
FILENAMES_FILE = "image_filenames.json"         # 檔名順序
OUTPUT_FEATURES_FILE = "all_joint_features.npy" # 新的融合特徵檔

print("Loading image features...")
img_feats = np.load(IMAGE_FEATURES_FILE)
print("Loading text features...")
txt_feats = np.load(TEXT_FEATURES_FILE)

if img_feats.shape[0] != txt_feats.shape[0]:
    print(f"ERROR: image features ({img_feats.shape[0]}) and text features ({txt_feats.shape[0]}) have different counts!")
    exit()

print(f"Image feature dim = {img_feats.shape[1]}, text feature dim = {txt_feats.shape[1]}")

# 方式一：直接拼接 (推薦先用這個試試)
joint_feats = np.concatenate([img_feats, txt_feats], axis=1)

print(f"Joint feature shape = {joint_feats.shape}")
print(f"Saving joint features to {OUTPUT_FEATURES_FILE}...")
np.save(OUTPUT_FEATURES_FILE, joint_feats)

# 可選：把 filenames 再檢查一次
with open(FILENAMES_FILE, "r") as f:
    filenames = json.load(f)
print(f"Joint features count = {joint_feats.shape[0]}, filenames count = {len(filenames)}")
print("01c_merge_features.py: Done.")
