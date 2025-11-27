import numpy as np
import json
import os
from typing import Tuple

# --- 1. 設定檔案路徑與權重 ---
# 檔案路徑
IMAGE_FEATURES_FILE = "all_image_features.npy"        # 圖像特徵向量檔案路徑
TEXT_FEATURES_FILE = "all_text_features.npy"    # 文字特徵向量檔案路徑
FILENAMES_FILE = "image_filenames.json"         # 檔名順序檔案路徑
OUTPUT_FEATURES_FILE = "all_joint_features.npy" # 輸出加權融合特徵檔案路徑

# 權重設定
# 圖像特徵的權重
IMAGE_WEIGHT = 0.65
# 文字特徵的權重
TEXT_WEIGHT = 0.35


def load_features(file_path: str) -> np.ndarray:
    """
    載入指定路徑的 .npy 特徵檔案。

    Args:
        file_path (str): .npy 檔案的路徑。

    Returns:
        np.ndarray: 載入的特徵矩陣 (N, D)，其中 N 是樣本數，D 是特徵維度。
    
    Raises:
        FileNotFoundError: 如果找不到指定的檔案。
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"錯誤：找不到特徵檔案 {file_path}！")
    print(f"Loading features from {file_path}...")
    return np.load(file_path)


def merge_features_weighted_concatenate(
    img_feats: np.ndarray, 
    txt_feats: np.ndarray, 
    img_weight: float, 
    txt_weight: float
) -> np.ndarray:
    """
    將圖像特徵和文字特徵進行加權處理後，使用 numpy.concatenate 進行拼接。

    拼接操作會將特徵維度相加 (例如 D1 + D2)。

    Args:
        img_feats (np.ndarray): 圖像特徵矩陣 (N, D_img)。
        txt_feats (np.ndarray): 文字特徵矩陣 (N, D_txt)。
        img_weight (float): 圖像特徵的權重。
        txt_weight (float): 文字特徵的權重。

    Returns:
        np.ndarray: 融合後的新特徵矩陣 (N, D_img + D_txt)。
    
    Raises:
        ValueError: 如果圖像和文字的樣本數量 N 不一致。
    """
    
    # 檢查特徵數量是否一致 (N, 樣本數)
    if img_feats.shape[0] != txt_feats.shape[0]:
        raise ValueError(
            f"ERROR: 圖像特徵數量 ({img_feats.shape[0]}) 與文字特徵數量 ({txt_feats.shape[0]}) 不一致！"
        )
    
    print(f"原始圖像特徵維度 (D_img) = {img_feats.shape[1]}")
    print(f"原始文字特徵維度 (D_txt) = {txt_feats.shape[1]}")
    
    # 1. 應用權重：將特徵矩陣的數值乘以對應的權重
    weighted_img = img_feats * img_weight
    weighted_txt = txt_feats * txt_weight
    
    # 2. 拼接：沿著特徵維度 (axis=1) 進行拼接
    joint_feats = np.concatenate([weighted_img, weighted_txt], axis=1)
    
    print(f"融合特徵形狀 = {joint_feats.shape}")
    return joint_feats


def main():
    """
    主要執行函式，負責載入特徵、加權拼接特徵並儲存結果。
    """
    
    # 1. 載入特徵
    try:
        img_feats = load_features(IMAGE_FEATURES_FILE)
        txt_feats = load_features(TEXT_FEATURES_FILE)
    except FileNotFoundError as e:
        print(e)
        return
    except Exception as e:
        print(f"載入檔案時發生錯誤: {e}")
        return

    print(f"使用的權重 -> 圖像: {IMAGE_WEIGHT}, 文字: {TEXT_WEIGHT}")

    # 2. 執行加權拼接
    try:
        joint_feats = merge_features_weighted_concatenate(
            img_feats, 
            txt_feats, 
            IMAGE_WEIGHT, 
            TEXT_WEIGHT
        )
        
    except ValueError as e:
        print(e)
        return

    # 3. 儲存結果
    print(f"正在儲存融合特徵到 {OUTPUT_FEATURES_FILE}...")
    np.save(OUTPUT_FEATURES_FILE, joint_feats)

    # 4. 檢查檔案名稱數量
    try:
        with open(FILENAMES_FILE, "r") as f:
            filenames = json.load(f)
        print(f"融合特徵數量 = {joint_feats.shape[0]}, 檔名數量 = {len(filenames)}")
    except FileNotFoundError:
        print(f"警告: 找不到檔名檔案 {FILENAMES_FILE}，跳過數量檢查。")

    print("\n---")
    print("01c_merge_feature_weighted_concat.py: 加權拼接完成。")


if __name__ == "__main__":
    main()