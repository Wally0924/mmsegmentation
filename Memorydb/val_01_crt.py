# 匯入 Numpy
import numpy as np

# 匯入 json (用來讀取檔名)
import json

# 載入您剛剛生成的 .npy 檔案
features = np.load("all_features.npy")

# 載入您剛剛生成的 .json 檔案
with open("image_filenames.json", 'r') as f:
    filenames = json.load(f)


print(features.shape)
print(len(filenames))
print(features.dtype)
print(filenames[0])
print(filenames[-1])