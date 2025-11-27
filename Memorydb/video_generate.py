import cv2
import os
import glob
import numpy as np
from tqdm import tqdm
import re

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def create_repeat_frame_video(image_folder: str, output_video_path: str, fps: int = 30, repeat_times: int = 5):
    """
    製作無殘影慢動作影片 (使用影格重複)。

    Args:
        fps (int): 影片的輸出幀率 (例如 30)。
        repeat_times (int): 每一張圖片要「停留」幾個 Frame。
                            數值越大 -> 影片越慢 (但也越像幻燈片)。
    """
    
    if not os.path.exists(image_folder):
        print("資料夾不存在")
        return

    types = ('*.jpg', '*.jpeg', '*.png', '*.bmp') 
    images = []
    for files in types:
        images.extend(glob.glob(os.path.join(image_folder, files)))
    
    images.sort(key=natural_sort_key)

    if not images:
        print("找不到圖片")
        return

    print(f"原始圖片張數: {len(images)}")
    
    # 初始化 VideoWriter
    first_frame = cv2.imread(images[0])
    height, width, _ = first_frame.shape
    
    if output_video_path.endswith('.mp4'):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    else:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')

    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    print(f"正在生成影片... (FPS={fps}, 每張圖重複={repeat_times}次)")
    
    # --- 核心邏輯：重複寫入 ---
    for i in tqdm(range(len(images))):
        img_path = images[i]
        img = cv2.imread(img_path)
        
        if img is None: continue
        
        # 防呆：縮放
        if img.shape[:2] != (height, width):
            img = cv2.resize(img, (width, height))

        # [關鍵修改] 不做混合，而是直接把這張圖寫入 N 次
        # 這會讓這張圖在影片中「停留」一段時間
        for _ in range(repeat_times):
            out.write(img)

    out.release()
    print(f"\n✅ 影片完成！已儲存至: {output_video_path}")
    
    # 計算影片長度
    total_frames = len(images) * repeat_times
    duration = total_frames / fps
    print(f"影片總長度約: {duration:.1f} 秒")

if __name__ == "__main__":
    INPUT_FOLDER = "data/training_images/cloudy_img/6am"
    OUTPUT_FILE = "output_hard_repeat.mp4"
    
    # [調整設定]
    # FPS = 30 : 讓播放器以流暢的頻率更新畫面
    # REPEAT = 5 : 每張圖重複 5 次 (相當於每張圖停留 5/30 = 0.16 秒)
    # 如果您覺得太快，就把 REPEAT 改大 (例如 10, 15)
    FPS = 30
    REPEAT_TIMES = 6

    create_repeat_frame_video(INPUT_FOLDER, OUTPUT_FILE, FPS, REPEAT_TIMES)