# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
import datetime  # 匯入 datetime 模組
import os.path   # 匯入 os.path 模組

from mmseg.apis import MMSegInferencer


def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('model', help='Config file')
    parser.add_argument('--checkpoint', default=None, help='Checkpoint file')
    parser.add_argument(
        '--out-dir', default='', help='Path to save result file')
    
    # [NEW] 新增 --exp-name 參數，用於自訂實驗名稱
    parser.add_argument(
        '--exp-name', 
        default=None, 
        help='Specific experiment name for the output folder. '
             'If not set, will use the parent folder name of the config file.')
    
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='Whether to display the drawn image.')
    parser.add_argument(
        '--dataset-name',
        default='cityscapes',
        help='Color palette used for segmentation map')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument(
        '--with-labels',
        action='store_true',
        default=False,
        help='Whether to display the class labels.')
    args = parser.parse_args()

    # [MODIFIED] 檢查 out-dir 是否有指定，若有，則添加模型名稱和時間戳
    if args.out_dir:
        # 1. 產生時間戳字串, e.g., '20251027_183000'
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 2. [MODIFIED] 決定實驗資料夾名稱
        if args.exp_name:
            # 優先使用使用者手動指定的 --exp-name
            model_folder_name = args.exp_name
        else:
            # 自動模式：抓取 config 檔案的 "上一層資料夾" 名稱
            # e.g., 'configs/mask2former/config.py' -> 'configs/mask2former'
            config_parent_dir = os.path.dirname(args.model)
            # e.g., 'configs/mask2former' -> 'mask2former'
            model_folder_name = os.path.basename(config_parent_dir)

        # 3. 組合新的巢狀路徑: base_dir/model_folder_name/timestamp/
        final_out_dir = os.path.join(args.out_dir, model_folder_name, timestamp)
    else:
        # 如果使用者沒有指定 out_dir，則保持為空字串 (不儲存檔案)
        final_out_dir = args.out_dir

    # build the model from a config file and a checkpoint file
    mmseg_inferencer = MMSegInferencer(
        args.model,
        args.checkpoint,
        dataset_name=args.dataset_name,
        device=args.device)

    # test a single image
    mmseg_inferencer(
        args.img,
        show=args.show,
        out_dir=final_out_dir,  # 使用我們組合好的新路徑
        opacity=args.opacity,
        with_labels=args.with_labels)


if __name__ == '__main__':
    main()