import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from dust3r.ev_depth_eval import depth_evaluation, group_by_directory
import numpy as np
import cv2
import json
from tqdm import tqdm
import glob
from PIL import Image


# def depth_read(filename):
#     # loads depth map D from png file
#     # and returns it as a numpy array
#     depth_png = np.asarray(Image.open(filename))
#     # make sure we have a proper 16bit depth map here.. not 8bit!
#     # print(np.min(depth_png),np.max(depth_png))
#     assert np.max(depth_png) > 255
#     depth = depth_png.astype(np.float64) / 5000.0
#     depth[depth_png == 0] = -1.0
#     return depth

def depth_read(filename):
    # loads depth map D from png file
    # and returns it as a numpy array
    depth_png = np.asarray(Image.open(filename))
    # make sure we have a proper 16bit depth map here.. not 8bit!
    # print(np.min(depth_png),np.max(depth_png))
    # assert np.max(depth_png) > 255
    depth = depth_png.astype(np.float64) / 5000.0
    depth[depth_png == 0] = -1.0
    return depth

# seq_list = ["ani2","ani12_new_f"]
# seq_list = ["seminar_g110_0315_ego1"] # High
# seq_list = ["dancingroom1_3rd"] # Low
# seq_list = ["dancingroom1_3rd2"] # Medium
# seq_list = ["ani2"] # High obj motion
seq_list = ["seminar_g110_0315_3rd"] # Low obj motion

img_pathes_folder = [f"/storage/dulanga/4DRecon/event_monst3r/data/point_odyssey/test/{seq}/rgbs/*.jpg" for seq in seq_list]
img_pathes = []
for img_pathes_folder_i in img_pathes_folder:
    img_pathes += glob.glob(img_pathes_folder_i)
img_pathes = sorted(img_pathes)
depth_pathes_folder = [f"/storage/dulanga/4DRecon/event_monst3r/data/point_odyssey/test/{seq}/depths/*.png" for seq in seq_list]
depth_pathes = []
for depth_pathes_folder_i in depth_pathes_folder:
    depth_pathes += glob.glob(depth_pathes_folder_i)
depth_pathes = sorted(depth_pathes)
pred_pathes = glob.glob("/storage/dulanga/4DRecon/event_monst3r/results2/point_odyssey_eval/*/*.npy")    #TODO: update the path to your prediction
pred_pathes = sorted(pred_pathes)

def get_video_results():
    grouped_pred_depth = group_by_directory(pred_pathes)
    grouped_gt_depth = group_by_directory(depth_pathes, idx=-2)
    gathered_depth_metrics = []
    print(grouped_gt_depth.keys())
    print(grouped_pred_depth.keys())
    for key in tqdm(grouped_gt_depth.keys()):
        pd_pathes = grouped_pred_depth[key]
        gt_pathes = grouped_gt_depth[key]
        gt_depth = np.stack([depth_read(gt_path) for gt_path in gt_pathes], axis=0)
        pr_depth = np.stack([cv2.resize(np.load(pd_path), (gt_depth.shape[2], gt_depth.shape[1]), interpolation=cv2.INTER_CUBIC)
                              for pd_path in pd_pathes], axis=0)
        
        min_len = min(pr_depth.shape[0], gt_depth.shape[0])
        min_len = 64
        pr_depth = pr_depth[:min_len]
        gt_depth = gt_depth[:min_len]
        
        # for depth eval, set align_with_lad2=False to use median alignment; set align_with_lad2=True to use scale&shift alignment
        depth_results, error_map, depth_predict, depth_gt = depth_evaluation(pr_depth, gt_depth, max_depth=70, align_with_lad2=True, use_gpu=False)
        gathered_depth_metrics.append(depth_results)

    depth_log_path = 'tmp.json'
    average_metrics = {
        key: np.average(
            [metrics[key] for metrics in gathered_depth_metrics], 
            weights=[metrics['valid_pixels'] for metrics in gathered_depth_metrics]
        )
        for key in gathered_depth_metrics[0].keys() if key != 'valid_pixels'
    }
    print('Average depth evaluation metrics:', average_metrics)
    with open(depth_log_path, 'w') as f:
        f.write(json.dumps(average_metrics))

get_video_results()
