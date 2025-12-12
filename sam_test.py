# from sam3.model_builder import build_sam3_image_model
# from sam3.model.sam3_image_processor import Sam3Processor
# from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor
# import torch

# import numpy as np
# import torch
# from PIL import Image
# from typing import List, Dict, Any

# from sam3.model.sam3_image_processor import Sam3Processor

# from PIL import Image
# # from sam3_auto_mask_generator import Sam3AutomaticMaskGenerator

# class Sam3AutomaticMaskGenerator:
#     """
#     Very small SAM 3 companion similar in spirit to SAM 1's SamAutomaticMaskGenerator.

#     Given an image, it:
#       1) builds a grid of point prompts
#       2) runs SAM 3 point prompt branch for each point
#       3) filters masks on score and area
#       4) runs mask NMS
#       5) returns a list of instance masks and boxes
#     """

#     def __init__(
#         self,
#         processor: Sam3Processor,
#         points_per_side: int = 16,
#         pred_iou_thresh: float = 0.8,
#         min_mask_region_area: int = 1000,
#         max_mask_region_ratio: float = 0.5,
#         nms_iou_thresh: float = 0.5,
#         device: str = "cuda",
#     ) -> None:
#         self.processor = processor
#         self.points_per_side = points_per_side
#         self.pred_iou_thresh = pred_iou_thresh
#         self.min_mask_region_area = min_mask_region_area
#         self.max_mask_region_ratio = max_mask_region_ratio
#         self.nms_iou_thresh = nms_iou_thresh
#         self.device = device

#     def _point_grid(self, h: int, w: int) -> np.ndarray:
#         ys = np.linspace(0, h - 1, self.points_per_side, dtype=np.float32)
#         xs = np.linspace(0, w - 1, self.points_per_side, dtype=np.float32)
#         grid = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)  # [P, 2] (x, y)
#         return grid

#     @staticmethod
#     def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
#         # a, b: [H, W] bool
#         inter = np.logical_and(a, b).sum()
#         union = np.logical_or(a, b).sum()
#         return float(inter) / (float(union) + 1e-8)

#     def _nms(self, masks: List[np.ndarray], scores: List[float]):
#         if len(masks) == 0:
#             return [], []
#         masks_arr = np.stack([m.astype(bool) for m in masks], axis=0)  # [N, H, W]
#         scores_arr = np.asarray(scores, dtype=np.float32)

#         keep_indices: List[int] = []
#         order = scores_arr.argsort()[::-1]  # descending

#         while len(order) > 0:
#             i = int(order[0])
#             keep_indices.append(i)
#             if len(order) == 1:
#                 break
#             rest = order[1:]
#             ious = []
#             for j in rest:
#                 iou_ij = self._mask_iou(masks_arr[i], masks_arr[j])
#                 ious.append(iou_ij)
#             ious = np.asarray(ious)
#             order = rest[ious < self.nms_iou_thresh]

#         kept_masks = [masks_arr[i] for i in keep_indices]
#         kept_scores = [float(scores_arr[i]) for i in keep_indices]
#         return kept_masks, kept_scores

#     def generate(self, image: Image.Image | np.ndarray) -> List[Dict[str, Any]]:
#         """
#         Args:
#             image: PIL.Image or HxWx3 uint8 array

#         Returns:
#             A list of dicts with keys:
#               - segmentation: [H, W] bool mask
#               - area: int
#               - bbox: [x, y, w, h]
#               - box_xyxy: [x1, y1, x2, y2]
#               - predicted_iou: float
#         """

#         # convert to PIL internally
#         if isinstance(image, np.ndarray):
#             pil_image = Image.fromarray(image)
#             img_array = image
#         else:
#             pil_image = image
#             img_array = np.array(image)

#         h, w = img_array.shape[:2]
#         total_pixels = h * w

#         # 1) set image once, reuse state
#         inference_state = self.processor.set_image(pil_image)

#         # 2) grid of points over the image
#         grid_points = self._point_grid(h, w)

#         all_masks: List[np.ndarray] = []
#         all_scores: List[float] = []

#         for pt in grid_points:
#             # one positive point
#             points = np.asarray([pt], dtype=np.float32)   # [1, 2]
#             labels = np.asarray([1], dtype=np.int32)      # [1], 1 = foreground

#             with torch.no_grad():
#                 output = self.processor.set_point_prompt(
#                     state=inference_state,
#                     points=points,
#                     labels=labels,
#                 )

#             masks = output["masks"]          # [K, Hm, Wm] or [K, 1, Hm, Wm]
#             scores = output["scores"]        # [K]

#             # move to cpu
#             masks_np = masks.detach().cpu().numpy()
#             scores_np = scores.detach().cpu().numpy()

#             if masks_np.ndim == 4:  # [K, 1, Hm, Wm]
#                 masks_np = masks_np[:, 0]

#             # take best mask for this point
#             best_idx = int(scores_np.argmax())
#             best_score = float(scores_np[best_idx])
#             if best_score < self.pred_iou_thresh:
#                 continue

#             mask = masks_np[best_idx]

#             # resize to original image if needed
#             if mask.shape != (h, w):
#                 # nearest neighbor resize
#                 mask = np.array(
#                     Image.fromarray(mask).resize((w, h), resample=Image.NEAREST)
#                 )

#             mask_bin = mask > 0.5
#             area = int(mask_bin.sum())
#             if area < self.min_mask_region_area:
#                 continue
#             if area > self.max_mask_region_ratio * total_pixels:
#                 continue

#             all_masks.append(mask_bin)
#             all_scores.append(best_score)

#         if len(all_masks) == 0:
#             return []

#         # 3) mask NMS
#         final_masks, final_scores = self._nms(all_masks, all_scores)

#         # 4) pack results
#         results: List[Dict[str, Any]] = []
#         for m, s in zip(final_masks, final_scores):
#             rows = np.any(m, axis=1)
#             cols = np.any(m, axis=0)
#             if not rows.any() or not cols.any():
#                 continue
#             y1, y2 = np.where(rows)[0][[0, -1]]
#             x1, x2 = np.where(cols)[0][[0, -1]]
#             w_box = int(x2 - x1 + 1)
#             h_box = int(y2 - y1 + 1)

#             results.append(
#                 dict(
#                     segmentation=m,
#                     area=int(m.sum()),
#                     bbox=[int(x1), int(y1), w_box, h_box],
#                     box_xyxy=[int(x1), int(y1), int(x2), int(y2)],
#                     predicted_iou=float(s),
#                 )
#             )
#         return results
    

# device = "cuda" if torch.cuda.is_available() else "cpu"

# model = build_sam3_image_model().to(device)
# model.eval()

# processor = Sam3Processor(model)

# auto_gen = Sam3AutomaticMaskGenerator(
#     processor,
#     points_per_side=24,       # more points gives more coverage but is slower
#     pred_iou_thresh=0.85,     # keep high quality masks
#     min_mask_region_area=500, # tune for your resolution
# )

# image = Image.open("/storage/dulanga/4DRecon/event_monst3r/data/point_odyssey/test/ani2/rgbs/rgb_00000.jpg").convert("RGB")
# instances = auto_gen.generate(image)

# print("num instances:", len(instances))
# print(instances[0].keys()) 


from transformers import pipeline
from accelerate import Accelerator
from PIL import Image
import requests
import numpy as np
import random
import cv2

# Pick device index for HF pipeline
accel_device = Accelerator().device
device_index = 0 if accel_device.type == "cuda" else -1   # 0 for first GPU, -1 for CPU

# Create SAM 3 automatic mask generator
generator = pipeline(
    "mask-generation",
    model="facebook/sam3",
    device=device_index,
)

# Load image (same as your example)
raw_image = Image.open("/storage/dulanga/4DRecon/event_monst3r/data/point_odyssey/train/seminar_h52_ego1/rgbs/rgb_00025.jpg").convert("RGB")
image_np = np.array(raw_image)

# Zero shot instance segmentation for the whole image
# outputs = generator(raw_image, points_per_batch=128, pred_iou_thresh=0.7,stability_score_thresh=0.7)
outputs = generator(raw_image, points_per_batch=128,pred_iou_thresh=0.8,stability_score_thresh=0.7)

print(outputs.keys())
# outputs is a dict with masks, boxes, and scores
masks  = outputs["masks"]   # list/array of instance masks
# boxes  = outputs["boxes"]   # [N, 4] xyxy
scores = outputs["scores"]  # [N]

print("num instances:", len(masks))
print("first mask shape:", masks[0])


overlay = image_np.copy()

# Apply random colors to each mask
for i,mask in enumerate(masks):
    color = [random.randint(0, 255) for _ in range(3)]
    cv2.imwrite("tmp/mask_"+str(i)+".png", mask.cpu().numpy().astype(np.uint8)*255)
    # alpha blend mask area
    overlay[mask] = (0.6 * overlay[mask] + 0.4 * np.array(color)).astype(np.uint8)

# Save visualization
overlay_img = Image.fromarray(overlay)
overlay_img.save("sam3_instance_segmentation.png")

print("Saved:", "sam3_instance_segmentation.png")