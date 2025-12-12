import numpy as np
import cv2
import os


event_path = "/storage/dulanga/4DRecon/event_monst3r/data/point_odyssey/train/ani/event_voxels/voxel_00010.npy"

voxel_grid = np.load(event_path)  # shape: [5, H, W]

# get the activity map by summing over the time dimension
activity_map = np.sum(np.abs(voxel_grid), axis=0)  # shape: [H, W]

# cv morphological operations to reduce noise
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
activity_map = cv2.morphologyEx(activity_map, cv2.MORPH_OPEN, kernel)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
activity_map = cv2.morphologyEx(activity_map, cv2.MORPH_CLOSE, kernel)

# kernel = np.ones((10, 10), np.uint8)
# activity_map = cv2.morphologyEx(activity_map, cv2.MORPH_OPEN, kernel)
# activity_map = cv2.morphologyEx(activity_map, cv2.MORPH_CLOSE, kernel)

actvity_map = activity_map.astype(np.uint8)*255

cv2.imwrite("activity_map.png", activity_map*255)
# cv2.imwrite("background.png", background)
print("Saved activity map to activity_map.png")

        # If you want to visualise:
        # display_mask = (mask * 255).astype(np.uint8)
        # cv2.imshow("foreground", display_mask)
        # if cv2.waitKey(1) == 27:  # escape key
        #     break

    # cv2.destroyAllWindows()

_, binary = cv2.threshold(
    activity_map.astype(np.uint8), 0, 255,
    cv2.THRESH_BINARY
)

# negate the binary for background
background = cv2.bitwise_not(binary)

cv2.imwrite("background.png", background)
cv2.imwrite("binary.png", binary)

