import numpy as np
import cv2

# event_path = "/storage/dulanga/4DRecon/event_monst3r/data/point_odyssey/train/ani/event_voxels/voxel_00001.npy"

# # load event voxel data
# V_k = np.load(event_path)

# S_k = np.sum(np.abs(V_k), axis=0)   # shape [H, W]
# S_k_blur = cv2.GaussianBlur(S_k.astype(np.float32), (3, 3), 0)

# alpha = 0.02            # how fast background adapts
# B = None                # background activity map, shape [H, W]


# import numpy as np
# import cv2



class EventForegroundLastBinDetector:
    """
    Detect moving objects in the last time bin of an event voxel grid.

    Input: V with shape [T, H, W], T = 5
    - Use bins 0..T-2 to estimate background activity per pixel
    - Use bin T-1 as the current frame
    - Foreground = pixels where last bin activity is significantly higher than background
    """

    def __init__(
        self,
        tau: float = 3.0,      # threshold on normalised difference
        min_area: int = 20,    # minimum connected component area
        blur_kernel: int = 3,  # spatial blur size
    ):
        self.tau = tau
        self.min_area = min_area
        self.blur_kernel = blur_kernel

    def _build_background_and_current(self, V: np.ndarray):
        """
        From voxel grid V [T, H, W], return:
          B: background map [H, W] from first T-1 bins
          C: current map [H, W] from last bin
        """
        T, H, W = V.shape
        if T < 2:
            raise ValueError("Need at least 2 time bins to form background and current.")

        # Absolute activity
        V_abs = np.abs(V).astype(np.float32)  # [T, H, W]

        # Background from first T-1 bins (mean over these bins)
        B = np.mean(V_abs[:-1, :, :], axis=0)  # [H, W]

        # Current from last bin
        C = V_abs[-1, :, :]  # [H, W]

        # Small blur to reduce noise
        if self.blur_kernel is not None and self.blur_kernel > 1:
            k = (self.blur_kernel, self.blur_kernel)
            B = cv2.GaussianBlur(B, k, 0)
            C = cv2.GaussianBlur(C, k, 0)

        return B, C

    def _foreground_mask(self, B: np.ndarray, C: np.ndarray) -> np.ndarray:
        """
        Compute foreground mask from background B and current C.

        B, C: [H, W]
        Returns:
            F: binary mask [H, W] with 0 or 1
        """
        # Raw difference: current minus background
        D = C - B

        # Only care about positive differences
        D[D < 0] = 0.0

        # Normalise by background level to get something like a z score
        R = D / (np.sqrt(B) + 1e-3)

        # Threshold
        F = (R > self.tau).astype(np.uint8)

        return F

    def _clean_and_label(self, F: np.ndarray):
        """
        Clean mask with morphology and get connected components.

        F: [H, W] binary

        Returns:
            F_clean: cleaned mask [H, W]
            blobs: list of dicts with label, area, bbox, centroid
        """
        kernel = np.ones((3, 3), np.uint8)
        F_clean = cv2.morphologyEx(F, cv2.MORPH_OPEN, kernel)
        F_clean = cv2.morphologyEx(F_clean, cv2.MORPH_CLOSE, kernel)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(F_clean)

        blobs = []
        for i in range(1, num_labels):  # skip background label 0
            area = stats[i, cv2.CC_STAT_AREA]
            if area < self.min_area:
                continue

            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            cx, cy = centroids[i]

            blobs.append(
                {
                    "label": int(i),
                    "area": int(area),
                    "bbox": (int(x), int(y), int(w), int(h)),
                    "centroid": (float(cx), float(cy)),
                }
            )

        return F_clean, blobs

    def process(self, V: np.ndarray):
        """
        Main entry.

        Args:
            V: numpy array with shape [T, H, W], T = 5

        Returns:
            mask: cleaned foreground mask [H, W] (uint8, 0 or 1)
            blobs: list of moving object regions
        """
        if V.ndim != 3:
            raise ValueError("V must have shape [T, H, W].")
        T, H, W = V.shape
        if T != 5:
            raise ValueError(f"Expected T = 5, got {T}.")

        # Build background from first 4 bins, current from last bin
        B, C = self._build_background_and_current(V)

        cv2.imwrite(f"background.png", (B / (B.max() + 1e-6) * 255).astype(np.uint8))
        cv2.imwrite(f"current.png", (C / (C.max() + 1e-6) * 255).astype(np.uint8))

        # Foreground mask for last bin
        F = self._foreground_mask(B, C)

        # save foreground mask as a png for debugging
        

        # Clean and get blobs
        F_clean, blobs = self._clean_and_label(F)

        cv2.imwrite(f"foreground_basic.png", (F_clean * 255).astype(np.uint8))

        return F_clean, blobs


# Example usage
if __name__ == "__main__":
    H, W = 540, 960
    detector = EventForegroundLastBinDetector(
        tau=3.0,
        min_area=20,
        blur_kernel=3,
    )

    # Simulate a stream of 100 frames
    # for k in range(100):
        # Replace this with your real voxel grid [5, H, W]
    event_path = "/storage/dulanga/4DRecon/event_monst3r/data/point_odyssey/train/ani/event_voxels/voxel_00010.npy"

    # load event voxel data
    V_k = np.load(event_path)
    # print("Loaded voxel grid shape:", V_k.shape)

    # Example: add a small moving blob to test
    # if 20 <= k < 40:
    #     y0, x0 = 50, 50 + (k - 20)
    #     V_k[:, y0:y0+10, x0:x0+10] += 5

    mask, blobs = detector.process(V_k)

    print(f"Frame, found {len(blobs)} moving blobs")
    for b in blobs:
        print("  label:", b["label"], "area:", b["area"], "bbox:", b["bbox"])

        # If you want to visualise:
        # display_mask = (mask * 255).astype(np.uint8)
        # cv2.imshow("foreground", display_mask)
        # if cv2.waitKey(1) == 27:  # escape key
        #     break

    # cv2.destroyAllWindows()
