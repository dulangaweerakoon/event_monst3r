import os
import numpy as np
import glob
import tqdm
import cv2
import h5py
from joblib import Parallel, delayed

N = 5  # Number of temporal subdivisions per frame
# find all rgb_paths in directory 
root_path = "/storage/dulanga/4DRecon/event_monst3r/data/point_odyssey/train/"
event_root = "/storage/dulanga/4DRecon/event_monst3r/data/point_odyssey/slowmo_events/train_slomo_h5/"
folders = [f for f in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, f))]

print(f"Found folders: {folders}")

prior_event_path = None

for idx,folder in enumerate(folders):
    print(f"Processing folder: {folder}")
    rgb_path = os.path.join(root_path, folder, "rgbs/")
    event_path = os.path.join(event_root, f"{folder}.h5")

    # check if event file exists
    if not os.path.exists(event_path):
        print(f"Event file not found for folder {folder}, skipping...")
        continue
    # make a folder to save output inside root_path if not exists
    out_path = os.path.join(root_path, folder, "event_voxels/")

    #check if out_path exists, if so skip
    if os.path.exists(out_path):
        print(f"Output path {out_path} already exists, skipping...")
        continue
    os.makedirs(out_path, exist_ok=True)
    # rgb_path = "/Users/dulanga/Documents/M3S/4DRecon/test/po_test_slomo/ani2/rgbs/"
    # event_path = "/Users/dulanga/Documents/M3S/4DRecon/test/po_test_slomo/ani2.h5"
    # out_path = "/Users/dulanga/Documents/M3S/4DRecon/out/"

    # --- Read event data ---
    # print(f"Reading event data from: {event_path}")
    h5_file = h5py.File(event_path, 'r')
    try:
        event_data = h5_file['events'][:]
    except KeyError:
        print(f"Key 'events' not found in {event_path}, skipping...")
        h5_file.close()
        continue
    # event_data = h5_file['events'][:]
    h5_file.close()

    rgb_files = sorted(glob.glob(os.path.join(rgb_path, '*.jpg')))
    num_frames = len(rgb_files)
    min_time = event_data[0][0]
    max_time = event_data[-1][0]
    time_segments = np.linspace(min_time, max_time, num_frames + 1)

    # --- Define the function for one frame ---
    def process_frame(i):
        start_time = time_segments[i]
        end_time = time_segments[i + 1]
        
        # Filter events in this frame range
        segment_events = event_data[(event_data[:, 0] >= start_time) & (event_data[:, 0] < end_time)]
        
        # Subdivide into N subsegments
        sub_time_segments = np.linspace(start_time, end_time, N + 1)
        voxel_grid = np.zeros((N, 540, 960), dtype=np.float32)

        for n in range(N):
            sub_start = sub_time_segments[n]
            sub_end = sub_time_segments[n + 1]
            sub_events = segment_events[(segment_events[:, 0] >= sub_start) & (segment_events[:, 0] < sub_end)]
            for t, x, y, p in sub_events:
                if p == 0:
                    p = -1
                voxel_grid[n, int(y), int(x)] += p

        # Save voxel grid as a .npy file
        voxel_filename = os.path.join(out_path, f'voxel_{i:05d}.npy')
        np.save(voxel_filename, voxel_grid)
        return voxel_filename

    # --- Run in parallel using joblib ---
    results = Parallel(n_jobs=64, backend="loky")(
        delayed(process_frame)(i) for i in tqdm.tqdm(range(num_frames), desc="Generating Voxel Grids")
    )

    # delete h5 file from storage
    # if idx > 0 and prior_event_path is not None:
    #     os.remove(prior_event_path)
    # prior_event_path = event_path
    # os.remove(event_path)

    # for visualization testing
    # num_frames = 30
    # print("✅ All voxel grids generated!")

    # # --- Optional: Sequential visualization ---
    # video_filename = 'voxel_visualization.avi'
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = None

    # for i in tqdm.tqdm(range(num_frames), desc="Writing Video"):
    #     voxel_grid = np.load(os.path.join(out_path, f'voxel_{i:05d}.npy'))
    #     rgb_image = cv2.imread(rgb_files[i])
    #     rgb_resized = cv2.resize(rgb_image, (960, 540))
    #     for n in range(N):
    #         voxel_viz = ((voxel_grid[n] + 1) / 2 * 255).astype(np.uint8)
    #         combined = np.hstack((rgb_resized, cv2.cvtColor(voxel_viz, cv2.COLOR_GRAY2BGR)))
    #         if out is None:
    #             out = cv2.VideoWriter(video_filename, fourcc, 10.0, (combined.shape[1], combined.shape[0]))
    #         out.write(combined)

    # if out:
    #     out.release()

    # print("🎬 Video writing complete!")
