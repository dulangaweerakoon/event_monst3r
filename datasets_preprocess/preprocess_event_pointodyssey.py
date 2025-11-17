# find all the folders in the dataset folder
import os
import shutil
import sys
from pathlib import Path
from typing import List
import numpy as np
import cv2


dataset_folder = "/storage/dulanga/4DRecon/event_monst3r/data/point_odyssey"
# find all the folders in the train folder
train_folder = os.path.join(dataset_folder, "train")
test_folder = os.path.join(dataset_folder, "test")

# get all the folders in the train folder
train_folders = [os.path.basename(f.path) for f in os.scandir(train_folder) if f.is_dir()]
test_folders = [os.path.basename(f.path) for f in os.scandir(test_folder) if f.is_dir()]

print(f"Found {len(train_folders)} train folders")
print(f"Found {len(test_folders)} test folders")

# print(train_folders)

# get the raw events folder inside the dataset folder underra_events folder
raw_events_folder = os.path.join(dataset_folder, "raw_events")

# get raw events train and test folders
raw_events_train_folder = os.path.join(raw_events_folder, "po_train")
raw_events_test_folder = os.path.join(raw_events_folder, "po_all_test")

# check if the file in the train folders are in the raw events train folder
for folder in train_folders:
    file_name = raw_events_train_folder + "/" + folder + "_events.avi"
    if not os.path.exists(file_name):
        print(f"File {file_name} does not exist")
        # sys.exit(1)

# check if the file in the test folders are in the raw events test folder
for folder in test_folders:
    file_name = raw_events_test_folder + "/" + folder + "_events.avi"
    if not os.path.exists(file_name):
        print(f"File {file_name} does not exist")
        # sys.exit(1)



# create a new folder under each train and test folder called events
for folder in train_folders:
    events_folder = os.path.join(train_folder, folder, "events")
    if not os.path.exists(events_folder):
        os.makedirs(events_folder)


for folder in test_folders:
    events_folder = os.path.join(test_folder, folder, "events")
    if not os.path.exists(events_folder):
        os.makedirs(events_folder)

# go through each train folder and copy the frames from the raw events train folder to the events folder
for folder in train_folders:
    file_name = raw_events_train_folder + "/" + folder + "_events.avi"
    cap = cv2.VideoCapture(file_name)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing {file_name} with {frame_count} frames")
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # save the frame as a png file in the events folder
        events_folder = os.path.join(train_folder, folder, "events")
        frame_file_name = os.path.join(events_folder, f"events_{frame_idx:05d}.png")
        # print(f"Saving frame {frame_idx} to {frame_file_name}")
        cv2.imwrite(frame_file_name, frame)
        frame_idx += 1
    cap.release()
    print(f"Saved {frame_idx} frames to {events_folder}")


# go through each test folder and copy the frames from the raw events test folder to the events folder
# for folder in test_folders:
#     file_name = raw_events_test_folder + "/" + folder + "_events.avi"
#     cap = cv2.VideoCapture(file_name)
#     frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     print(f"Processing {file_name} with {frame_count} frames")
#     frame_idx = 0
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         # save the frame as a png file in the events folder
#         events_folder = os.path.join(test_folder, folder, "events")
#         frame_file_name = os.path.join(events_folder, f"events_{frame_idx:05d}.png")
#         # print(f"Saving frame {frame_idx} to {frame_file_name}")
#         cv2.imwrite(frame_file_name, frame)
#         frame_idx += 1
#     cap.release()
#     print(f"Saved {frame_idx} frames to {events_folder}")