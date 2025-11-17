# write a list of frames to a video file

import cv2
import os
import numpy as np
from typing import List
from tqdm import tqdm
from PIL import Image
import torch
from torchvision.utils import save_image
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def write_to_video(frames: List[np.ndarray], output_path: str, fps: int = 30):
    """
    Write a list of frames to a video file.

    Args:
        frames (List[np.ndarray]): List of frames (H, W, 3) in uint8 format.
        output_path (str): Path to save the output video file.
        fps (int): Frames per second for the output video.
    """
    if len(frames) == 0:
        raise ValueError("No frames to write to video.")

    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for .mp4 files
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in tqdm(frames, desc="Writing video"):
        if frame.shape != (height, width, 3):
            raise ValueError(f"Frame shape {frame.shape} does not match expected shape {(height, width, 3)}.")
        video_writer.write(frame)

    video_writer.release()
    print(f"Video saved to {output_path}")

def save_frames_as_images(frames: List[np.ndarray], output_dir: str):
    """
    Save a list of frames as individual image files.

    Args:
        frames (List[np.ndarray]): List of frames (H, W, 3) in uint8 format.
        output_dir (str): Directory to save the output image files.
    """ 
    os.makedirs(output_dir, exist_ok=True)

    for idx, frame in enumerate(tqdm(frames, desc="Saving images")):
        if frame.ndim == 2:  # Grayscale image
            frame = np.stack([frame]*3, axis=-1)  # Convert to 3-channel
        elif frame.shape[2] == 1:  # Single channel image
            frame = np.concatenate([frame]*3, axis=-1)  # Convert to 3-channel
        elif frame.shape[2] != 3:
            raise ValueError(f"Frame shape {frame.shape} is not supported for saving as image.")
        
        image_path = os.path.join(output_dir, f"frame_{idx:05d}.png")
        cv2.imwrite(image_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    print(f"Images saved to {output_dir}")

def save_tensor_as_image(tensor: torch.Tensor, output_path: str):
    """
    Save a PyTorch tensor as an image file.

    Args:
        tensor (torch.Tensor): Tensor of shape (C, H, W) or (N, C, H, W) in [0, 1] range.
        output_path (str): Path to save the output image file.          
    """
    if tensor.ndim == 4:
        tensor = tensor[0]  # Take the first image in the batch
    elif tensor.ndim != 3:
        raise ValueError(f"Tensor shape {tensor.shape} is not supported for saving as image.")
    
    save_image(tensor, output_path)
    print(f"Image saved to {output_path}")

# read from a folder of images and write to a video file
def images_to_video(input_dir: str, output_path: str, fps: int = 30):
    """
    Read images from a folder and write them to a video file.

    Args:
        input_dir (str): Directory containing input image files.
        output_path (str): Path to save the output video file.
        fps (int): Frames per second for the output video.
    """
    image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    frames = []
    idx = 0 
    for image_file in tqdm(image_files, desc="Reading images"):
        image_path = os.path.join(input_dir, image_file)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not read image {image_path}. Skipping.")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        frames.append(image)
        idx+=1
        if idx >= 65:  # Limit to first 300 frames
            break
    
    # frames = frames[:65]  # Limit to first 300 frames

    write_to_video(frames, output_path, fps)    

if __name__ == "__main__":
    # Example usage
    # event_monst3r/data/point_odyssey/test/seminar_g110_0315_3rd.mp4
    input_dir = "/storage/dulanga/4DRecon/event_monst3r/data/point_odyssey/test/seminar_g110_0315_3rd/rgbs"
    output_video_path = "output_video.mp4"
    fps = 30

    images_to_video(input_dir, output_video_path, fps)