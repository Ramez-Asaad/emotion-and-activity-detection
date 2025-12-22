"""
UCF101 Dataset Loader
====================

Custom dataset class for loading UCF101 video frames.
"""

import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np


class UCF101Dataset(Dataset):
    """
    UCF101 dataset loader for video frames.
    
    Supports two modes:
    1. Pre-extracted frames (recommended)
    2. On-the-fly frame extraction from videos
    """
    
    def __init__(self, root_dir, transform=None, frames_per_video=1, 
                 frame_sampling='center', use_videos=False):
        """
        Initialize UCF101 dataset.
        
        Args:
            root_dir (str): Root directory with class folders
            transform: Torchvision transforms to apply
            frames_per_video (int): Number of frames to sample per video
            frame_sampling (str): 'center', 'random', or 'uniform'
            use_videos (bool): If True, load from .avi files; if False, load from frame images
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.frames_per_video = frames_per_video
        self.frame_sampling = frame_sampling
        self.use_videos = use_videos
        
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.samples = self._load_samples()
    
    def _load_samples(self):
        """Load all samples (video paths or frame paths)."""
        samples = []
        
        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            class_idx = self.class_to_idx[class_name]
            
            if self.use_videos:
                # Load video files (.avi, .mp4)
                video_files = list(class_dir.glob('*.avi')) + list(class_dir.glob('*.mp4'))
                for video_path in video_files:
                    samples.append((str(video_path), class_idx))
            else:
                # Load frame images (assumes frames are organized in subfolders)
                # Structure: root_dir/class_name/video_name/frame_001.jpg
                video_folders = [d for d in class_dir.iterdir() if d.is_dir()]
                
                if video_folders:
                    # Frames organized in video folders
                    for video_folder in video_folders:
                        frame_files = sorted(list(video_folder.glob('*.jpg')) + 
                                           list(video_folder.glob('*.png')))
                        if frame_files:
                            samples.append((video_folder, class_idx))
                else:
                    # Frames directly in class folder (treat each image as a sample)
                    frame_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
                    for frame_path in frame_files:
                        samples.append((str(frame_path), class_idx))
        
        return samples
    
    def _extract_frames_from_video(self, video_path):
        """Extract frames from video file."""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if self.frame_sampling == 'center':
            # Extract center frame
            frame_indices = [total_frames // 2]
        elif self.frame_sampling == 'uniform':
            # Uniformly sample frames
            frame_indices = np.linspace(0, total_frames - 1, 
                                       self.frames_per_video, dtype=int)
        else:  # random
            # Randomly sample frames
            frame_indices = np.random.choice(total_frames, 
                                           self.frames_per_video, 
                                           replace=False)
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame))
        
        cap.release()
        return frames
    
    def _load_frames_from_folder(self, folder_path):
        """Load frames from a folder of images."""
        frame_files = sorted(list(folder_path.glob('*.jpg')) + 
                           list(folder_path.glob('*.png')))
        
        if not frame_files:
            raise ValueError(f"No frames found in {folder_path}")
        
        total_frames = len(frame_files)
        
        if self.frame_sampling == 'center':
            # Load center frame
            frame_indices = [total_frames // 2]
        elif self.frame_sampling == 'uniform':
            # Uniformly sample frames
            frame_indices = np.linspace(0, total_frames - 1, 
                                       self.frames_per_video, dtype=int)
        else:  # random
            # Randomly sample frames
            frame_indices = np.random.choice(total_frames, 
                                           min(self.frames_per_video, total_frames), 
                                           replace=False)
        
        frames = []
        for idx in frame_indices:
            frame = Image.open(frame_files[idx]).convert('RGB')
            frames.append(frame)
        
        return frames
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get a sample.
        
        Returns:
            tuple: (image, label) where image is a tensor
        """
        sample_path, label = self.samples[idx]
        
        # Load frame(s)
        if self.use_videos:
            frames = self._extract_frames_from_video(sample_path)
        else:
            if Path(sample_path).is_dir():
                frames = self._load_frames_from_folder(Path(sample_path))
            else:
                # Single image file
                frames = [Image.open(sample_path).convert('RGB')]
        
        # For now, use only the first frame (can be extended for multi-frame models)
        frame = frames[0]
        
        # Apply transforms
        if self.transform:
            frame = self.transform(frame)
        
        return frame, label


class UCF101ImageFolder(Dataset):
    """
    Simplified UCF101 loader using ImageFolder structure.
    
    Expected structure:
    root_dir/
        class1/
            image1.jpg
            image2.jpg
            ...
        class2/
            image1.jpg
            ...
    """
    
    def __init__(self, root_dir, transform=None):
        """
        Initialize simple image folder dataset.
        
        Args:
            root_dir (str): Root directory with class folders
            transform: Torchvision transforms to apply
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.samples = []
        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            class_idx = self.class_to_idx[class_name]
            
            # Load all image files
            image_files = (list(class_dir.glob('*.jpg')) + 
                          list(class_dir.glob('*.png')) + 
                          list(class_dir.glob('*.jpeg')))
            
            for img_path in image_files:
                self.samples.append((str(img_path), class_idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
