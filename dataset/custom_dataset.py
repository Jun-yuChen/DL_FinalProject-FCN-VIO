import sys
sys.path.append('..')
import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from pathlib import Path
from utils.utils import rotationError, read_pose_from_text
from utils import custom_transform
from collections import Counter
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang
from scipy.ndimage import convolve1d

IMU_FREQ = 10  # Your golden position is at 10Hz
IMU_SAMPLE_RATE = 100  # Your IMU is at 100Hz

class CustomDataset(Dataset):
    def __init__(self, root,
                 sequence_length=11,
                 train_seqs=['seq_01', 'seq_02'],  # Your sequence names
                 transform=None):
        """
        Custom Dataset Loader
        
        Args:
            root: Root directory containing your dataset
            sequence_length: Number of frames in each segment
            train_seqs: List of sequence folder names
            transform: Transform pipeline
            
        Expected directory structure:
            root/
                seq_01/
                    imu.txt          # IMU data at 100Hz (space-separated: ax ay az gx gy gz)
                    poses.txt        # Ground truth poses at 10Hz (KITTI format)
                    images/          # Image folder at 10Hz
                        000008.png
                        000024.png
                        ...
                seq_02/
                    ...
        """
        
        self.root = Path(root)
        self.sequence_length = sequence_length
        self.transform = transform
        self.train_seqs = train_seqs
        self.make_dataset()
    
    def load_imu_data(self, imu_path):
        """
        Load IMU data from txt/csv file
        
        Args:
            imu_path: Path to the IMU file (txt or csv)
            
        Returns:
            numpy array of shape (N, 6) containing [ax, ay, az, gx, gy, gz]
        """
        # Try space-separated format first (most common for your data)
        try:
            df = pd.read_csv(imu_path, delim_whitespace=True, header=None)
            if df.shape[1] >= 6:
                imu_data = df.values[:, :6]
                return imu_data.astype(np.float32)
        except:
            pass
        
        # Try standard CSV format
        try:
            df = pd.read_csv(imu_path)
            # Check if columns are properly named
            if 'ax' in df.columns:
                imu_data = df[['ax', 'ay', 'az', 'gx', 'gy', 'gz']].values
            elif df.shape[1] >= 6:
                imu_data = df.values[:, :6]
            else:
                raise ValueError("CSV does not have enough columns")
            return imu_data.astype(np.float32)
        except:
            pass
        
        # Last resort: manual parsing
        imu_data = []
        with open(imu_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):  # Skip comments
                    # Split by spaces and take first 6 values
                    values = line.split()
                    if len(values) >= 6:
                        imu_data.append([float(v) for v in values[:6]])
        
        if len(imu_data) == 0:
            raise ValueError(f"Could not load IMU data from {imu_path}")
        
        return np.array(imu_data).astype(np.float32)
    
    def make_dataset(self):
        sequence_set = []
        
        for folder in self.train_seqs:
            seq_path = self.root / folder
            
            # Load pose data
            pose_file = seq_path / 'poses.txt'
            poses, poses_rel = read_pose_from_text(str(pose_file))
            
            # Load IMU data
            imu_file = seq_path / 'imu.txt'  # Changed from imu.csv to imu.txt
            if not imu_file.exists():
                imu_file = seq_path / 'imu.csv'  # Fallback to .csv if .txt doesn't exist
            imus = self.load_imu_data(str(imu_file))
            
            # Get sorted image files
            img_folder = seq_path / 'images'
            fpaths = sorted(img_folder.glob("*.png"))
            
            # Verify data alignment
            num_poses = len(poses)
            num_images = len(fpaths)
            
            if num_poses != num_images:
                print(f"Warning: In {folder}, number of poses ({num_poses}) != number of images ({num_images})")
                # Use the minimum to avoid index errors
                num_frames = min(num_poses, num_images)
            else:
                num_frames = num_poses
            
            # Verify IMU data length (should be ~10x the number of frames for 100Hz IMU with 10Hz poses)
            expected_imu_samples = num_frames * (IMU_SAMPLE_RATE // IMU_FREQ)
            if len(imus) < expected_imu_samples:
                print(f"Warning: In {folder}, insufficient IMU data. Expected ~{expected_imu_samples}, got {len(imus)}")
            
            # Create segments
            for i in range(num_frames - self.sequence_length):
                img_samples = fpaths[i:i+self.sequence_length]
                
                # Extract IMU samples for this segment
                # For 100Hz IMU and 10Hz poses, we need 10 IMU samples per frame interval
                imu_start_idx = i * (IMU_SAMPLE_RATE // IMU_FREQ)
                imu_end_idx = (i + self.sequence_length - 1) * (IMU_SAMPLE_RATE // IMU_FREQ) + 1
                
                if imu_end_idx > len(imus):
                    print(f"Warning: In {folder}, segment {i} exceeds IMU data length. Skipping.")
                    continue
                
                imu_samples = imus[imu_start_idx:imu_end_idx]
                
                pose_samples = poses[i:i+self.sequence_length]
                pose_rel_samples = poses_rel[i:i+self.sequence_length-1]
                
                # Calculate rotation for weighting
                segment_rot = rotationError(pose_samples[0], pose_samples[-1])
                
                sample = {
                    'imgs': img_samples,
                    'imus': imu_samples,
                    'gts': pose_rel_samples,
                    'rot': segment_rot
                }
                sequence_set.append(sample)
        
        self.samples = sequence_set
        
        # Generate weights based on rotation (same as KITTI dataset)
        rot_list = np.array([np.cbrt(item['rot']*180/np.pi) for item in self.samples])
        rot_range = np.linspace(np.min(rot_list), np.max(rot_list), num=10)
        indexes = np.digitize(rot_list, rot_range, right=False)
        num_samples_of_bins = dict(Counter(indexes))
        emp_label_dist = [num_samples_of_bins.get(i, 0) for i in range(1, len(rot_range)+1)]

        # Apply 1d convolution to get the smoothed effective label distribution
        lds_kernel_window = get_lds_kernel_window(kernel='gaussian', ks=7, sigma=5)
        eff_label_dist = convolve1d(np.array(emp_label_dist), weights=lds_kernel_window, mode='constant')

        self.weights = [np.float32(1/eff_label_dist[bin_idx-1]) for bin_idx in indexes]

    def __getitem__(self, index):
        sample = self.samples[index]
        imgs = [np.asarray(Image.open(img)) for img in sample['imgs']]
        
        if self.transform is not None:
            imgs, imus, gts = self.transform(imgs, np.copy(sample['imus']), np.copy(sample['gts']))
        else:
            imus = np.copy(sample['imus'])
            gts = np.copy(sample['gts']).astype(np.float32)
        
        rot = sample['rot'].astype(np.float32)
        weight = self.weights[index]

        return imgs, imus, gts, rot, weight

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Training sequences: '
        for seq in self.train_seqs:
            fmt_str += '{} '.format(seq)
        fmt_str += '\n'
        fmt_str += '    Number of segments: {}\n'.format(self.__len__())
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))

        return fmt_str


def get_lds_kernel_window(kernel, ks, sigma):
    assert kernel in ['gaussian', 'triang', 'laplace']
    half_ks = (ks - 1) // 2
    if kernel == 'gaussian':
        base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(gaussian_filter1d(base_kernel, sigma=sigma))
    elif kernel == 'triang':
        kernel_window = triang(ks)
    else:
        laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
        kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(map(laplace, np.arange(-half_ks, half_ks + 1)))

    return kernel_window

# Example usage
if __name__ == "__main__":
    # Define transforms (same as VS-VIO)
    transform = custom_transform.Compose([
        custom_transform.ToTensor(),
        custom_transform.Resize(size=(256, 512)),
        custom_transform.RandomHorizontalFlip(p=0.5),
        custom_transform.RandomColorAug(p=0.5),
    ])
    
    # Create dataset
    dataset = CustomDataset(
        root='/workspace/Junyu/Visual-Selective-VIO/custom_data',
        sequence_length=11,
        train_seqs=['seq_01'],
        transform=transform
    )
    
    print(dataset)
    print(f"Total segments: {len(dataset)}")
    
    # Test loading a sample
    imgs, imus, gts, rot, weight = dataset[0]
    print(f"Images shape: {imgs.shape}")
    print(f"IMUs shape: {imus.shape}")
    print(f"Ground truth poses shape: {gts.shape}")
    print(f"Rotation: {rot}")
    print(f"Weight: {weight}")