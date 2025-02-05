# import os
from PIL import Image
from torch.utils.data import Dataset
Image.MAX_IMAGE_PIXELS = None

from utils.augment import SimCLRAugment


class ContrastiveDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.augment = SimCLRAugment()
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load the image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Apply the transformations to the image (for contrastive learning, we use different augmentations)
        img1, img2 = self.augment(image)
        return img1, img2