from PIL import Image
from torch.utils.data import Dataset
from typing import List


class ClassificationDataset(Dataset):
    def __init__(self, image_paths, labels, uq_classes: List, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.uq_classes = uq_classes
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load the image
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        
        # Get the label
        label = self.labels[idx]
        label = [1 if l in label else 0 for l in self.uq_labels]
        
        # Apply the transformations (if any)
        if self.transform:
            img = self.transform(img)
        
        return img, label