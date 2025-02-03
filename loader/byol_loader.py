from PIL import Image
from torch.utils.data import Dataset

from utils.augment import BYOLAugment

class BYOLDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.transform = BYOLAugment

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")

        # Generate two augmented views
        x1, x2 = self.augment(image)
        return x1, x2
