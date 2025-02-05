from PIL import Image
from torch.utils.data import Dataset
Image.MAX_IMAGE_PIXELS = None

from utils.augment import SimSiamAugment


class SimSiamDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.augment = SimSiamAugment()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        image = Image.open(img_path).convert("RGB")
        x1, x2 = self.augment(image)  # Apply strong augmentations
        return x1, x2
