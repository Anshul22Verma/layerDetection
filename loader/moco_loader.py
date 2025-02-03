from PIL import Image
from torch.utils.data import Dataset

from utils.augment import MoCoAugment


class MoCoDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths
        # has a strong and weak augmentation within the augmentor
        self.augment = MoCoAugment()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")

        # Generate two different augmented views
        x_q, x_k = self.augment(image)
        return x_q, x_k
