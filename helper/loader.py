import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.transforms import Lambda
from tqdm import tqdm


PIL.Image.MAX_IMAGE_PIXELS = 933120000
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((512, 512)),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

label_encoding = {
    "Sachet/Pouch": 0,
    "Carton/Box": 1, 
    "Label": 2,
    "Blister": 3
}

target_transform = Lambda(lambda y: torch.zeros(
    4, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))

class PackTypeDataset(Dataset):
    def __init__(self, img_dir: str, df_loc: str, train: bool, label_encoding: dict, 
                 transform=None, target_transform=None):
        self.img_dir = img_dir
        self.df = pd.read_csv(df_loc)
        self.label_encoding = label_encoding

        self.df = self.df[self.df["Pack Type"].isin(["Sachet/Pouch", "Carton/Box", "Label", "Blister"])]
        self.df = self.df[[True if os.path.exists(os.path.join(img_dir, x)) else False for x in self.df["File"].values.tolist()]]



        if train:
            self.images = self.df["File"].values.tolist()[:-50]
            self.labels = self.df["Pack Type"].values.tolist()[:-50]
        else:
            self.images = self.df["File"].values.tolist()[-50:]
            self.labels = self.df["Pack Type"].values.tolist()[-50:]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        image = Image.open(img_path).convert("RGB")
        label = self.label_encoding[self.labels[idx]]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label, img_path
