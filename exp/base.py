import os
import pandas as pd
import PIL
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision.models.resnet import resnet50, ResNet50_Weights
import torchvision.transforms as transforms
from torchvision.transforms import Lambda
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from helper.loader import PackTypeDataset, label_encoding
from helper.train_utils import test, train, train_epochs, validate


class ResNet50Cls(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.base = resnet50(weights=ResNet50_Weights.DEFAULT)
        num_features = self.base.fc.in_features
        self.base.fc = nn.Linear(num_features, n_classes)

    def forward(self, x):
        x = self.base(x)
        return x


if __name__ == "__main__":
    batch_size = 12
    num_epochs = 20


    trainset = PackTypeDataset(img_dir="/home/azureuser/images/cad-dielines", df_loc="/home/azureuser/master.csv", 
                               train=True, label_encoding=label_encoding,
                               transform=transform, target_transform=target_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = PackTypeDataset(img_dir="/home/azureuser/images/cad-dielines", df_loc="/home/azureuser/master.csv", 
                              train=False, label_encoding=label_encoding,
                              transform=transform, target_transform=target_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    classes = ("Sachet/Pouch", "Carton/Box", "Label", "Blister")
    net = ResNet50Cls(n_classes=4)
    

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    dev = torch.device('cuda:0') if torch.cuda.is_available() else torch.cpu()
    net = net.to(dev)

    os.mkdirs("/home/azureuser/runs", exists_ok=True)
    os.mkdirs("/home/azureuser/runs/shallowLKCNN", exists_ok=True)
    writer = SummaryWriter(log_dir="/home/azureuser/runs/base")


    net, writer = train_epochs(model=net, writer=writer, trainloader=trainloader, testloader=testloader, criterion=criterion, 
                               optimizer=optimizer, device=dev, num_epochs=num_epochs)
    validate(model=net, trainloader=trainloader, testloader=testloader, device=dev, path="/home/azureuser/runs/base")

    PATH = '/home/azureuser/runs/base/base_model.pth'
    torch.save(net.state_dict(), PATH)
