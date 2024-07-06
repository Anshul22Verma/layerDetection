import cv2
import numpy as np
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

from helper.loader import PackTypeDataset
from helper.train_utils import test, train, train_epochs, validate


class ShallowLKCNN(nn.Module):
    def __init__(self, n_classes: int = 4):
        super(ShallowLKCNN, self).__init__()
        
        # Define the first convolutional layer with a large kernel size
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=17, stride=3, padding=5)
        
        # Define the second convolutional layer with a large kernel size
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=9, stride=3, padding=3)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=7, stride=1, padding=3)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=3)
        # Define a fully connected layer
        self.fc1 = nn.Linear(32 * 4 * 4, 128)  # Adjust the input size according to the input image size and convolution layers
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)
        # print(x.shape)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x


if __name__ == "__main__":
    batch_size = 12
    num_epochs = 50


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
    net = ShallowLKCNN(n_classes=4)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    dev = torch.device('cuda:0') if torch.cuda.is_available() else torch.cpu()
    net = net.to(dev)

    os.mkdirs("/home/azureuser/runs", exists_ok=True)
    os.mkdirs("/home/azureuser/runs/shallowLKCNN", exists_ok=True)
    writer = SummaryWriter(log_dir="/home/azureuser/runs/shallowLKCNN")


    net, writer = train_epochs(model=net, writer=writer, trainloader=trainloader, testloader=testloader, criterion=criterion,
                               optimizer=optimizer, device=dev, num_epochs=num_epochs)

    validate(model=net, trainloader=trainloader, testloader=testloader, device=dev, path="/home/azureuser/runs/base")

    PATH = '/home/azureuser/runs/shallowLKCNN/shallowLK_model.pth'
    torch.save(net.state_dict(), PATH)
