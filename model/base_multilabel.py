import torch
import torch.nn as nn
import torchvision.models as models
from timm import create_model  # For EfficientNet and ViT


class PretrainedModelWrapper(nn.Module):
    def __init__(self, architecture="resnet50", num_classes=10, pretrained=True):
        """
        A wrapper for loading different pre-trained models and modifying their final layer 
        for multi-label classification (sigmoid activation).
        
        Args:
        - architecture (str): The name of the model to load (resnet50, densenet121, efficientnet_b0, vit_base_patch16_224, etc.)
        - num_classes (int): Number of output classes
        - pretrained (bool): Whether to load pre-trained weights
        """
        super(PretrainedModelWrapper, self).__init__()
        
        self.architecture = architecture
        self.num_classes = num_classes

        # Load the model based on the architecture type
        if "resnet" in architecture:
            self.model = getattr(models, architecture)(pretrained=pretrained)
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_classes)

        elif "densenet" in architecture:
            self.model = getattr(models, architecture)(pretrained=pretrained)
            in_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(in_features, num_classes)

        elif "efficientnet" in architecture:
            self.model = create_model(architecture, pretrained=pretrained)
            in_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(in_features, num_classes)

        elif "vit" in architecture:
            self.model = create_model(architecture, pretrained=pretrained)
            in_features = self.model.head.in_features
            self.model.head = nn.Linear(in_features, num_classes)

        else:
            raise ValueError(f"Unsupported model architecture: {architecture}")

        # Sigmoid activation for multi-label classification
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        return self.activation(x)  # Apply sigmoid for multi-label output
