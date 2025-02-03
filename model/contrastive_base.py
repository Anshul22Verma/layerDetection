import torch
import torch.nn as nn
import torchvision.models as models
from timm import create_model  # For EfficientNet and ViT


class ContrastiveModelWrapper(nn.Module):
    def __init__(self, architecture="resnet50", embedding_dim=128, pretrained=True):
        """
        Wrapper for contrastive learning with a learnable projection head (FCN).
        
        Args:
        - architecture (str): Backbone model architecture (resnet50, densenet121, efficientnet_b0, vit_base_patch16_224, etc.)
        - embedding_dim (int): Size of the base encoder output features
        - projection_dim (int): Size of the projection head output
        - pretrained (bool): Whether to use ImageNet pre-trained weights
        """
        super(ContrastiveModelWrapper, self).__init__()
        self.architecture = architecture
        # Load the model
        if "resnet" in architecture:
            self.encoder = getattr(models, architecture)(pretrained=pretrained)
            in_features = self.encoder.fc.in_features
            self.encoder.fc = nn.Identity()  # Remove the default FC layer

        elif "densenet" in architecture:
            self.encoder = getattr(models, architecture)(pretrained=pretrained)
            in_features = self.encoder.classifier.in_features
            self.encoder.classifier = nn.Identity()

        elif "efficientnet" in architecture:
            self.encoder = create_model(architecture, pretrained=pretrained)
            in_features = self.encoder.classifier.in_features
            self.encoder.classifier = nn.Identity()

        elif "vit" in architecture:
            self.encoder = create_model(architecture, pretrained=pretrained)
            in_features = self.encoder.head.in_features
            self.encoder.head = nn.Identity()

        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

        # Add the projection head (FCN)
        self.projection_head = nn.Linear(in_features, embedding_dim)


    def forward(self, x):
        features = self.encoder(x)  # Extract base features
        projections = self.projection_head(features)  # Apply FCN
        return projections

    def remove_projection_head(self):
        """
        Removes the projection head for fine-tuning the encoder on downstream tasks.
        """
        self.projection_head = nn.Identity()
    
    @property
    def name(self):
        """Property to return the architecture name."""
        return self.architecture


class FineTuneContrastiveBaseModel(nn.Module):
    def __init__(self, base_model, num_classes):
        """
        Fine-tune the contrastive model for supervised classification.

        Args:
        - base_model (ContrastiveModelWrapper): The pre-trained contrastive model.
        - num_classes (int): Number of target classes.
        """
        super(FineTuneContrastiveBaseModel, self).__init__()
        self.base_model = base_model  # Keep the pre-trained encoder
        self.base_model.remove_projection_head()  # remove the MLP projection head

        in_features = base_model.projection_head[0].in_features

        # Add a classification head
        self.classifier = nn.Sequential(
            nn.Linear(in_features, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.encoder(x)
        outputs = self.classifier(features)
        return outputs
