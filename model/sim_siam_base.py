import torch
import torch.nn as nn
import torch.nn.functional as F
import timm  # For loading EfficientNet and ViT models
import torchvision.models as models


class SimSiamModelWrapper(nn.Module):
    def __init__(self, architecture="resnet50", projection_dim=128, hidden_dim=512, queue_size=65536):
        """
        Universal Contrastive Learning Wrapper supporting MoCo, SimSiam, and BYOL.

        Args:
        - architecture (str): Backbone model ('resnet50', 'densenet121', 'efficientnet_b0', 'vit_base_patch16_224')
        - projection_dim (int): Output size of the projection head
        - hidden_dim (int): Hidden layer size in projection/prediction heads
        """
        super(SimSiamModelWrapper, self).__init__()
        self.queue_size = queue_size

        # Load backbone model
        self.encoder_q = self._load_backbone(architecture)

        # Projection head (MLP)
        self.projector = self._get_mlp(self.encoder_q.out_features, projection_dim, hidden_dim)
        
        # Predictor head for SimSiam
        self.predictor = self._get_mlp(projection_dim, projection_dim, hidden_dim)

    def _load_backbone(self, architecture):
        """Loads the specified backbone and removes classification layers."""
        if architecture.startswith("resnet"):
            model = getattr(models, architecture)(pretrained=True)
            out_features = model.fc.in_features
            model.fc = nn.Identity()
        elif architecture.startswith("densenet"):
            model = getattr(models, architecture)(pretrained=True)
            out_features = model.classifier.in_features
            model.classifier = nn.Identity()
        elif architecture.startswith("efficientnet"):
            model = getattr(models, architecture)(pretrained=True)
            out_features = model.classifier[1].in_features
            model.classifier = nn.Identity()
        elif architecture.startswith("vit"):
            model = timm.create_model(architecture, pretrained=True, num_classes=0)
            out_features = model.embed_dim
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

        model.out_features = out_features
        return model

    def _get_mlp(self, input_dim, output_dim, hidden_dim):
        """Creates an MLP with one hidden layer."""
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x1, x2=None):
        """Forward pass based on the contrastive method used."""
        z1 = self.projector(self.encoder_q(x1))
        p1, p2 = self.predictor(z1), self.predictor(self.projector(self.encoder_q(x2)))
        return p1, z1.detach(), p2, z1.detach()

    def remove_contrastive_heads(self):
        """
        Converts the contrastive learning model to a base encoder that only outputs feature projections.
        - Removes the projector & predictor heads.
        - Returns a model that only includes the backbone encoder.
        """
        self.projector = None  # Remove projector
        self.predictor = None  # Remove predictor
        return self.encoder_q  # Return base encoder
