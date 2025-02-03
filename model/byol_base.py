import torch
import torch.nn as nn
import torch.nn.functional as F
import timm  # For loading different backbones
import torchvision.models as models


class ByolModelWrapper(nn.Module):
    def __init__(self, architecture="resnet50", projection_dim=128, hidden_dim=512, momentum=0.999, pretrained: bool=True):
        """
        Universal Contrastive Learning Wrapper supporting BYOL.

        Args:
        - architecture (str): Backbone model ('resnet50', 'densenet121', 'efficientnet_b0', 'vit_base_patch16_224')
        - projection_dim (int): Output size of the projection head
        - hidden_dim (int): Hidden layer size in projection/prediction heads
        - momentum (float): Momentum for updating key encoder BYOL
        - pretrained (bool): Key to start from pre-trained weights        
        """
        super(ByolModelWrapper, self).__init__()
        self.momentum = momentum

        # Load backbone
        self.encoder_q = self._load_backbone(architecture, pretrained=pretrained)
        self.encoder_k = self._load_backbone(architecture, pretrained=pretrained)
        
        # Projection head (MLP)
        self.projector = self._get_mlp(self.encoder_q.out_features, projection_dim, hidden_dim)
        
        # Predictor head for BYOL
        self.predictor = self._get_mlp(projection_dim, projection_dim, hidden_dim)
        
        # Stop gradient for momentum encoder
        if self.encoder_k:
            for param in self.encoder_k.parameters():
                param.requires_grad = False

    def _load_backbone(self, architecture, pretrained):
        """Loads the specified backbone and removes classification layers."""
        if architecture.startswith("resnet"):
            model = getattr(models, architecture)(pretrained=pretrained)
            out_features = model.fc.in_features
            model.fc = nn.Identity()
        elif architecture.startswith("densenet"):
            model = getattr(models, architecture)(pretrained=pretrained)
            out_features = model.classifier.in_features
            model.classifier = nn.Identity()
        elif architecture.startswith("efficientnet"):
            model = getattr(models, architecture)(pretrained=pretrained)
            out_features = model.classifier[1].in_features
            model.classifier = nn.Identity()
        elif architecture.startswith("vit"):
            model = timm.create_model(architecture, pretrained=pretrained, num_classes=0)
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

    @torch.no_grad()
    def _momentum_update(self):
        """Momentum update of key encoder for MoCo and BYOL."""
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)

    def forward(self, x1, x2=None):
        """Forward pass based on the contrastive method used."""
        z1 = self.projector(self.encoder_q(x1))
        z2 = self.projector(self.encoder_q(x2))

        with torch.no_grad():
            # Momentum update
            self._momentum_update()
            # Target encoder output (from the momentum encoder)
            target_z1 = self.projector(self.encoder_k(x1))
            target_z2 = self.projector(self.encoder_k(x2))
            
        # Return embeddings for loss computation
        return z1, z2, target_z1, target_z2
    
    def remove_contrastive_heads(self):
        """
        Converts the contrastive learning model to a base encoder that only outputs feature projections.
        - Removes the projector & predictor heads.
        - Returns a model that only includes the backbone encoder.
        """
        self.projector = None  # Remove projector
        self.predictor = None  # Remove predictor
        return self.encoder_q  # Return base encoder


class FineTuneBYOLBaseModel(nn.Module):
    def __init__(self, base_model, num_classes):
        """
        Fine-tune the contrastive model for supervised classification.

        Args:
        - base_model (ContrastiveModelWrapper): The pre-trained contrastive model.
        - num_classes (int): Number of target classes.
        """
        super(FineTuneBYOLBaseModel, self).__init__()
        self.base_model = base_model  # Keep the pre-trained encoder
        self.encoder = self.base_model.remove_contrastive_heads()  # remove the projector and returnes the encoder only

        in_features = self.encoder.out_features

        # Add a classification head
        self.classifier = nn.Sequential(
            nn.Linear(in_features, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.encoder(x)
        outputs = self.classifier(features)
        return outputs
