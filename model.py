import torch
import torch.nn as nn
from torchvision import models


class SkinCancerDenseNet(nn.Module):
    """
    DenseNet-121 for binary skin cancer classification.
    Output classes:
      0 -> non-cancer
      1 -> cancer
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()

        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        self.model = models.densenet121(weights=weights)

        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)


def build_model(device: str = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SkinCancerDenseNet(num_classes=2, pretrained=True)
    model = model.to(device)
    return model, device