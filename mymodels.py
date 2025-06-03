import torch
import torch.nn as nn
from torchvision import models

NUM_CLASSES = 12

class EfficientNetV2(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, hidden_units=256, dropout_rate=0.5):
        super(EfficientNetV2, self).__init__()
        self.model = models.efficientnet_v2_s(weights=None)
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Linear(num_ftrs, int(hidden_units)),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(int(hidden_units), num_classes)
        )

    def forward(self, x):
        return self.model(x)


class ConvNeXt(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, hidden_units=256, dropout_rate=0.5):
        super(ConvNeXt, self).__init__()
        self.model = models.convnext_large(weights=None)
        num_ftrs = self.model.classifier[-1].in_features
        self.model.classifier = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.LayerNorm((num_ftrs,), eps=1e-06, elementwise_affine=True),
            nn.Linear(num_ftrs, int(hidden_units)),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(int(hidden_units), num_classes)
        )

    def forward(self, x):
        return self.model(x)


class ResNet50V2(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, hidden_units=256, dropout_rate=0.5):
        super(ResNet50V2, self).__init__()
        self.model = models.resnet50(weights=None)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, int(hidden_units)),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(int(hidden_units), num_classes)
        )

    def forward(self, x):
        return self.model(x)