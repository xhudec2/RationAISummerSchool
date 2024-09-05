from torch import nn
from torchvision.models import resnet18


def resnet() -> nn.Module:
    backbone = resnet18(pretrained=True)
    backbone = nn.Sequential(*list(backbone.children())[:-1])
    return backbone
