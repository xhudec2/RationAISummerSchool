from torch import nn
from torchvision.models import vgg16


def vgg() -> nn.Module:
    backbone = vgg16(pretrained=True)
    backbone = nn.Sequential(*list(backbone.children())[:-1])
    return backbone
