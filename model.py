import torch.nn as nn
from torchvision.models import vgg16_bn as vgg16_bn
from torchvision.models import resnet101 as resnet101


class VGG(nn.Module):

    def __init__(self, freeze=True, pretrained=True, k=100, d=300):
        super(VGG, self).__init__()
        model = vgg16_bn(pretrained=pretrained)
        self.features = nn.Sequential(*list(model.features.children()))
        if freeze:
            for param in self.features.parameters():
                param.requires_grad = False
        self.classifier = nn.Sequential(*list(model.classifier.children())[:4])
        self.classifier._modules['3'] = nn.Linear(4096, k * d)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class RESNET(nn.Module):

    def __init__(self, freeze=True, pretrained=True, k=100, d=300):
        super(RESNET, self).__init__()
        model = resnet101(pretrained=pretrained)
        self.features = nn.Sequential(*list(model.children())[:-1])
        if freeze:
            for param in self.features.parameters():
                param.requires_grad = False
        self.classifier = nn.Sequential(nn.Linear(2048, k * d))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
