import torch
import torch.nn as nn
from torchvision.models import resnet50, efficientnet_b0, efficientnet_b1


def models(model_name: str):
    if model_name == 'resnet50':
        model = resnet50(weights='DEFAULT')
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 5)
    elif model_name == 'efficientnet_b0':
        pretrained = efficientnet_b0(weights='DEFAULT')
        model = EfficientNet(pretrained=pretrained)
    elif model_name == 'efficientnet_b1':
        pretrained = efficientnet_b1(weights='DEFAULT')
        model = EfficientNet(pretrained=pretrained)
    elif model_name == 'vcmcnn':
        model = VehicleColorModel()
    return model


class EfficientNet(torch.nn.Module):
    def __init__(self, pretrained):
        super(EfficientNet, self).__init__()
        self.pretrained = pretrained
        self.classifier_layer = torch.nn.Sequential(
            torch.nn.Linear(1280, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.Dropout(0.7),
            torch.nn.Linear(512, 256),
            torch.nn.Dropout(0.6),
            torch.nn.Linear(256, 5))

    def forward(self, x):
        x = self.pretrained.features(x)
        x = self.pretrained.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.pretrained.classifier[0](x)
        x = self.classifier_layer(x)
        return x


class VehicleColorModel(nn.Module):
    def __init__(self):
        super(VehicleColorModel, self).__init__()
        self.top_conv1 = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=(11, 11), stride=(4, 4)),
            nn.ReLU(),
            nn.BatchNorm2d(48),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        # first top convolution layer    after split
        self.top_top_conv2 = nn.Sequential(
            # 1-1 conv layer
            nn.Conv2d(24, 64, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.top_bot_conv2 = nn.Sequential(
            # 1-1 conv layer
            nn.Conv2d(24, 64, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        #  need a concat
        # after concat
        self.top_conv3 = nn.Sequential(
            # 1-1 conv layer
            nn.Conv2d(128, 192, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU()
        )

        # fourth top convolution layer
        # split feature map by half
        self.top_top_conv4 = nn.Sequential(
            # 1-1 conv layer
            nn.Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU()
        )

        self.top_bot_conv4 = nn.Sequential(
            # 1-1 conv layer
            nn.Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU()
        )

        # fifth top convolution layer
        self.top_top_conv5 = nn.Sequential(
            # 1-1 conv layer
            nn.Conv2d(96, 64, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.top_bot_conv5 = nn.Sequential(
            # 1-1 conv layer
            nn.Conv2d(96, 64, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

#        # ===============================  bottom ================================

#         # first bottom convolution layer
        self.bottom_conv1 = nn.Sequential(
            # 1-1 conv layer
            nn.Conv2d(3, 48, kernel_size=(11, 11), stride=(4, 4)),
            nn.ReLU(),
            nn.BatchNorm2d(48),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        # first top convolution layer    after split
        self.bottom_top_conv2 = nn.Sequential(
            # 1-1 conv layer
            nn.Conv2d(24, 64, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.bottom_bot_conv2 = nn.Sequential(
            # 1-1 conv layer
            nn.Conv2d(24, 64, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        #  need a concat

        # after concat
        self.bottom_conv3 = nn.Sequential(
            # 1-1 conv layer
            nn.Conv2d(128, 192, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU()
        )

        # fourth top convolution layer
        # split feature map by half
        self.bottom_top_conv4 = nn.Sequential(
            # 1-1 conv layer
            nn.Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU()
        )

        self.bottom_bot_conv4 = nn.Sequential(
            # 1-1 conv layer
            nn.Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU()
        )

        # fifth top convolution layer
        self.bottom_top_conv5 = nn.Sequential(
            # 1-1 conv layer
            nn.Conv2d(96, 64, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.bottom_bot_conv5 = nn.Sequential(
            # 1-1 conv layer
            nn.Conv2d(96, 64, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        # Fully-connected layer
        self.classifier = nn.Sequential(
            nn.Linear(5*5*64*4, 4096),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(4096, 5)
        )

    def forward(self, x):
        x_top = self.top_conv1(x)
        x_top_conv = torch.split(x_top, 24, 1)
        x_top_top_conv2 = self.top_top_conv2(x_top_conv[0])
        x_top_bot_conv2 = self.top_bot_conv2(x_top_conv[1])
        x_top_cat1 = torch.cat([x_top_top_conv2, x_top_bot_conv2], 1)
        x_top_conv3 = self.top_conv3(x_top_cat1)
        x_top_conv3 = torch.split(x_top_conv3, 96, 1)
        x_top_top_conv4 = self.top_top_conv4(x_top_conv3[0])
        x_top_bot_conv4 = self.top_bot_conv4(x_top_conv3[1])
        x_top_top_conv5 = self.top_top_conv5(x_top_top_conv4)
        x_top_bot_conv5 = self.top_bot_conv5(x_top_bot_conv4)
        x_bottom = self.bottom_conv1(x)
        x_bottom_conv = torch.split(x_bottom, 24, 1)
        x_bottom_top_conv2 = self.bottom_top_conv2(x_bottom_conv[0])
        x_bottom_bot_conv2 = self.bottom_bot_conv2(x_bottom_conv[1])
        x_bottom_cat1 = torch.cat([x_bottom_top_conv2, x_bottom_bot_conv2], 1)
        x_bottom_conv3 = self.bottom_conv3(x_bottom_cat1)
        x_bottom_conv3 = torch.split(x_bottom_conv3, 96, 1)
        x_bottom_top_conv4 = self.bottom_top_conv4(x_bottom_conv3[0])
        x_bottom_bot_conv4 = self.bottom_bot_conv4(x_bottom_conv3[1])
        x_bottom_top_conv5 = self.bottom_top_conv5(x_bottom_top_conv4)
        x_bottom_bot_conv5 = self.bottom_bot_conv5(x_bottom_bot_conv4)
        x_cat = torch.cat([x_top_top_conv5, x_top_bot_conv5,
                           x_bottom_top_conv5, x_bottom_bot_conv5], 1)
        flatten = x_cat.view(x_cat.size(0), -1)
        output = self.classifier(flatten)
        return output
