import torchvision

import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageModelTirg(nn.Module):

    def __init__(self, config):
        super(ImageModelTirg, self).__init__()

        # img model
        if config.model.cnn_type == "resnet50":
            img_model = torchvision.models.resnet50(pretrained=True)
        elif config.model.cnn_type == "resnet18":
            img_model = torchvision.models.resnet18(pretrained=True)

        class GlobalAvgPool2d(torch.nn.Module):

            def forward(self, x):
                return F.adaptive_avg_pool2d(x, (1, 1))

        img_model.avgpool = GlobalAvgPool2d()
        if config.model.cnn_type == "resnet50":
            img_model.fc = torch.nn.Sequential(torch.nn.Linear(2048, 512))
        elif config.model.cnn_type == "resnet18":
            img_model.fc = torch.nn.Sequential(torch.nn.Linear(512, 512))

        self.cnn = img_model


    def forward(self, data):

        return self.cnn(data)
