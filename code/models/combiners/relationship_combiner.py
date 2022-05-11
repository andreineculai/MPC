import torch
import torch.nn as nn


class RelationshipCombiner(nn.Module):

    def __init__(self, config):
        super(RelationshipCombiner, self).__init__()
        self.config = config
        self.input_dim = 512 + 512
        self.output_dim = 1024
        self.num_classes = 512

        self.relationship = nn.Sequential(
            nn.Conv2d(self.input_dim, self.output_dim, kernel_size=1),
            nn.BatchNorm2d(self.output_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Sequential(
            nn.Linear(self.output_dim, self.num_classes),
            nn.Sigmoid(),
        )


    def forward(self, features):

        rez = features[0]
        for i in range(1, len(features)):
            rez = self.combine(rez, features[i])
        return rez


    def combine(self, f1, f2):

        x = torch.cat((f2, f1), 1)
        x = x.unsqueeze(2).unsqueeze(2)
        x = self.relationship(x)
        x = x.squeeze()
        out = self.fc(x)

        return out