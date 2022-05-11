import torch.nn as nn
import torch.nn.functional as F

class FilmCombiner(nn.Module):
    # FiLM: Visual Reasoning with a General Conditioning Layer
    def __init__(self, config):
        super().__init__()
        input_dim, output_dim, num_classes = 512, 1024, 512

        self.fc11 = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
        )
        self.fc12 = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.BatchNorm1d(output_dim)
        )
        self.fc2_gamma = nn.Sequential(
            nn.Linear(input_dim, output_dim),
        )
        self.fc2_beta = nn.Sequential(
            nn.Linear(input_dim, output_dim),
        )
        self.fc = nn.Sequential(
            nn.Linear(output_dim, num_classes),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU()

        self.normalization = False


    def forward(self, features):

        rez = features[0]
        for i in range(1, len(features)):
            rez = self.combine(rez, features[i])
        return rez

    def combine(self, f1, f2):
        if self.normalization:
            dim = len(f1.shape) - 1
            f1 = F.normalize(f1, dim=dim)
            f2 = F.normalize(f2, dim=dim)

        gamma = self.fc2_gamma(f2)
        beta = self.fc2_beta(f2)

        x = self.fc11(f1)
        x_in = x
        x = self.fc12(x)
        x = gamma * x + beta
        x = self.relu(x) + x_in

        out = self.fc(x)

        return out