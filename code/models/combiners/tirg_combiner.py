import torch
import torch.nn as nn
import torch.nn.functional as F


class TirgCombiner(nn.Module):

    def __init__(self, config):
        super(TirgCombiner, self).__init__()
        self.config = config
        embed_dim = config.model.embed_size
        self.a = torch.nn.Parameter(torch.tensor([1.0, 10.0, 1.0, 1.0]))
        self.gated_feature_composer = torch.nn.Sequential(
            ConCatModule(), torch.nn.BatchNorm1d(2 * embed_dim), torch.nn.ReLU(),
            torch.nn.Linear(2 * embed_dim, embed_dim))
        self.res_info_composer = torch.nn.Sequential(
            ConCatModule(), torch.nn.BatchNorm1d(2 * embed_dim), torch.nn.ReLU(),
            torch.nn.Linear(2 * embed_dim, 2 * embed_dim), torch.nn.ReLU(),
            torch.nn.Linear(2 * embed_dim, embed_dim))

    def forward(self, features):

        rez = features[0]
        for i in range(1, len(features)):
            rez = self.combine(rez, features[i])
        return rez

    def combine(self, features1, features2):

        f1 = self.gated_feature_composer((features1, features2))
        f2 = self.res_info_composer((features1, features2))
        f = F.sigmoid(f1) * features1 * self.a[0] + f2 * self.a[1]
        return f

class ConCatModule(torch.nn.Module):
    def __init__(self):
        super(ConCatModule, self).__init__()

    def forward(self, x):
        x = torch.cat(x, dim=1)
        return x
