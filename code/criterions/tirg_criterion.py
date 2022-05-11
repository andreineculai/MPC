import torch
import torch.nn as nn

from trainers import torch_functions
import torch.nn.functional as F

class TirgCriterion(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.normalization_layer = torch_functions.NormalizationLayer(
            normalize_scale=4.0, learn_scale=True)

    def forward(self, f1, f2):
        return self.compute_loss(f1, f2)

    def compute_loss(self,
                     query,
                     target):

        mod_img1 = self.normalization_layer(query)
        img2 = self.normalization_layer(target)
        return self.compute_batch_based_classification_loss_(mod_img1, img2)

    def compute_batch_based_classification_loss_(self, mod_img1, img2):
        x = torch.mm(mod_img1, img2.transpose(0, 1))
        labels = torch.tensor(range(x.shape[0])).long()
        labels = torch.autograd.Variable(labels).cuda()
        return F.cross_entropy(x, labels)


