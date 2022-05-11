import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from third_party.pcme.utils.tensor_utils import l2_normalize, sample_gaussian_tensors


class MpcRetrievalLoss(nn.Module):

    def __init__(self, config):
        super(MpcRetrievalLoss, self).__init__()
        self.temperature = config.temperature

    def forward(self, query_mean, query_logsigma, query_z, target_mean, target_logsigma, target_z, reduction = 'mean', recall=False):
        query_mean = l2_normalize(query_mean)
        target_mean = l2_normalize(target_mean)
        target_samples = sample_gaussian_tensors(target_mean, target_logsigma, 7).cuda()


        inv_sigmas = torch.exp(-query_logsigma)
        loc = -0.5 * torch.mean(torch.sum(((target_samples.unsqueeze(0) - query_mean.unsqueeze(1).unsqueeze(2)) ** 2) * inv_sigmas.unsqueeze(1).unsqueeze(2), dim=-1), dim=-1)
        norm_constant = (-query_mean.shape[-1]/2) * torch.log(torch.Tensor([2*math.pi]).cuda()) - 0.5 * (torch.sum(query_logsigma, dim=-1))
        scores = query_z + norm_constant + loc
        scores = scores - torch.max(scores, dim=0, keepdim=True).values

        labels = torch.arange(0, query_mean.shape[0]).long().to(scores.get_device())
        loss = F.cross_entropy(scores/self.temperature, labels, reduction=reduction)
        if recall:
            max_scores = torch.max(scores, dim=0).indices - torch.arange(0, query_mean.shape[0], device='cuda')
            recall = torch.count_nonzero(max_scores == 0)
            return loss, recall
        return loss

