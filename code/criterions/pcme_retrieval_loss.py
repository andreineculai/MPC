import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from third_party.pcme.utils.tensor_utils import l2_normalize

MIN_SCORE = -1

class PcmeRetrievalLoss(nn.Module):

    def __init__(self, config, distance_function):
        super(PcmeRetrievalLoss, self).__init__()
        self.distance_function = distance_function
        self.temperature = config.temperature


    def forward(self, query_features, target_features, reduction = 'mean', recall=False):

        query_features = l2_normalize(query_features)
        target_features = l2_normalize(target_features)
        batch_size_query = query_features.shape[0]

        scores = (query_features.unsqueeze(1) @ target_features.unsqueeze(0).transpose(-2, -1))
        scores = self.distance_function(scores)

        labels = torch.arange(0, target_features.shape[0]).long().to(scores.get_device())
        loss = F.cross_entropy(scores / self.temperature, labels, reduction=reduction)
        if recall:
            max_scores = torch.max(scores, dim=0).indices - torch.arange(0, batch_size_query, device='cuda')
            recall = torch.count_nonzero(max_scores == 0)
            return loss, recall
        return loss

