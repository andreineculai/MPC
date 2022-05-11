import torch
import torch.nn as nn

from criterions.pcme_retrieval_loss import PcmeRetrievalLoss
from criterions.utils import mean_distance
from third_party.pcme.utils.tensor_utils import sample_gaussian_tensors

class PcmeCriterion(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.distance_function = mean_distance
        self.config = config
        self.retrieval_loss = PcmeRetrievalLoss(config.criterion.train, self.distance_function)
        self.retrieval_loss_weight = self.config.criterion.train.retrieval_loss_weight
        self.logsigma_l2_loss_weight = self.config.criterion.train.logsigma_l2_loss_weight

        self.n_samples = self.config.model.n_samples_inference

    def forward(self, embeddings):

        logsigma_l2_loss = torch.zeros([], device='cuda')
        for emb in (embeddings['source'] + [embeddings['query']]):
            logsigma_l2_loss += torch.mean(torch.square(emb['logsigma']))
        logsigma_l2_loss /= (len(embeddings['source']) + 1)

        target_samples = sample_gaussian_tensors(embeddings['target']['mean'], embeddings['target']['logsigma'], self.n_samples).cuda()
        query_samples = sample_gaussian_tensors(embeddings['query']['mean'], embeddings['query']['logsigma'], self.n_samples).cuda()

        query_to_target_loss_value, recall = self.retrieval_loss(query_samples, target_samples, recall = True)
        target_to_query_loss_value = self.retrieval_loss(target_samples, query_samples)
        retrieval_loss_value = (query_to_target_loss_value + target_to_query_loss_value) / 2

        loss = self.retrieval_loss_weight * retrieval_loss_value + \
               self.logsigma_l2_loss_weight * logsigma_l2_loss

        loss_dict = {
            'query_to_target_loss': query_to_target_loss_value.item(),
            'target_to_query_loss': target_to_query_loss_value.item(),
            'retrieval_loss': retrieval_loss_value.item(),
            'logsigma_l2_loss': logsigma_l2_loss.item(),
            'loss': loss.item(),
            'r@1_within_batch': recall,
        }

        return loss, loss_dict
