import torch
import torch.nn as nn

from criterions.mpc_retrieval_loss import MpcRetrievalLoss
from third_party.pcme.utils.tensor_utils import l2_normalize

class MpcCriterion(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.logsigma_l2_loss_weight = self.config.criterion.train.logsigma_l2_loss_weight

        self.retrieval_loss = MpcRetrievalLoss(config.criterion.train)
        self.retrieval_loss_weight = self.config.criterion.train.retrieval_loss_weight

    def forward(self, embeddings):

        logsigma_l2_loss = torch.zeros([], device='cuda')
        for emb in (embeddings['source'] + [embeddings['query']]):
            logsigma_l2_loss += torch.mean(torch.square(emb['logsigma']))
        logsigma_l2_loss /= (len(embeddings['source']) + 1)

        query_to_target_loss_value, recall = self.retrieval_loss(embeddings['query']['mean'],
                                                                 embeddings['query']['logsigma'],
                                                                 embeddings['query_z'],
                                                                 embeddings['target']['mean'],
                                                                 embeddings['target']['logsigma'],
                                                                 embeddings['target_z'],
                                                                 recall=True)
        target_to_query_loss_value = self.retrieval_loss(embeddings['target']['mean'],
                                                         embeddings['target']['logsigma'],
                                                         embeddings['target_z'],
                                                         embeddings['query']['mean'],
                                                         embeddings['query']['logsigma'],
                                                         embeddings['query_z'])
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
