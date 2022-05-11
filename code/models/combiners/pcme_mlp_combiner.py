import torch
import torch.nn as nn


class PcmeMlpCombiner(nn.Module):

    def __init__(self, config):
        super(PcmeMlpCombiner, self).__init__()
        self.config = config
        self.embed_dim = config.model.embed_size
        self.combiner = torch.nn.Sequential(
            torch.nn.Linear(4*self.embed_dim, 2*self.embed_dim),
            torch.nn.BatchNorm1d(2 * self.embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2*self.embed_dim, 2*self.embed_dim)
        ).cuda()

    def forward(self, embeddings):
        prior_mean = embeddings[0]['embedding']
        prior_variance = embeddings[0]['logsigma']
        log_z_total = 0

        for i  in range(1, len(embeddings)):

            posterior_mean, posterior_variance, log_z = self.combine_2_gaussian_random_variables(prior_mean, prior_variance, embeddings[i]['embedding'], embeddings[i]['logsigma'])
            log_z_total += log_z
            prior_mean = posterior_mean
            prior_variance = posterior_variance
        return posterior_mean, posterior_variance, log_z_total

    def combine_2_gaussian_random_variables(self, mean1, variance1, mean2, variance2):


        rez = self.combiner(torch.cat([mean1, variance1, mean2, variance2], dim=1))

        return rez[:, 0:self.embed_dim], rez[:, self.embed_dim:2*self.embed_dim], torch.zeros(mean1.shape[0], device='cuda')
