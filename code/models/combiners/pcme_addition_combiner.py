import torch
import torch.nn as nn


class PcmeAdditionCombiner(nn.Module):

    def __init__(self, config):
        super(PcmeAdditionCombiner, self).__init__()
        self.config = config

    def forward(self, embeddings):
        prior_mean = embeddings[0]['embedding']
        prior_variance = embeddings[0]['logsigma']
        log_z_total = 0

        for i  in range(1, len(embeddings)):

            posterior_mean, posterior_variance, log_z = add_2_gaussian_random_variables(prior_mean, prior_variance, embeddings[i]['embedding'], embeddings[i]['logsigma'])
            log_z_total += log_z
            prior_mean = posterior_mean
            prior_variance = posterior_variance
        return posterior_mean, posterior_variance, log_z_total

def add_2_gaussian_random_variables(mean1, variance1, mean2, variance2):

    variance1 = torch.exp(variance1)
    variance2 = torch.exp(variance2)

    return mean1 + mean2, torch.log(variance1 + variance2), torch.zeros(mean1.shape[0], device='cuda')
