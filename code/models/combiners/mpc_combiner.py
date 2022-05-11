import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal


class MpcCombiner(nn.Module):

    def __init__(self, config):
        super(MpcCombiner, self).__init__()
        self.config = config

    def forward(self, embeddings):
        prior_mean = embeddings[0]['embedding']
        prior_variance = embeddings[0]['logsigma']
        log_z_total = 0
        for i  in range(1, len(embeddings)):
            posterior_mean, posterior_variance, log_z = product_2_gaussians(prior_mean, prior_variance, embeddings[i]['embedding'], embeddings[i]['logsigma'])
            log_z_total += log_z
            prior_mean = posterior_mean
            prior_variance = posterior_variance

        return posterior_mean, posterior_variance, log_z_total

def product_2_gaussians(mean1, variance1, mean2, variance2):
    if len(mean1.shape) == 1:
        mean1 = mean1.unsqueeze(0)
    if len(variance1.shape) == 1:
        variance1 = variance1.unsqueeze(0)
    if len(mean2.shape) == 1:
        mean2 = mean2.unsqueeze(0)
    if len(variance2.shape) == 1:
        variance2 = variance2.unsqueeze(0)
    variance1 = torch.exp(variance1)
    variance2 = torch.exp(variance2)

    target_mean = mean2
    target_variance = variance2

    inv_variance1 = 1 / variance1
    inv_target_variance = 1 / target_variance
    C = torch.diag_embed(1 / (inv_variance1 + inv_target_variance))
    c = torch.matmul(C, torch.matmul(torch.diag_embed(inv_variance1), mean1[:, :, None]) + torch.matmul(
        torch.diag_embed(inv_target_variance), target_mean[:, :, None])).squeeze()
    log_z = MultivariateNormal(target_mean, torch.diag_embed(variance1 + target_variance + 1e-6)).log_prob(mean1)
    C = torch.diagonal(C, dim1=-2, dim2=-1)
    C = torch.log(C)

    return c, C, log_z
