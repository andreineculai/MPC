"""Distance modules for retrieval
"""
import numpy as np
import torch
import torch.nn as nn


def gmean_sigma(sigma):
    if len(sigma.shape) != 1:
        raise ValueError(sigma.shape)
    sigma = np.exp(sigma / 2) ** (1.0 / sigma.shape[0])
    return sigma.prod()


def dismatch_prob(samples, negative_scale, shift, eps=1e-6, return_negative_logit=False):
    N = len(samples)
    samples1 = samples.unsqueeze(1)
    samples2 = samples.unsqueeze(2)
    dists = torch.sqrt(((samples1 - samples2) ** 2).sum(-1) + eps).view(N, -1)
    logits = -negative_scale * dists + shift
    if return_negative_logit:
        return -logits.mean().detach().cpu().numpy()
    match_prob = torch.exp(logits) / (torch.exp(logits) + torch.exp(-logits))
    # match_prob = match_prob.mean(axis=1).detach().cpu().numpy()
    match_prob = match_prob.mean().detach().cpu().numpy()
    return 1 - match_prob


class MatMulModule(nn.Module):
    def set_g_features(self, g_features, g_sigmas=None):
        self._g_features = g_features
        self.g_features = None

    def forward(self, q_features, n_embeddings=1, reduction=None,
                embeddings_come_from_same_source=False,
                q_indices=None, q_sigmas=None):
        if self.g_features is None:
            self.g_features = self._g_features.to(q_features.device)
        sims = q_features.mm(self.g_features)

        if n_embeddings > 1:
            sims = sims.view(int(len(q_features) / n_embeddings),
                             n_embeddings,
                             int(self.g_features.size()[-1] / n_embeddings),
                             n_embeddings)
            sims = sims.permute(0, 1, 3, 2)

            if reduction == 'sum':
                sims = torch.sum(torch.sum(sims, axis=1), axis=1)
            elif reduction == 'max':
                sims = torch.max(torch.max(sims, axis=1)[0], axis=1)[0]

        if embeddings_come_from_same_source:
            # ignore self
            for idx, qidx in enumerate(q_indices):
                sims[idx, qidx] = -np.inf
        sims, pred_ranks = (-sims).sort()
        return sims, pred_ranks


class MatchingProbModule(nn.Module):
    def __init__(self, match_prob_fn):
        super().__init__()
        self.match_prob_fn = match_prob_fn

    def set_g_features(self, g_features, g_sigmas=None):
        self._g_features = g_features
        self.g_features = None

    def forward(self, q_features, n_embeddings=1, reduction=None,
                embeddings_come_from_same_source=False,
                q_indices=None, q_sigmas=None):
        if self.g_features is None:
            self.g_features = self._g_features.to(q_features.device)
        sims = torch.zeros(len(q_features), len(self.g_features))
        for idx, q_feature in enumerate(q_features):
            _sim = self.match_prob_fn(q_feature.unsqueeze(0), self.g_features)
            sims[idx] = _sim
        sims, pred_ranks = (-sims).sort()
        return sims, pred_ranks


def elk_dist(mu1, mu2, log_sigma1, log_sigma2):
    sum_sigma_sq = torch.exp(2 * log_sigma1) + torch.exp(2 * log_sigma2)
    dist = (mu1 - mu2) ** 2
    dist = dist.squeeze(1)
    elk = dist / (sum_sigma_sq) + torch.log(sum_sigma_sq)
    return -0.5 * torch.sum(elk, dim=1)


def kl_dist(mu1, mu2, log_sigma1, log_sigma2):
    dist = (mu1 - mu2) ** 2
    dist = dist.squeeze(1)

    var2 = torch.exp(log_sigma2 * 2)
    dist = dist / var2 + 2 * (log_sigma1 - log_sigma2) + torch.exp(log_sigma1 * 2) / var2
    return -torch.sum(dist, dim=1)


def js_dist(mu1, mu2, log_sigma1, log_sigma2):
    dist = (mu1 - mu2) ** 2
    dist = dist.squeeze(1)

    var1 = torch.exp(log_sigma1 * 2)
    var2 = torch.exp(log_sigma2 * 2)
    dist = (dist + var1) / var2 + (dist + var2) / var1
    return -torch.sum(dist, dim=1)


def bhattacharyya_dist(mu1, mu2, log_sigma1, log_sigma2):
    dist = (mu1 - mu2) ** 2
    dist = dist.squeeze(1)
    sigma1 = torch.exp(log_sigma1)
    sigma2 = torch.exp(log_sigma2)

    dist = dist / (torch.exp(log_sigma1 * 2) + torch.exp(log_sigma2 * 2))
    dist = dist + 2 * torch.log(sigma1 / sigma2 + sigma2 / sigma1)
    ddd = 2 * torch.log(torch.ones(1) * 2)
    dist = dist.float() - ddd.to(dist.device).float()
    dist = dist / 4
    return -torch.sum(dist, dim=1)


def wasserstein_dist(mu1, mu2, log_sigma1, log_sigma2):
    dist = (mu1 - mu2) ** 2
    dist = dist.squeeze(1)
    dist = dist + (torch.exp(log_sigma1) - torch.exp(log_sigma2)) ** 2
    return -torch.sum(dist, dim=1)


class HybridDistanceModule(nn.Module):
    def __init__(self, dist_fn_name):
        super().__init__()
        print(f'Initializing {dist_fn_name} module')
        if dist_fn_name == 'elk':
            self.dist_fn = elk_dist
        elif dist_fn_name == 'wasserstein':
            self.dist_fn = wasserstein_dist
        elif dist_fn_name == 'kl':
            self.dist_fn = kl_dist
        elif dist_fn_name == 'js':
            self.dist_fn = js_dist
        elif dist_fn_name == 'bhattacharyya':
            self.dist_fn = bhattacharyya_dist
        else:
            raise ValueError(dist_fn_name)

    def set_g_features(self, g_features, g_sigmas=None):
        self._g_features = g_features
        self.g_features = None
        self._g_sigmas = torch.from_numpy(g_sigmas)
        self.g_sigmas = None

    def forward(self, q_features, n_embeddings=1, reduction=None,
                embeddings_come_from_same_source=False,
                q_indices=None, q_sigmas=None):
        if self.g_features is None:
            self.g_features = self._g_features.to(q_features.device)
            self.g_sigmas = self._g_sigmas.to(q_features.device)

        sims = torch.zeros(len(q_features), len(self.g_features))

        q_sigmas = torch.from_numpy(q_sigmas).to(q_features.device)
        for idx, (q_feature, q_sigma) in enumerate(zip(q_features, q_sigmas)):
            _sim = self.dist_fn(q_feature.unsqueeze(0), self.g_features, q_sigma.unsqueeze(0), self.g_sigmas)
            sims[idx] = _sim

        sims, pred_ranks = (-sims).sort()
        return sims, pred_ranks
