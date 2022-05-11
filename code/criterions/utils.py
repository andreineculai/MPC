import torch


def distance_function_factory(function_name):
    if function_name == 'mean':
        return mean_distance
    if function_name == 'chamfer':
        return chamfer_distance
    if function_name == 'hausdorff':
        return hausdorff_distance


def hausdorff_distance(scores):
    return torch.min(torch.max(scores, dim=-1).values, dim=-1).values

def chamfer_distance(scores):
    return torch.mean(torch.max(scores, dim=-1).values, dim=-1)

def mean_distance(scores):
    return torch.mean(torch.mean(scores, dim=-1), dim=-1)

def sum_distance(scores):
    return torch.sum(torch.sum(scores, dim=-1), dim=-1)

def max_distance(scores):
    return torch.max(torch.max(scores, dim=-1).values, dim=-1).values
