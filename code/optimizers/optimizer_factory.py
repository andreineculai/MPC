from torch.optim import Adam, SGD


def optimizer_factory(config, params):
    if config.optimizer.name == 'adam':
        return Adam(params, lr=config.optimizer.learning_rate, weight_decay=config.optimizer.weight_decay)
    if config.optimizer.name == 'sgd':
        return SGD(params, lr=config.optimizer.learning_rate, momentum=0.9, weight_decay=config.optimizer.weight_decay)

