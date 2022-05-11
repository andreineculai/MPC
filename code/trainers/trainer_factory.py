from trainers.trainer_probabilistic import TrainerProbabilistic
from trainers.trainer_deterministic import TrainerDeterministic


def trainer_factory(trainer_name, config, path=None):
    if trainer_name == 'probabilistic':
        return TrainerProbabilistic(config, path=path)
    if trainer_name == 'deterministic':
        return TrainerDeterministic(config, path=path)
