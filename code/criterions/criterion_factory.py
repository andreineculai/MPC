from .mpc_criterion import MpcCriterion
from .pcme_criterion import PcmeCriterion
from .tirg_criterion import TirgCriterion


def criterion_factory(criterion_name, config):
    if criterion_name == 'mpc':
        return MpcCriterion(config)
    if criterion_name == 'tirg':
        return TirgCriterion(config)
    if criterion_name == 'pcme':
        return PcmeCriterion(config)