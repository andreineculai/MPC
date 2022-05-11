from models.combiners.mpc_combiner import MpcCombiner
from models.combiners.film_combiner import FilmCombiner
from models.combiners.mrn_combiner import MrnCombiner
from models.combiners.pcme_mlp_combiner import PcmeMlpCombiner
from models.combiners.pcme_addition_combiner import PcmeAdditionCombiner
from models.combiners.relationship_combiner import RelationshipCombiner
from models.combiners.tirg_combiner import TirgCombiner


def modality_combiner_factory(config):
    if config.model.combiner_type == 'mpc':
        return MpcCombiner(config)
    if config.model.combiner_type == 'tirg':
        return TirgCombiner(config)
    if config.model.combiner_type == 'film':
        return FilmCombiner(config)
    if config.model.combiner_type == 'relationship':
        return RelationshipCombiner(config)
    if config.model.combiner_type == 'mrn':
        return MrnCombiner(config)
    if config.model.combiner_type == 'pcme_addition':
        return PcmeAdditionCombiner(config)
    if config.model.combiner_type == 'pcme_mlp':
        return PcmeMlpCombiner(config)

