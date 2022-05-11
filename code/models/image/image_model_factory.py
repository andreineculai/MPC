from models.image.imge_model_pcme import ImageModelPCME
from models.image.image_model_tirg import ImageModelTirg


def image_model_factory(config):
    if config.model.name == 'pcme':
        return ImageModelPCME(config)
    if config.model.name == 'tirg':
        return ImageModelTirg(config)
