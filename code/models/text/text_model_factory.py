from models.text.text_model_pcme import TextModelPCME
from models.text.text_model_tirg import TextModelTirg


def text_model_factory(config, vocab=None):
    if config.model.name == 'pcme':
        return TextModelPCME(config, vocab)
    if config.model.name == 'tirg':
        return TextModelTirg(vocab)
