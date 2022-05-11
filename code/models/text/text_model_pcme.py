from third_party.pcme.datasets._dataloader import load_vocab
from third_party.pcme.datasets.vocab import Vocabulary
from third_party.pcme.models.caption_encoder import EncoderText
from third_party.pcme.config import parse_config
import torch.nn as nn


class TextModelPCME(nn.Module):

    def __init__(self, config, vocab):
        super(TextModelPCME, self).__init__()
        self.encoder = EncoderText(config.model, vocab)

    def forward(self, data, lengths, n_samples=1):
        return self.encoder.forward(data, lengths, n_samples)
