""" Caption encoder based on PVSE implementation.
Reference code:
    https://github.com/yalesong/pvse/blob/master/model.py
"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import torchtext

from third_party.pcme.models.pie_model import PIENet
from third_party.pcme.models.uncertainty_module import UncertaintyModuleText
from third_party.pcme.utils.tensor_utils import l2_normalize, sample_gaussian_tensors


def get_pad_mask(max_length, lengths, set_pad_to_one=True):
    ind = torch.arange(0, max_length).unsqueeze(0).to(lengths.device)
    mask = (ind >= lengths.unsqueeze(1)) if set_pad_to_one \
        else (ind < lengths.unsqueeze(1))
    mask = mask.to(lengths.device)
    return mask


class EncoderText(nn.Module):
    def __init__(self, config, word2idx):
        super(EncoderText, self).__init__()

        wemb_type, word_dim, embed_dim = \
            config.wemb_type, config.word_dim, config.embed_size
        self.config = config
        self.embed_dim = embed_dim
        self.use_attention = True
        self.use_probemb = True
        self.num_embeds = config.num_embeds
        # Word embedding
        self.embed = nn.Embedding(len(word2idx), word_dim)
        self.embed.weight.requires_grad = False

        # Sentence embedding
        self.rnn = nn.GRU(word_dim, embed_dim // 2, bidirectional=True, batch_first=True)

        if self.use_attention:
            self.pie_net = PIENet(self.num_embeds, word_dim, embed_dim, word_dim // 2)

        self.uncertain_net = UncertaintyModuleText(self.num_embeds, word_dim, embed_dim, word_dim // 2)
        self.init_weights(wemb_type, word2idx, word_dim)

        # self.n_samples_inference = config.get('n_samples_inference', 0)

    def init_weights(self, wemb_type, word2idx, word_dim):
        if wemb_type is None:
            nn.init.xavier_uniform_(self.embed.weight)
        else:
            # Load pretrained word embedding
            if 'fasttext' == wemb_type.lower():
                wemb = torchtext.vocab.FastText()
            elif 'glove' == wemb_type.lower():
                wemb = torchtext.vocab.GloVe(name=self.config.glove_size)
            else:
                raise Exception('Unknown word embedding type: {}'.format(wemb_type))
            assert wemb.vectors.shape[1] == word_dim

            # quick-and-dirty trick to improve word-hit rate
            missing_words = []
            for word, idx in word2idx.items():
                if word not in wemb.stoi:
                    word = word.replace('-', '').replace('.', '').replace("'", '')
                    if '/' in word:
                        word = word.split('/')[0]
                if word in wemb.stoi:
                    self.embed.weight.data[idx] = wemb.vectors[wemb.stoi[word]]
                else:
                    missing_words.append(word)
            print('Words: {}/{} found in vocabulary; {} words missing'.format(
                len(word2idx) - len(missing_words), len(word2idx), len(missing_words)))

    def forward(self, x, lengths, n_samples=1):
        # Embed word ids to vectors
        wemb_out = self.embed(x)
        # Forward propagate RNNs
        packed = pack_padded_sequence(wemb_out, lengths.cpu(), batch_first=True, enforce_sorted=False)
        if torch.cuda.device_count() > 1:
            self.rnn.flatten_parameters()
        rnn_out, _ = self.rnn(packed)

        padded = pad_packed_sequence(rnn_out, batch_first=True)

        # Reshape *final* output to (batch_size, hidden_size)
        I = lengths.expand(self.embed_dim, 1, -1).permute(2, 1, 0) - 1
        out = torch.gather(padded[0], 1, I).squeeze(1)

        output = {}
        output['embedding_no_attn'] = out

        if self.use_attention:
            pad_mask = get_pad_mask(wemb_out.shape[1], lengths, True)
            out, attn, residual = self.pie_net(out, wemb_out, pad_mask)
            # output['attention'] = attn
            # output['residual'] = residual

        if self.use_probemb:
            if not self.use_attention:
                pad_mask = get_pad_mask(wemb_out.shape[1], lengths, True)
            uncertain_out = self.uncertain_net(wemb_out, pad_mask, lengths)
            logsigma = uncertain_out['logsigma']
            output['logsigma'] = logsigma
            # output['uncertainty_attention'] = uncertain_out['attention']

        # out = l2_normalize(out)
        if self.use_probemb and n_samples > 1:
            output['embedding'] = sample_gaussian_tensors(out, logsigma, n_samples)
        else:
            output['embedding'] = out

        # output['logsigma'] = torch.zeros_like(output['logsigma'])
        return output
