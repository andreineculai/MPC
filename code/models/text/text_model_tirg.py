# Copyright 2018 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Class for text data."""
import string
import numpy as np
import torch
import torchtext


class TextModelTirg(torch.nn.Module):

    def __init__(self,
                 vocab,
                 word_embed_dim=300,
                 lstm_hidden_dim=1024,
                 output_dim=512):

        super(TextModelTirg, self).__init__()

        self.vocab = vocab
        vocab_size = len(self.vocab)

        self.word_embed_dim = word_embed_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.embedding_layer = torch.nn.Embedding(vocab_size, word_embed_dim)
        self.embedding_layer.weight.requires_grad = True

        self.lstm = torch.nn.LSTM(word_embed_dim, lstm_hidden_dim)
        self.fc_output = torch.nn.Sequential(
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(lstm_hidden_dim, output_dim),
        )
        self.init_weights(vocab)

    def init_weights(self, word2idx):
        wemb = torchtext.vocab.GloVe(name='42B')

        # quick-and-dirty trick to improve word-hit rate
        missing_words = []
        for word, idx in word2idx.items():
            if word not in wemb.stoi:
                word = word.replace('-', '').replace('.', '').replace("'", '')
                if '/' in word:
                    word = word.split('/')[0]
            if word in wemb.stoi:
                self.embedding_layer.weight.data[idx] = wemb.vectors[wemb.stoi[word]]
            else:
                missing_words.append(word)
        print('Words: {}/{} found in vocabulary; {} words missing'.format(
            len(word2idx) - len(missing_words), len(word2idx), len(missing_words)))

    def forward(self, x):
        return self.forward_encoded_texts(x)

    def forward_encoded_texts(self, texts):
        # to tensor
        lengths = [len(t) for t in texts]
        itexts = torch.zeros((np.max(lengths), len(texts))).long()
        for i in range(len(texts)):
            itexts[:lengths[i], i] = torch.tensor(texts[i])

        # embed words
        itexts = torch.autograd.Variable(itexts).cuda()
        etexts = self.embedding_layer(itexts)

        # lstm
        lstm_output, _ = self.forward_lstm_(etexts)

        # get last output (using length)
        text_features = []
        for i in range(len(texts)):
            text_features.append(lstm_output[lengths[i] - 1, i, :])

        # output
        text_features = torch.stack(text_features)
        text_features = self.fc_output(text_features)
        return text_features

    def forward_lstm_(self, etexts):
        batch_size = etexts.shape[1]
        first_hidden = (torch.zeros(1, batch_size, self.lstm_hidden_dim),
                        torch.zeros(1, batch_size, self.lstm_hidden_dim))
        first_hidden = (first_hidden[0].cuda(), first_hidden[1].cuda())
        lstm_output, last_hidden = self.lstm(etexts, first_hidden)
        return lstm_output, last_hidden
