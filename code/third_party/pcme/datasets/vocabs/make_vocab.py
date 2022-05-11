""" a script for making vocabulary pickle.

Original code:
https://github.com/yalesong/pvse/blob/master/vocab.py
"""

import nltk
import pickle
from collections import Counter
import fire
import os
from tqdm import tqdm


class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

# def from_txt(txt):
#     captions = []
#     with open(txt, 'rb') as f:
#         for line in f:
#             captions.append(line.strip())
#     return captions

def from_txt(txt):
    captions = []
    with open(txt, 'rb') as f:
        for line in f:
            line = line.decode('utf-8').split('\t')
            captions.append(caption_post_process(line[2].strip()))
    return captions


def caption_post_process(s):
    return s.strip().replace('.',
                             'dotmark').replace('?', 'questionmark').replace(
        '&', 'andmark').replace('*', 'starmark')


def build_vocab(data_path, threshold):
    """Build a simple vocabulary wrapper."""
    counter = Counter()
    captions = []
    for cname in os.listdir(data_path):
        # for fname in os.listdir(os.path.join(data_path, cname)):
        full_path = os.path.join(data_path, cname)
        captions.extend(from_txt(full_path))

    for i, caption in tqdm(enumerate(captions), total=len(captions)):
        tokens = nltk.tokenize.word_tokenize(
            caption.lower())
        counter.update(tokens)

    # Discard if the occurrence of the word is less than min_word_cnt.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')
    vocab.add_word('replace')
    vocab.add_word('with')

    # Add words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    dict_vocab = {
        'idx': vocab.idx,
        'idx2word': vocab.idx2word,
        'word2idx': vocab.word2idx,
    }
    return dict_vocab


def main(data_path):
    vocab = build_vocab(data_path, threshold=4)
    with open('./vocab_local.pkl', 'wb') as f:
        pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    fire.Fire(main)
