""" Create a vocabulary wrapper.

Original code:
https://github.com/yalesong/pvse/blob/master/vocab.py
"""

from collections import Counter
import json
import os
import pickle

import fire
from nltk.tokenize import word_tokenize
from pycocotools.coco import COCO

ANNOTATIONS = {
    'mrw': ['mrw-v1.0.json'],
    'tgif': ['tgif-v1.0.tsv'],
    'coco': ['annotations/captions_train2017.json',
             'annotations/captions_val2017.json'],
    'mit_states': ['adj_ants.csv'],
    'fashion_200k': ['labels/dress_train_detect_all.txt',
                     'labels/dress_test_detect_all.txt',
                     'labels/jacket_train_detect_all.txt',
                     'labels/jacket_test_detect_all.txt',
                     'labels/pants_train_detect_all.txt',
                     'labels/pants_test_detect_all.txt',
                     'labels/skirt_train_detect_all.txt',
                     'labels/skirt_test_detect_all.txt',
                     'labels/top_train_detect_all.txt',
                     'labels/top_test_detect_all.txt'
                     ],
    'fashion_iq': ['captions/cap.dress.train.json',
                   'captions/cap.dress.val.json',
                   'captions/cap.shirt.train.json',
                   'captions/cap.shirt.val.json',
                   'captions/cap.toptee.train.json',
                   'captions/cap.toptee.val.json',
                   ]
}


class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(self):
        self.idx = 0
        self.word2idx = {}
        self.idx2word = {}

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def load_from_pickle(self, data_path):
        with open(data_path, 'rb') as fin:
            data = pickle.load(fin)

        self.idx = data.idx
        self.word2idx = data.word2idx
        self.idx2word = data.idx2word
        # self.idx = data['idx']
        # self.word2idx = data['word2idx']
        # self.idx2word = data['idx2word']
    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


def from_tgif_tsv(path):
    captions = [line.strip().split('\t')[1]
                for line in open(path, 'r').readlines()]
    return captions


def from_mrw_json(path):
    dataset = json.load(open(path, 'r'))
    captions = []
    for datum in dataset:
        cap = datum['sentence']
        cap = cap.replace('/r/', '')
        cap = cap.replace('r/', '')
        cap = cap.replace('/u/', '')
        cap = cap.replace('u/', '')
        cap = cap.replace('..', '')
        cap = cap.replace('/', ' ')
        cap = cap.replace('-', ' ')
        captions += [cap]
    return captions


def from_coco_json(path):
    coco = COCO(path)
    ids = coco.anns.keys()
    captions = []
    for idx in ids:
        captions.append(str(coco.anns[idx]['caption']))
    return captions


def from_txt(path):
    captions = []
    with open(path, 'rb') as f:
        for line in f:
            captions.append(line.strip())
    return captions


def from_fashion_200k(path):
    captions = []
    with open(path, 'rb') as f:
        for line in f:
            line = line.decode('utf-8').strip().split('\t')
            captions.append(' '.join(line[2:]))
    return captions

def from_fashion_iq(path):
    captions = []
    with open(path, "r") as content:
        labels_json = json.loads(content.read())
        for asin in labels_json:
            captions.append(' '.join(asin['captions']))
    return captions


def build_vocab(data_path, data_name, jsons, threshold):
    """Build a simple vocabulary wrapper."""
    # counter = Counter()
    # for path in jsons[data_name]:
    #     print(path)
    #     full_path = os.path.join(os.path.join(data_path, data_name), path)
    #     if data_name == 'tgif':
    #         captions = from_tgif_tsv(full_path)
    #     elif data_name == 'mrw':
    #         captions = from_mrw_json(full_path)
    #     elif data_name == 'coco':
    #         captions = from_coco_json(full_path)
    #     elif data_name == 'fashion_200k':
    #         captions = from_fashion_200k(full_path)
    #     elif data_name == 'fashion_iq':
    #         captions = from_fashion_iq(full_path)
    #     else:
    #         captions = from_txt(full_path)
    #
    #     for caption in captions:
    #         # if data_name != 'fashion_200k':
    #         #     caption = caption.decode('utf-8')
    #
    #         tokens = word_tokenize(caption.lower())
    #         counter.update(tokens)

    captions = []
    coco = COCO(os.path.join(os.path.join(data_path, data_name), 'annotations/instances_train2017.json'))
    cats_names = [x['name'] for x in coco.loadCats(coco.getCatIds())]
    captions.append(' '.join(cats_names))
    counter = Counter()
    print(captions)
    for caption in captions:
        # if data_name != 'fashion_200k':
        #     caption = caption.decode('utf-8')

        tokens = word_tokenize(caption.lower())
        counter.update(tokens)

    # Discard if the occurrence of the word is less than min_word_cnt.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')
    # Add words to the vocabulary.
    if data_name == 'fashion_200k':
        vocab.add_word('replace')
        vocab.add_word('with')

    for word in words:
        print(word)
        vocab.add_word(word)
    print('Vocabulary size: {}'.format(len(words)))

    return vocab


def main(data_path, data_name, threshold=0):
    vocab = build_vocab(data_path, data_name, jsons=ANNOTATIONS, threshold=threshold)
    if not os.path.isdir('./vocabs'):
        os.makedirs('./vocabs')
    with open('./vocabs/%s_vocab.pkl' % data_name, 'wb') as f:
        pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)
    print("Saved vocabulary file to ", './vocabs/%s_vocab.pkl' % data_name)


if __name__ == '__main__':
    fire.Fire(main)
