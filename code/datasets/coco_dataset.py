import json
import random

import torch
import torchvision
from PIL import Image
import os
from torch.utils.data import Dataset
from third_party.pcme.datasets._dataloader import load_vocab
from third_party.pcme.datasets._transforms import tokenize
from third_party.pcme.datasets.vocab import Vocabulary
from pycocotools.coco import COCO
import math
from datasets.base_dataset import BaseDataset

import torchvision.transforms.functional as F

class Coco(BaseDataset):

    def __init__(self, config, split, transform = None):
        super().__init__(config, split)

        if transform is None:
            if split == 'train':
                self.transform = self.get_augmentation_transform()
                self.caption_dropout = self.config.dataloader.get('caption_dropout', 0)
            else:
                self.transform = self.get_default_transform()
                self.caption_dropout = 0
        else:
            self.transform = transform
        self.caption_start_end_token = (self.config.dataloader.get('caption_start_end_token', True) == True)

        self.path = os.path.join(config.dataset_root, "coco")
        self.coco_api = COCO(annotation_file=os.path.join(self.path, 'instances_train2017.json'))
        self.region_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.region_model.eval()
        self.full_regions = torch.Tensor([0, 0, 224, 224]).repeat(4, 1)
        self.split = split

        vocab_root = config.vocab_root
        vocab_path = os.path.join(vocab_root, "coco_vocab.pkl")
        vocab = load_vocab(vocab_path)
        self.vocab = vocab

        self.k = self.config.number_of_categories_combined
        # self.k = 1
        self.cat_combinations = self.load_cat_combinations()
        self.img_ids_per_cats = self.load_img_ids_per_cats()


        self.img_root_path = os.path.join(config.imgs_root, 'train2017')

        if split == 'val' or split == 'test':
            self.test_queries = self.load_test_queries()
            self.gallery = self.load_gallery()
            self.gallery_cats = self.load_gallery_cats()

    def get_cats_for_image_id(self, id):
        annotation_ids = self.coco_api.getAnnIds(id)
        return [x['category_id'] for x in self.coco_api.loadAnns(annotation_ids)]

    def load_gallery(self):
        with open(os.path.join(self.path, "{}_imgs.json".format(self.split))) as json_file:
            return json.load(json_file)

    def load_cats_for_feasibility_study(self):
        with open(os.path.join(self.path, "1_cat_combinations.json")) as json_file:
            seen = json.load(json_file)
        with open(os.path.join(self.path, "1_cat_combinations_zero_shot.json")) as json_file:
            unseen = json.load(json_file)
        with open(os.path.join(self.path, "1_cat_combinations_impossible.json")) as json_file:
            impossible = json.load(json_file)
        # seen = random.sample(seen, len(impossible))
        return seen, unseen, impossible

    def load_gallery_cats(self):
        gallery_cats = {}
        for id in self.gallery:
            gallery_cats[id] = self.get_cats_for_image_id(id)

        return gallery_cats

    def load_set_algebra_test_cases(self):
        with open(os.path.join(self.path, "set_algebra_test_cases.json")) as json_file:
            return json.load(json_file)

    def load_test_queries(self):
        with open(os.path.join(self.path, "{}_{}_cases.json".format(self.k, self.split))) as json_file:
            return json.load(json_file)

    def load_cat_combinations(self):
        with open(os.path.join(self.path, "{}_cat_combinations.json".format(self.k))) as json_file:
            return json.load(json_file)

    def load_img_ids_per_cats(self):
        with open(os.path.join(self.path, "{}_img_ids_per_cats_{}.json".format(self.k, self.split))) as json_file:
            return json.load(json_file)

    def get_text_for_cat(self, cat):
        text = self.coco_api.loadCats(cat)[0]['name']
        return tokenize(text, self.vocab, self.caption_dropout, self.caption_start_end_token).long()

    def get_text_for_cats(self, cats):
        text = ' '.join([self.coco_api.loadCats(cat)[0]['name'] for cat in cats])
        return tokenize(text, self.vocab, self.caption_dropout, self.caption_start_end_token).long()

    def retrieve_modality(self, modality, id, cat):
        if modality == 'image':
            return self.get_img_for_id_and_cat(id, cat)
        if modality == 'text':
            return self.get_text_for_cat(cat)

    def get_img_for_id_and_cat(self, id, cat=None):
        img_path = os.path.join(self.img_root_path, self.format_img_file_for_id(id))
        with open(img_path, 'rb') as f:
            img = Image.open(f).convert('RGB')

        if cat:
            annIds = self.coco_api.getAnnIds(imgIds=id, catIds=cat, iscrowd=None)
            anns = self.coco_api.loadAnns(annIds)
            bbox = anns[0]['bbox']
            img = img.crop((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))
        img = self.transform(img)
        return img

    def get_img_for_id_and_cats(self, id, cats):
        img_path = os.path.join(self.img_root_path, self.format_img_file_for_id(id))
        with open(img_path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        new_image = Image.new('RGB', img.size, (0, 0, 0))
        for cat in cats:
            annIds = self.coco_api.getAnnIds(imgIds=id, catIds=cat, iscrowd=None)
            anns = self.coco_api.loadAnns(annIds)
            bbox = anns[0]['bbox']
            bbox = [math.floor(bbox[0]), math.floor(bbox[1]), math.ceil(bbox[2]), math.ceil(bbox[3])]

            patch = img.crop((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))
            new_image.paste(patch, (bbox[0], bbox[1]))
        img = self.transform(new_image)
        return img

    def format_img_file_for_id(self, id):
        return str(id).zfill(12) + '.jpg'


    def get_vocab(self):
        return self.vocab


    def __getitem__(self, idx):

        cats = self.cat_combinations[idx]
        ids = [random.choice(self.img_ids_per_cats[str(cat)]) for cat in cats]
        data = {
            'image': [self.retrieve_modality('image', id, cat) for id, cat in zip(ids, cats)],
            'text': [self.retrieve_modality('text', id, cat) for id, cat in zip(ids, cats)],
        }

        target_id = random.choice(self.img_ids_per_cats['_'.join([str(cat) for cat in cats])])
        target_img = self.retrieve_modality('image', target_id, cat=None)

        out = {}
        out['source_data'] = data
        out['source_lens'] = [torch.tensor([text.shape[0]]) for text in data['text']]
        out['target_img'] = target_img

        return out

    def __len__(self):
        return len(self.cat_combinations)

def crop_and_resize(img, x1, y1, x2, y2):
    x1, y1, x2, y2 = math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2)
    img = F.resized_crop(img, y1, x1, y2 - y1, x2 - x1, [224, 224])
    return img
