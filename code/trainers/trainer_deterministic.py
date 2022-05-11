import random

from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from criterions.criterion_factory import criterion_factory
from datasets.dataset_factory import dataset_factory
from models.combiners.modality_combiner_factory import modality_combiner_factory
from models.image.image_model_factory import image_model_factory
from models.text.text_model_factory import text_model_factory

from datetime import datetime
import os
import yaml
import torch
import torch.distributions
import torch.utils.data
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import pprint

from optimizers.optimizer_factory import optimizer_factory
from trainers import torch_functions


pp = pprint.PrettyPrinter(indent=4)

class TrainerDeterministic:
    def __init__(self, config, path=None):

        self.config = config
        self.train_dataset = dataset_factory(config, split='train')
        self.test_dataset = dataset_factory(config, split='test')

        self.train_dataloader = self.train_dataset.get_loader()
        self.test_dataloader = self.test_dataset.get_loader()
        date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


        self.image_encoder = image_model_factory(config).cuda()
        self.text_encoder = text_model_factory(config, self.train_dataset.get_vocab().word2idx).cuda()
        self.modality_combiner = modality_combiner_factory(config).cuda()

        self.criterion = criterion_factory(config.criterion.train.name, config).cuda()
        params = self.get_model_params()

        self.optimizer = optimizer_factory(config, params)

        self.scheduler = StepLR(self.optimizer, step_size=self.config.optimizer.lr_decay_step,
                                gamma=self.config.optimizer.lr_decay_rate)

        if path:
            print("Path given. Loading previous model")
            self.load_models(path)

        self.save_model_path = os.path.join(config.models_root, config.dataset_name, date)
        self.save_config(self.save_model_path)

        print(yaml.dump(self.config))
        self.train_writer = SummaryWriter(os.path.join(self.save_model_path, "train"))
        self.test_writer = SummaryWriter(os.path.join(self.save_model_path, "test"))

    def set_train(self):
        self.image_encoder.train()
        self.text_encoder.train()
        self.modality_combiner.train()
        self.criterion.train()

    def set_eval(self):
        self.image_encoder.eval()
        self.text_encoder.eval()
        self.modality_combiner.eval()
        self.criterion.eval()

    def get_model_params(self):

        params = []
        params.append({
            'params': [p for p in self.image_encoder.cnn.fc.parameters()],
            'lr': self.config.optimizer.learning_rate
        })
        params.append({
            'params': [p for p in self.image_encoder.cnn.parameters()],
            'lr': 0.1 * self.config.optimizer.learning_rate
            })
        params_list = []
        params_list += [param for param in self.image_encoder.parameters()
                   if param.requires_grad]
        params_list += [param for param in self.text_encoder.parameters()
                   if param.requires_grad]
        params_list += [param for param in self.modality_combiner.parameters()
                   if param.requires_grad]
        params_list += [param for param in self.criterion.parameters()
                   if param.requires_grad]
        params.append({'params': params_list})

        for _, p1 in enumerate(params):  # remove duplicated params
            for _, p2 in enumerate(params):
                if p1 is not p2:
                    for p11 in p1['params']:
                        for j, p22 in enumerate(p2['params']):
                            if p11 is p22:
                                p2['params'][j] = torch.tensor(0.0, requires_grad=True)

        return params

    def train(self):
        batch_number = 0
        self.set_train()
        for epoch in range(self.config.train.num_epochs):

            # out = self.eval(self.test_dataset)
            # pp.pprint(out)
            # with open(os.path.join(self.save_model_path, "results_{}.txt".format(epoch + 1)),
            #           'wt') as fout:
            #     pprint.pprint(out, stream=fout)

            loss_total = 0

            for data in tqdm(self.train_dataloader, desc='Training for epoch ' + str(epoch)):
                source, source_modalities, source_lens, target = self.process_input(data)


                if batch_number % self.config.train.log_step == 0 and batch_number != 0:
                    with torch.no_grad():
                        loss = \
                            self.compute_loss(source, source_modalities, source_lens, target)
                        self.train_writer.add_scalar('retrieval_loss', loss, batch_number)

                    test_loss = self.compute_test_loss()

                    self.test_writer.add_scalar('retrieval_loss', test_loss, batch_number)

                loss = \
                    self.compute_loss(source, source_modalities, source_lens, target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_total += loss

                batch_number += 1

            print("loss is {}".format(loss_total))

            if epoch%self.config.train.epochs_between_checkpoints == self.config.train.epochs_between_checkpoints - 1:
                self.save_models(self.save_model_path, "model{}".format(epoch+1))
            if epoch % self.config.train.val_epochs == self.config.train.val_epochs - 1:
                out = self.eval(self.test_dataset)
                pp.pprint(out)
                with open(os.path.join(self.save_model_path, "results_{}.txt".format(epoch + 1)),
                          'wt') as fout:
                    pprint.pprint(out, stream=fout)

            self.scheduler.step()
        self.train_writer.close()
        self.test_writer.close()

    def process_input(self, data):

        source_modalities = random.choices(['image', 'text'], k=self.config.number_of_categories_combined)
        source = []

        for idx, modality in enumerate(source_modalities):
            current_modality = []
            for sample in data:
                current_modality.append(sample['source_data'][modality][idx])
            source.append(current_modality)

        source = [self.prepare_data_by_modality(modality, data) for data, modality in zip(source, source_modalities)]

        source_lens = [list(x) for x in zip(*[d['source_lens'] for d in data])]
        source_lens = [self.prepare_text_lens(lens) for lens in source_lens]
        target = self.prepare_image_data([d['target_img'] for d in data])

        return source, source_modalities, source_lens, target

    def prepare_data_by_modality(self, modality, data):
        if modality == 'image':
            return self.prepare_image_data(data)
        if modality == 'text':
            return self.prepare_text_data(data)

    def prepare_image_data(self, data):
        return torch.from_numpy(np.stack(data)).float().cuda()

    def prepare_text_data(self, data):
        return nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value=0).long().cuda()

    def prepare_text_lens(self, data):
        return torch.from_numpy(np.stack(data)).long().flatten().cuda()

    def encode_image(self, images):
        return self.image_encoder(images)

    def encode_text(self, texts):
        return self.text_encoder(texts)

    def encode(self, modality, data):
        if modality == 'image':
            return self.encode_image(data)
        if modality == 'text':
            return self.encode_text(data)

    def compute_loss(self, source, source_modalities, source_lens, target):

        source_embeddings = [self.encode(modality, data) for modality, data, lens in zip(source_modalities, source, source_lens)]

        target_embeddings = self.encode_image(target)

        query_embedding = self.modality_combiner(source_embeddings)

        loss = self.criterion(query_embedding, target_embeddings)

        return loss

    @torch.no_grad()
    def compute_test_loss(self):
        self.set_eval()

        data = next(iter(self.test_dataloader))
        source, source_modalities, source_lens, target = self.process_input(data)

        loss = self.compute_loss(source, source_modalities, source_lens, target)

        self.set_train()
        return loss


    def prepare_test_image(self, data):
        return torch.stack(data).float().cuda()

    def prepare_test_texts(self, data, lens):
        text_query = torch.zeros(len(data), max(lens)).long().cuda()
        for i, cap in enumerate(data):
            end = lens[i]
            text_query[i, :end] = cap[:end]
        return text_query

    def prepare_test_data_by_modality(self, modality, data, lens):
        if modality == 'image':
            return self.prepare_test_image(data)
        if modality == 'text':
            return self.prepare_test_texts(data, lens)


    @torch.no_grad()
    def eval(self, dataset):
        self.set_eval()
        out = []

        test_queries = dataset.get_test_queries()

        all_imgs_f = []
        data = []
        caption_lens = []

        all_target_ids = self.test_dataset.gallery
        all_target_cats = self.test_dataset.gallery_cats

        # compute all image features
        imgs = []
        for i in tqdm(range(len(all_target_ids))):
            imgs += [self.test_dataset.get_img_for_id_and_cat(all_target_ids[i])]
            if len(imgs) >= 128 or i == len(dataset.imgs) - 1:

                imgs = self.prepare_test_image(imgs)
                embeddings = self.encode_image(imgs)

                all_imgs_f += [embeddings.data.cpu().numpy()]
                imgs = []

        all_imgs_f = np.concatenate(all_imgs_f)

        for modality_combination, queries in test_queries.items():
            all_queries_f = []
            modalities = modality_combination.split('_')
            for t in tqdm(queries):
                torch.cuda.empty_cache()
                current_data = [dataset.retrieve_modality(modality, id, cat) for modality, id, cat in zip(modalities, t['images'], t['categories'])]
                data.append(current_data)
                caption_lens.append([torch.tensor([d.shape[0]]) if modality == 'text' else 0 for modality, d in zip(modalities, current_data)])

                if len(data) >= 128 or t is queries[-1]:

                    data = [list(x) for x in zip(*data)]
                    caption_lens = [list(x) for x in zip(*caption_lens)]

                    data = [self.prepare_test_data_by_modality(modality, d, lens) for d, modality, lens in zip(data, modalities, caption_lens)]
                    caption_lens = [self.prepare_text_lens(lens) for lens in caption_lens]

                    source_embeddings = [self.encode(modality, d) for modality, d, lens in
                                         zip(modalities, data, caption_lens)]

                    query_embedding = self.modality_combiner(source_embeddings)

                    f = query_embedding.data.cpu().numpy()

                    all_queries_f += [f]

                    data = []
                    caption_lens = []

            all_queries_f = np.concatenate(all_queries_f)

            queries_f_tensor = torch.from_numpy(all_queries_f)
            imgs_f_tensor = torch.from_numpy(all_imgs_f)

            # match test queries to target images, get nearest neighbors

            sims_mean_only = -torch_functions.pairwise_distances(queries_f_tensor, imgs_f_tensor)
            print(sims_mean_only.shape)

            sims_dict = {
                'mean-only': sims_mean_only.cpu(),
            }
            out += [('Modality combination: ', modality_combination)]

            for sim_type, sims in sims_dict.items():
                out += [('Sim type: ', sim_type)]
                print("sims shape: ", sims.shape)
                nn_result = [np.argsort(-sims[i, :])[:1500] for i in range(sims.shape[0])]

                # compute recalls

                nn_result = [[all_target_ids[nn] for nn in nns] for nns in nn_result]
                out += [('sample_size ', len(nn_result))]
                for k in [1,5,10, 50]:
                    r = 0.0
                    for i, nns in enumerate(nn_result):
                        query_cats = set(queries[i]['categories'])
                        query_imgs = [img for img, modality in zip(queries[i]['images'], modalities) if modality != 'text']
                        if any(query_cats.issubset(all_target_cats[x]) and x not in query_imgs for x in nns[:k]):
                            r += 1

                    r /= len(nn_result)
                    out += [('recall_top' + str(k) + '_correct_composition', r)]

                total = 0

                for i, nns in enumerate(nn_result):
                    query_cats = set(queries[i]['categories'])
                    number_of_images = len(
                        self.test_dataset.img_ids_per_cats['_'.join([str(x) for x in sorted(query_cats)])])

                    query_imgs = [img for img, modality in zip(queries[i]['images'], modalities) if modality != 'text']
                    number_of_positives = sum([query_cats.issubset(all_target_cats[x]) and x not in query_imgs for x in
                                               nns[:number_of_images]])
                    total += number_of_positives / number_of_images
                out += [('R_P score: ', total / len(nn_result))]


        self.set_train()

        return out

    def save_config(self, save_to):
        if not os.path.exists(save_to):
            os.makedirs(save_to)
        path = os.path.join(save_to, 'config.yaml')
        file = open(path, "w")
        yaml.dump(self.config, file)
        file.close()

    def save_models(self, save_to, name):
        if not os.path.exists(save_to):
            os.makedirs(save_to)
        state_dict = {
            'image_encoder': self.image_encoder.state_dict(),
            'text_encoder': self.text_encoder.state_dict(),
            'modality_combiner': self.modality_combiner.state_dict(),
            'criterion': self.criterion.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(state_dict, os.path.join(save_to, name))

    def load_models(self, load_from):

        state_dict = torch.load(load_from)

        self.image_encoder.load_state_dict(state_dict['image_encoder'])
        self.text_encoder.load_state_dict(state_dict['text_encoder'])
        self.modality_combiner.load_state_dict(state_dict['modality_combiner'])
        self.criterion.load_state_dict(state_dict['criterion'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
