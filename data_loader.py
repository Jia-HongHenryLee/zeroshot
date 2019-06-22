import torch

import pandas as pd
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms

import os
from os.path import join as PJ
import numpy as np
import yaml


def ConceptSets(state, concepts):

    split_list = state['split_list'] + ['general']
    data_dir = PJ('./dataset', state['dataset'], 'list')
    concept_file = pd.read_csv(PJ(data_dir, 'concepts', 'concepts_' + concepts + '.txt'))

    def _concept_split():
        if state['mode'] == "train_val":
            # load origin train_id
            data = pd.read_csv(PJ(data_dir, "train_test", "train.txt"), header=None)
            train_path = PJ(data_dir, "train_val", "train.txt")
            val_path = PJ(data_dir, "train_val", "val.txt")

            if not (os.path.exists(train_path) and os.path.exists(val_path)):

                msk = np.random.rand(len(data)) < 0.8

                train_data = data[msk]
                train_data.to_csv(train_path, index=False, header=False)

                val_data = data[~msk]
                val_data.to_csv(val_path, index=False, header=False)

        return yaml.load(open(PJ(data_dir, "train_test", 'id_split' + '.txt')))

    def _concept(split_mode, concept_split):

        if split_mode in ['train', 'trainval', 'test_seen']:
            concept_label = concept_split['train_id']

        elif split_mode in ['val', 'test', 'test_unseen']:
            concept_label = concept_split['test_id']

        elif split_mode in ['general']:
            concept_label = list(range(concept_file.shape[1]))
        else:
            assert "Split Mode Error"

        concept_vector = [torch.cuda.FloatTensor(concept_file.iloc[:, i].values) for i in concept_label]

        return {'concept_label': concept_label, 'concept_vector': torch.stack(concept_vector)}

    concept_split = _concept_split()
    return {s: _concept(s, concept_split) for s in split_list}


def ClassDatasets(state):

    split_list = state['split_list']

    class ClassDataset(Dataset):

        def __init__(self, split_mode, state):

            self.dataset = state['dataset']

            self.split_mode = split_mode
            self.root = PJ('./dataset', state['dataset'])
            self.csv_file = PJ(self.root, 'list', state['mode'], self.split_mode.strip("_g") + '.txt')

            self.data = pd.read_csv(self.csv_file, header=None)

            if state['dataset'] == 'nus_wide':
                self.neg_weight = 0.5
            else:
                pos = (self.data.iloc[:, 1:] == 1).sum().sum()
                neg = (self.data.iloc[:, 1:] == 0).sum().sum()
                self.neg_weight = pos / (pos + neg)
                self.class_weight = (self.data.iloc[:, 1:] == 1).sum()
                self.class_weight = self.class_weight.reset_index(drop=True)

            self.img_transform = self.img_transform()

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):

            if state['dataset'] == 'nus_wide':
                image = Image.open(PJ(self.root, self.data.iloc[idx, 0])).convert('RGB')

            else:
                image = Image.open(PJ(self.root, 'img', self.data.iloc[idx, 0])).convert('RGB')

            image = self.img_transform(image)
            label = torch.FloatTensor(self.data.iloc[idx, 1:].tolist())

            sample = {'image': image, 'label': label}
            return sample

        def img_transform(self):

            if self.split_mode.find("train") != -1:
                img_transform = transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor()])
                    # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                    #                      std=[0.229, 0.224, 0.225])])
            else:
                img_transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor()])
                    # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                    #                      std=[0.229, 0.224, 0.225])])

            return img_transform

    return {s: ClassDataset(s, state) for s in split_list}
