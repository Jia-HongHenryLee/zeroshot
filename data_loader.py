import torch

import pandas as pd
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms

from os.path import join as PJ
import random
import yaml


def ConceptSets(state, concepts):

    split_list = state['split_list'] + ['general']
    data_dir = PJ('./dataset', state['dataset'], 'list')
    concept_file = pd.read_csv(PJ(data_dir, 'concepts', 'concepts_' + concepts + '.txt'))

    def _concept_split():
        if state['mode'] == "train_val":
            # load origin train_id
            ids = yaml.load(open(PJ(data_dir, 'train_test', 'id_split' + '.txt')))
            ids = ids['train_id']

            # random generate id_split
            val_nums = {'apy': 5, 'awa2': 13, 'cub': 50, 'sun': 65}
            val_num = val_nums[state['dataset']]
            random.shuffle(ids)

            id_split = {'train_id': sorted(ids[val_num:]), 'test_id': sorted(ids[:val_num])}
            print(id_split['train_id'])
            print(id_split['test_id'])

            # produce split data file
            data = pd.read_csv(PJ(data_dir, "train_val", "trainval.txt"), header=None)

            train_data = data[data.iloc[:, 1].isin(id_split['train_id'])]
            train_data.to_csv(PJ(data_dir, "train_val", "train.txt"), index=False, header=False)

            test_data = data[data.iloc[:, 1].isin(id_split['test_id'])]
            test_data.to_csv(PJ(data_dir, "train_val", "val.txt"), index=False, header=False)

            return id_split
        else:
            return yaml.load(open(PJ(data_dir, state['mode'], 'id_split' + '.txt')))

    def _concept(split_mode, concept_split):

        if split_mode in ['train', 'trainval', 'test_seen']:
            concept_label = concept_split['train_id']

        elif split_mode in ['val', 'test_unseen']:
            concept_label = concept_split['test_id']

        elif split_mode in ['general']:
            concept_label = list(range(concept_file.shape[1]))
        else:
            assert "Split Mode Error"

        concept_vector = {i: torch.cuda.FloatTensor(concept_file.iloc[:, i].values) for i in concept_label}

        id2label = {idx: label for idx, label in enumerate(concept_label)}

        return {'concept_label': concept_label, 'concept_vector': concept_vector, 'id2label': id2label}

    concept_split = _concept_split()
    return {s: _concept(s, concept_split) for s in split_list}


def ClassDatasets(state):

    split_list = state['split_list']

    class ClassDataset(Dataset):

        def __init__(self, split_mode, state):

            self.split_mode = split_mode
            self.root = PJ('./dataset', state['dataset'])
            self.csv_file = PJ(self.root, 'list', state['mode'], self.split_mode.strip("_g") + '.txt')

            self.data = pd.read_csv(self.csv_file, header=None)
            self.img_transform = self.img_transform()

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):

            image = Image.open(PJ(self.root, 'img', self.data.iloc[idx, 0])).convert('RGB')
            image = self.img_transform(image)

            label = torch.LongTensor([self.data.iloc[idx, 1]])

            sample = {'image': image, 'label': label}
            return sample

        def img_transform(self):

            if self.split_mode.find("train"):
                img_transform = transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])])
            else:
                img_transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])])

            return img_transform

    return {s: ClassDataset(s, state) for s in split_list}
