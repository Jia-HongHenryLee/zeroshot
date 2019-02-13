import torch

import pandas as pd
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms

from os.path import join as PJ
from gensim.models import KeyedVectors
import numpy as np

import json


class ClassDataset(Dataset):

    def __init__(self, dataset='apy', mode='train_val', img_transform=None, label_transform=None):
        self.mode = mode
        self.dataset = dataset
        self.root = PJ('./dataset', dataset)

        self.img_root = PJ(self.root, 'img')
        self.csv_file = PJ(self.root, 'list', mode + '.txt')

        self.data = pd.read_csv(self.csv_file, header=None)
        self.img_transform = img_transform
        self.label_transform = label_transform['concept_id']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        image = Image.open(PJ(self.img_root, self.data.iloc[idx, 0])).convert('RGB')
        label = torch.LongTensor([self.data.iloc[idx, 1]])

        if self.img_transform:
            image = self.img_transform(image)

        if self.label_transform is not None:
            label = torch.nonzero(self.label_transform == label).reshape(-1)

        sample = {'image': image, 'label': label}
        return sample


class LabelSet():

    def __init__(self, dataset='apy', word2vec=None):

        self.root = PJ('./dataset', dataset)
        self.word2vec = word2vec

        with open(PJ(self.root, 'list', 'id_split.txt')) as f:
            self.id_split = json.load(f)
        self.concepts = pd.read_csv(PJ(self.root, 'list', 'concepts.txt'), header=None).iloc[:, 0].values

        self.train = self._concept('train_id')
        self.test = self._concept('test_id')
        self.all = self._concept(None)

    def _concept(self, mode):

        concept_id = torch.arange(len(self.concepts)) if mode is None else torch.LongTensor(self.id_split[mode])
        concept_vectors = torch.FloatTensor([self.word2vec[self.concepts[i]].tolist() for i in concept_id])

        return {'concept_id': concept_id, 'concept_vectors': concept_vectors}


if __name__ == '__main__':

    dataset = 'apy'
    mode = ['train_val', 'test_seen', 'test_unseen']

    # dataset test
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    # word2vec
    weight_path = PJ('.', 'dataset', 'google_news', 'GoogleNews-vectors-gensim-normed.bin')
    word2vec = KeyedVectors.load(weight_path, mmap='r')
    word2vec.vectors_norm = word2vec.vectors

    labelset = LabelSet('apy', word2vec)

    trainval = ClassDataset(dataset='apy', mode='train_val',
                            img_transform=train_transform,
                            label_transform=labelset.train)

    trainval_loader = torch.utils.data.DataLoader(trainval, batch_size=3, shuffle=False)

    # show test data
    for i, batch_data in enumerate(trainval_loader, 1):

        images = batch_data['image']
        labels = batch_data['label']

    #     for img in images:
    #         transforms.ToPILImage()(img).show()
    #     break
