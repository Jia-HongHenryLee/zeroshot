import torch
from os.path import join as PJ

from data_loader import ClassDatasets, ConceptSets
from torch.utils.data import DataLoader

from model import visual_semantic_model
import matplotlib.pyplot as plt
import numpy as np
import json

global DEVICE
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET = 'nus_wide'
EXP_NAME = 'traintest_adaptive'

k = 1
d = 300

LOAD_MODEL = PJ('.', 'runs_multi', DATASET, 'traintest_adaptive_twolayer_adaptive_weight_non_linear_Adam', 'test_table_1.txt')

with open(LOAD_MODEL) as f:
    data = json.load(f)
print(data['predicts_zsl'][2])
print(data['gts_zsl'][2])
aaaa

print("Loading pretrained model")
model = visual_semantic_model(pretrained=False, k=k, d=d)
model.load_state_dict(torch.load(LOAD_MODEL))
model = model.to(DEVICE)

# state
STATE = {
    'dataset': DATASET,
    'mode': 'train_test',
    'split_list': ['train', 'test']
}

# data setting
print("load data")
concepts = ConceptSets(STATE, "vec")
datasets = ClassDatasets(STATE)
data_loader = DataLoader(datasets['test'], batch_size=1, shuffle=False)

model.eval()

with torch.no_grad():

    for batch_i, batch_data in enumerate(data_loader, 1):

        batch_img = batch_data['image']
        batch_label = batch_data['label'].to(DEVICE)

        gt = batch_label[:, concepts['test']['concept_label']][..., None]

        if torch.sum(gt) == 0:
            continue

        print(torch.nonzero(gt.reshape(-1)))

        concept_vectors = concepts['test']['concept_vector']
        outputs = model(torch.autograd.Variable(batch_img.to(DEVICE)), concept_vectors)
        print(torch.sigmoid(outputs).reshape(-1))
        print(torch.round(torch.sigmoid(outputs)))
        aaa
        print(torch.round(torch.sigmoid(outputs).reshape(-1)).nonzero().reshape(-1))

        show = batch_img[0].view(batch_img[0].shape[1], batch_img[0].shape[2], batch_img[0].shape[0])
        plt.imshow(show)
        plt.show()
