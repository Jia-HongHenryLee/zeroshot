import torch
from os.path import join as PJ

from data_loader import ClassDatasets, ConceptSets
from torch.utils.data import DataLoader

from model import RESNET
import matplotlib.pyplot as plt
import numpy as np

global DEVICE
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET = 'nus_wide'
EXP_NAME = 'traintest_sgd_fast0tag'

k = 1
d = 300

SAVE_PATH = PJ('.', 'runs_multi', DATASET, EXP_NAME)
LOAD_MODEL = PJ(SAVE_PATH, 'epoch2.pkl')

print("Loading pretrained model")
model = RESNET(freeze=True, pretrained=False, k=k, d=d)
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

        batch_img = batch_data['image'].to(DEVICE)
        batch_label = batch_data['label'].to(DEVICE)

        outputs = model(torch.autograd.Variable(batch_img)).view(-1, k, d)

        # predict
        concept_vectors = concepts['test']['concept_vector'].t()
        concept_vectors = concept_vectors.expand(outputs.shape[0], d, -1)
        p_ranks = torch.bmm(outputs, concept_vectors).norm(p=2, dim=1)

        for p_rank, gts in zip(p_ranks, batch_label):
            if sum(gts) == 0:
                continue
            gts = gts[concepts['test']['concept_label']]
            p = torch.sort(p_rank)[1]
            print(p)
            print((gts == 1).nonzero())

        # general
        concept_vectors_g = concepts['general']['concept_vector'].t()
        concept_vectors_g = concept_vectors_g.expand(outputs.shape[0], d, -1)
        g_ranks = torch.bmm(outputs, concept_vectors_g).norm(p=2, dim=1)

        for g_rank, g_gts in zip(g_ranks, batch_label):
            gp = torch.sort(g_rank)[1].cpu().numpy()
            gt = (g_gts == 1).nonzero().cpu().numpy()
            print(gp)
            print(gt)
            print(sorted([np.nonzero((gp == g))[0] + 1 for g in gt]))

        show = batch_img[0].view(batch_img[0].shape[1], batch_img[0].shape[2], batch_img[0].shape[0])
        plt.imshow(show)
        plt.show()

        break
