import torch
from os.path import join as PJ

from data_loader import ClassDatasets, ConceptSets
from torch.utils.data import DataLoader

from model import visual_semantic_model
import matplotlib.pyplot as plt
import numpy as np
import json
import yaml
import cv2
import utils
import pandas as pd

from collections import deque
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from inspect import signature

global DEVICE
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET = 'apy'
CONCEPT_VEC = 'new'
EXP_NAME = 'nonlinear_softmax_noalpha_nonormal_w2v'
EPOCH = 12

PRINT = False
HIT = False

CONCEPT = PJ('.', 'dataset', DATASET, 'list', 'concepts', 'concepts.txt')
SPLIT = PJ('.', 'dataset', DATASET, 'list', 'train_test', 'id_split.txt')
with open(CONCEPT) as f:
    CONCEPT = np.array([l.strip() for l in f.readlines()])
SPLIT = yaml.load(open(SPLIT))
SPLIT = {
    'test_seen': SPLIT['train_id'],
    'test_unseen': SPLIT['test_id']
}

k = 100
d = 300
batch_size = 512

LOAD_MODEL = PJ('.', 'runs_test', DATASET, EXP_NAME, 'epoch' + str(EPOCH) + '.pkl')

step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})

print("Loading pretrained model")
model = visual_semantic_model(pretrained=False, k=k, d=d)
model.load_state_dict(torch.load(LOAD_MODEL))
for param in model.parameters():
        param.requires_grad = False
model = model.to(DEVICE)

# state
STATE = {
    'dataset': DATASET,
    'mode': 'train_test',
    'split_list': ['trainval', 'test_seen', 'test_unseen']
}

# data setting
print("load data")
concepts = ConceptSets(STATE, CONCEPT_VEC)
# print(concepts['trainval']['concept_label'])
# print(concepts['test_seen']['concept_label'])
# print(concepts['test_unseen']['concept_label'])
# print(concepts['general']['concept_label'])
# aaa
datasets = ClassDatasets(STATE)

model.eval()
print("test")

# for concept in ['test_seen', 'test_unseen']:
for concept in ['test_unseen']:

    records = {
        'total': deque(),
        'total_g': deque(),
        'correct': deque(),
        'correct_g': deque(),
        'gt_g': deque(),
        'rank_g': deque()
    }

    data_loader = DataLoader(datasets[concept], batch_size=batch_size, shuffle=False)

    for batch_i, batch_data in enumerate(data_loader, 1):

        batch_img = batch_data['image']
        batch_label = batch_data['label'].to(DEVICE)

        gt = batch_label[:, concepts[concept]['concept_label']][..., None]

        if torch.sum(gt) != 0:

            # gt = gt.reshape(batch_label.shape[0], -1).tolist()
            records['total'].extend(gt.tolist())

            concept_vectors = concepts[concept]['concept_vector']

            outputs = model(torch.autograd.Variable(batch_img.to(DEVICE)), concept_vectors)
            # print(model.matrix.reshape(-1))
            # print(model.bias)

            maxs = torch.max(outputs, 1)[1][..., None]
            maxs_onehot = torch.zeros(outputs.shape).cuda().scatter_(1, maxs, 1)

            if PRINT:
                sort_ = torch.sort(-torch.squeeze(outputs))[1].tolist()
                print(CONCEPT[concepts[concept]['concept_label']][torch.max(gt, 1)[1].tolist()])
                print(CONCEPT[concepts[concept]['concept_label']][sort_])
                print(torch.sort(-torch.squeeze(outputs))[0].tolist())

            # probs = torch.squeeze(outputs).reshape(batch_label.shape[0], -1).tolist()
            records['correct'].extend(maxs_onehot.tolist())

        gt_g = batch_label[:, concepts['general']['concept_label']][..., None]
        # gt_g = gt_g.reshape(batch_label.shape[0], -1).tolist()
        records['total_g'].extend(gt_g.tolist())
        records['gt_g'].append(torch.nonzero(gt_g.reshape(-1)).tolist()[0])
        # print(torch.nonzero(gt_g.reshape(-1)))

        concept_vectors_g = concepts['general']['concept_vector']
        outputs_g = model(torch.autograd.Variable(batch_img.to(DEVICE)), concept_vectors_g)

        maxs_g = torch.max(outputs_g, 1)[1][..., None]
        maxs_g_onehot = torch.zeros(outputs_g.shape).cuda().scatter_(1, maxs_g, 1)
        # probs_g = torch.squeeze(outputs_g).reshape(batch_label.shape[0], -1).tolist()
        records['correct_g'].extend(maxs_g_onehot.tolist())
        records['rank_g'].append(torch.sort(-outputs_g.reshape(-1))[1].tolist())

        if PRINT:
            sort_g = torch.sort(-torch.squeeze(outputs_g))[1].tolist()
            print(CONCEPT[concepts['general']['concept_label']][sort_g])
            print(torch.sort(-torch.squeeze(outputs_g))[0].tolist())
            print()
            plt.imshow(batch_img[0].permute(1, 2, 0))
            plt.show()

        # print(torch.round(torch.squeeze(outputs)).reshape(-1).nonzero())
        # print(torch.sort(-torch.squeeze(outputs))[1].reshape(-1))

        # show = batch_img[0].view(batch_img[0].shape[1], batch_img[0].shape[2], batch_img[0].shape[0])
        # plt.imshow(show)
        # plt.show()

        if batch_i == 100 and PRINT:
            aaa
            break

    if HIT:
        hit_unseen = {CONCEPT[c]: [] for c in concepts['test_unseen']['concept_label']}
        for rank, gt in zip(records['rank_g'], records['gt_g']):
            for r in concepts['test_seen']['concept_label']:
                rank.remove(r)
            hit_unseen[CONCEPT[gt[0]]].append(rank.index(gt[0]) + 1)

        hit_unseen_acc = {h: sum(hit_unseen[h]) / len(hit_unseen[h]) if len(hit_unseen[h]) > 0 else 0 for h in hit_unseen}
        print(hit_unseen_acc)

        with open(PJ('.', 'analysis', 'hit_unseen_remove.json'), 'w') as f:
            json.dump(hit_unseen, f)

    print(utils.cal_acc(records, False))
    print(utils.cal_acc(records, True))

    total_num = np.squeeze(np.where(np.array(records['total']).sum(axis=0) > 0))
    correct_num = np.squeeze(np.where(np.array(records['correct']).sum(axis=0) > 0))
    total_g_num = np.squeeze(np.where(np.array(records['total_g']).sum(axis=0) > 0))
    correct_g_num = np.squeeze(np.where(np.array(records['correct_g']).sum(axis=0) > 0))

    ind = np.union1d(total_num, correct_num)
    ind_g = np.union1d(total_g_num, correct_g_num)

    matrix = confusion_matrix(np.squeeze(np.array(records['total'])).astype(int).argmax(axis=1), np.squeeze(np.array(records['correct'])).astype(int).argmax(axis=1))
    report = classification_report(np.squeeze(np.array(records['total'])).astype(int).argmax(axis=1), np.squeeze(np.array(records['correct'])).astype(int).argmax(axis=1), labels=list(range(20)))
    print(report)

    matrix_g = confusion_matrix(np.squeeze(np.array(records['total_g'])).astype(int).argmax(axis=1), np.squeeze(np.array(records['correct_g'])).astype(int).argmax(axis=1))
    conf = pd.DataFrame(matrix, columns=CONCEPT[concepts[concept]['concept_label']][ind])
    conf.set_index(CONCEPT[concepts[concept]['concept_label']][ind], inplace=True)
    conf_g = pd.DataFrame(matrix_g, columns=CONCEPT[concepts['general']['concept_label']][ind_g])
    conf_g.set_index(CONCEPT[concepts['general']['concept_label']][ind_g], inplace=True)

    conf = (conf.div(conf.sum(axis=1), axis=0)).fillna(0).round(4) * 100
    conf_g = (conf_g.div(conf_g.sum(axis=1), axis=0)).fillna(0).round(4) * 100

    conf.to_csv(PJ('.', 'analysis', EXP_NAME + "_" + DATASET + "_" + concept + "_" + str(EPOCH) + "_confusion_matrix.csv"))
    conf_g.to_csv(PJ('.', 'analysis', EXP_NAME + "_" + DATASET + "_" + concept + "_" + str(EPOCH) + "_confusion_matrix_g.csv"))
