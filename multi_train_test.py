import torch

from data_loader import ClassDatasets, ConceptSets
from torch.utils.data import DataLoader

import torch.optim as optim
from model import visual_semantic_model, model_epoch

import utils

import numpy as np
from os.path import join as PJ
import yaml
import json
from tensorboardX import SummaryWriter

global DEVICE
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':

    # setting
    CONFIG = yaml.load(open("multi_train_test_config.yaml"))

    EXP_NAME = CONFIG['exp_name']

    DATASET = CONFIG['dataset']
    CONCEPTS = CONFIG['concepts']

    SAVE_PATH = PJ('.', 'runs_multi', DATASET, EXP_NAME)
    LOAD_MODEL = None
    # LOAD_MODEL = PJ('.', 'runs_multi', DATASET, 'traintest_two_layer', 'epoch10.pkl')
    # LOAD_MODEL = PJ(SAVE_PATH, 'epoch' + str(CONFIG['start_epoch'] - 1) + '.pkl')
    # LOAD_MODEL = PJ(SAVE_PATH, 'epoch1.pkl')

    L_RATE = np.float64(CONFIG['l_rate'])

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build model
    if LOAD_MODEL is None:
        model = visual_semantic_model(pretrained=CONFIG['pretrained'], freeze=CONFIG['freeze'], k=CONFIG['k'], d=CONFIG['d'])
        model = model.to(DEVICE)

    else:
        print("Loading pretrained model")
        model = visual_semantic_model(pretrained=CONFIG['pretrained'], freeze=CONFIG['freeze'], k=CONFIG['k'], d=CONFIG['d'])
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
    concepts = ConceptSets(STATE, CONCEPTS)

    datasets = ClassDatasets(STATE)

    train_loader = DataLoader(datasets['train'], batch_size=CONFIG['train_batch_size'], shuffle=True)
    test_loader = DataLoader(datasets['test'], batch_size=CONFIG['test_batch_size'], shuffle=False)

    CONFIG['loss_args']['alpha'] = CONFIG['loss_args']['alpha'] * datasets['train'].neg_weight

    ##########################################################################################

    writer = SummaryWriter(PJ(SAVE_PATH))

    # optim setting
    if CONFIG['freeze']:
        params = [
            {'params': model.transform.parameters(), 'lr': L_RATE}
        ]
    else:
        params = [
            {'params': model.parameters(), 'lr': L_RATE}
        ]

    if CONFIG['optim'] == 'SGD':
        optimizer = optim.SGD(params, L_RATE, momentum=CONFIG['momentum'])

    elif CONFIG['optim'] == 'Adam':
        optimizer = optim.Adam(params, L_RATE)

    # lr_list = [1, 2, 5, 10, 500, 1000, 5000, 10000, 50000, 100000]
    # scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: lr_list[epoch])
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9999)

    for epoch in range(CONFIG['start_epoch'], CONFIG['end_epoch']):

        # training
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        train_metric = model_epoch(loss_name="train", mode="multi", epoch=epoch,
                                   model=model, loss_args=CONFIG['loss_args'],
                                   data_loader=train_loader, concepts=concepts,
                                   optimizer=optimizer, writer=writer)

        torch.save(model.state_dict(), PJ(SAVE_PATH, 'epoch' + str(epoch) + '.pkl'))

        for g in [False, True]:
            record_name = 'train_g' if g else 'train'
            train_map = utils.cal_map(train_metric, general=g)
            train_miap = utils.cal_miap(train_metric, g)
            train_prf1 = utils.cal_prf1(train_metric, general=g)
            writer.add_scalar(record_name + '_mAP', (sum(train_map) / len(train_map) * 100), epoch)
            writer.add_scalar(record_name + '_miap', train_miap * 100, epoch)
            writer.add_scalar(record_name + '_of', train_prf1['o_f1'] * 100, epoch)

        # scheduler.step()

        ######################################################################################

        # test
        record = {'test': {'miap': 0.0, 'prf1': None},
                  'test_g': {'miap': 0.0, 'prf1': None}}

        test_metric = model_epoch(mode="multi", epoch=epoch, loss_name='test',
                                  model=model, loss_args=CONFIG['loss_args'],
                                  data_loader=test_loader, concepts=concepts,
                                  optimizer=optimizer, writer=writer)

        for g in [False, True]:
            test_map = utils.cal_map(test_metric, general=g)
            test_miap = utils.cal_miap(test_metric, general=g)
            test_prf1 = utils.cal_prf1(test_metric, general=g)
            record_name = 'test_g' if g else 'test'
            record[record_name]['maps'] = test_map
            record[record_name]['miap'] = test_miap.item()
            record[record_name]['prf1'] = {k: v.item() for k, v in test_prf1.items()}

            writer.add_scalar(record_name + '_miap', record[record_name]['miap'], epoch)
            writer.add_scalar(record_name + '_mAP', (sum(test_map) / len(test_map) * 100), epoch)
            writer.add_scalar(record_name + '_of', record[record_name]['prf1']['o_f1'] * 100, epoch)

        text = utils.write_table({'conv': record['test']['prf1'], 'general': record['test_g']['prf1']})
        writer.add_text('Test Table', text, epoch)

        ######################################################################################
        test_metric = {n: np.asarray(test_metric[n]).tolist() for n in test_metric}
        with open(PJ(SAVE_PATH, "test_table_" + str(epoch) + ".txt"), "w") as f:
            table = json.dump(test_metric, f)
