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
    CONFIG = yaml.load(open("single_train_test_config.yaml"))

    EXP_NAME = CONFIG['exp_name']

    DATASET = CONFIG['dataset']
    CONCEPTS = CONFIG['concepts']

    SAVE_PATH = PJ('.', 'runs_test', DATASET, EXP_NAME)
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
        'split_list': ['trainval', 'test_seen', 'test_unseen']
    }

    # data setting
    print("load data")
    concepts = ConceptSets(STATE, CONCEPTS)

    datasets = ClassDatasets(STATE)

    train_loader = DataLoader(datasets['trainval'], batch_size=CONFIG['train_batch_size'], shuffle=True)
    test_loaders = {tn: DataLoader(datasets[tn], batch_size=CONFIG['test_batch_size'], shuffle=False)
                    for tn in STATE['split_list'][1:]}

    CONFIG['loss_args']['alpha'] = CONFIG['loss_args']['alpha'] * datasets['trainval'].neg_weight
    CONFIG['loss_args']['class_weight'] = datasets['trainval'].class_weight[concepts['trainval']['concept_label']]

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
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=2)

    for epoch in range(CONFIG['start_epoch'], CONFIG['end_epoch']):

        # training
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        #  scheduler.step()quantity of data

        train_metrics = model_epoch(loss_name="trainval", mode="single", epoch=epoch,
                                    model=model, loss_args=CONFIG['loss_args'],
                                    data_loader=train_loader, concepts=concepts,
                                    optimizer=optimizer, writer=writer)

        torch.save(model.state_dict(), PJ(SAVE_PATH, 'epoch' + str(epoch) + '.pkl'))

        for g in [False, True]:
            record_name = 'train_g' if g else 'train'
            train_class, train_acc = utils.cal_acc(train_metrics, g)
            writer.add_scalar(record_name + '_acc', train_acc * 100, epoch)

        ######################################################################################

        # test
        record = {tn: {'acc': 0.0, 'class': None} for tn in STATE['split_list'][1:]}
        record.update({tn + '_g': {'acc': 0.0, 'class': None} for tn in STATE['split_list'][1:]})

        for tn in STATE['split_list'][1:]:

            test_metric = model_epoch(mode="single", epoch=epoch, loss_name=tn,
                                      model=model, loss_args=CONFIG['loss_args'],
                                      data_loader=test_loaders[tn], concepts=concepts,
                                      optimizer=optimizer, writer=writer)

            for g in [False, True]:
                test_class, test_acc = utils.cal_acc(test_metric, g)
                record_name = tn + '_g' if g else tn
                record[record_name]['acc'] = test_acc
                record[record_name]['class'] = test_class

                writer.add_scalar(record_name + '_acc', test_acc * 100, epoch)

                with open(PJ(SAVE_PATH, "test_table_" + str(epoch) + ".txt"), "a+") as f:
                    table = json.dump({record_name: test_class}, f)

        writer.add_scalar('conv_acc', 100 * utils.cal_h_acc(record, False), epoch)
        writer.add_scalar('H_acc', 100 * utils.cal_h_acc(record, True), epoch)
