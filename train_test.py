import torch

from data_loader import ClassDatasets, ConceptSets
from torch.utils.data import DataLoader

import torch.optim as optim
from model import RESNET_nonlinear, model_epoch

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
    CONFIG = yaml.load(open("train_test_config.yaml"))

    EXP_NAME = CONFIG['exp_name']

    DATASET = CONFIG['dataset']
    CONCEPTS = CONFIG['concepts']

    SAVE_PATH = PJ('.', 'runs_multi', DATASET, EXP_NAME)
    LOAD_MODEL = None
    # LOAD_MODEL = PJ(SAVE_PATH, 'epoch' + str(CONFIG['start_epoch'] - 1) + '.pkl')

    L_RATE = np.float64(CONFIG['l_rate'])

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build model
    if LOAD_MODEL is None:
        model = RESNET_nonlinear(freeze=CONFIG['freeze'], pretrained=True,
                                 k1=CONFIG['k1'], k2=CONFIG['k2'], d=CONFIG['d'])
        model = model.to(DEVICE)

    else:
        print("Loading pretrained model")
        model = RESNET_nonlinear(freeze=True, pretrained=False, k=CONFIG['k'], d=CONFIG['d'])
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

    train_loader = DataLoader(datasets['train'], batch_size=CONFIG['train_batch_size'], shuffle=True, num_workers=4)
    test_loader = DataLoader(datasets['test'], batch_size=CONFIG['test_batch_size'], shuffle=False, num_workers=4)

    ##########################################################################################

    writer = SummaryWriter(PJ(SAVE_PATH))

    # optim setting
    params = model.transform.parameters() if CONFIG['freeze'] else model.parameters()

    if CONFIG['optim'] == 'SGD':
        optimizer = optim.SGD(params, L_RATE, momentum=CONFIG['momentum'])

    elif CONFIG['optim'] == 'Adam':
        optimizer = optim.Adam(params, L_RATE)

    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    for epoch in range(CONFIG['start_epoch'], CONFIG['end_epoch']):

        # training
        train_metrics = model_epoch(loss_name="train", mode="train", epoch=epoch,
                                    model=model, sample_rate=CONFIG['sample'],
                                    data_loader=train_loader, concepts=concepts,
                                    optimizer=optimizer, writer=writer)

        torch.save(model.state_dict(), PJ(SAVE_PATH, 'epoch' + str(epoch) + '.pkl'))

        for g in [False, True]:
            record_name = 'train_g' if g else 'train'
            train_miap = utils.cal_miap(train_metrics, g)
            writer.add_scalar(record_name + '_miap', train_miap * 100, epoch)

        scheduler.step()

        ######################################################################################

        # test
        record = {'test': {'miap': 0.0, 'top3_prf1': None, 'top10_prf1': None},
                  'test_g': {'miap': 0.0, 'top3_prf1': None, 'top10_prf1': None}}

        test_metric = model_epoch(mode="test", epoch=epoch, loss_name='test',
                                  model=model, sample_rate=CONFIG['sample'],
                                  data_loader=test_loader, concepts=concepts,
                                  optimizer=optimizer, writer=writer)

        for g in [False, True]:
            test_miap = utils.cal_miap(test_metric, general=g)
            test_top3_prf1, test_top10_prf1 = utils.cal_top(test_metric, general=g)
            record_name = 'test_g' if g else 'test'
            record[record_name]['miap'] = test_miap.item()
            record[record_name]['top3_prf1'] = {k: v.item() for k, v in test_top3_prf1.items()}
            record[record_name]['top10_prf1'] = {k: v.item() for k, v in test_top10_prf1.items()}

        text = '| conv | general|\n| :----: | :----: |\n' + \
            '{:.2f}'.format(record['test']['miap'] * 100) + ' | ' + \
            '{:.2f}'.format(record['test_g']['miap'] * 100) + ' | ' + '\n'
        writer.add_text('miap', text, epoch)
        writer.add_scalar('test_miap', utils.cal_miap(test_metric), epoch)
        writer.add_scalar('test_g_miap', utils.cal_miap(test_metric, True), epoch)

        text = utils.write_table({'top3': record['test']['top3_prf1'], 'top10': record['test_g']['top10_prf1']})
        writer.add_text('Test Table', text, epoch)
        writer.add_scalar('top3_of', record['test']['top3_prf1']['o_f1'] * 100, epoch)
        writer.add_scalar('top10_of', record['test_g']['top10_prf1']['o_f1'] * 100, epoch)

        ######################################################################################
        with open(PJ(SAVE_PATH, "test_table.txt"), "a+") as f:
            table = json.dump({str(epoch): record}, f)
