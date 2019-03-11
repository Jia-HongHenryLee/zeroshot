import torch

from data_loader import ClassDatasets, ConceptSets
from torch.utils.data import DataLoader

import torch.optim as optim
from model import RESNET, model_epoch

from utils import cal_acc

import numpy as np
from os.path import join as PJ
import yaml
from tensorboardX import SummaryWriter

global DEVICE
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':

    # setting
    CONFIG = yaml.load(open("train_test_config.yaml"))

    EXP_NAME = CONFIG['exp_name']

    DATASET = CONFIG['dataset']
    CONCEPTS = CONFIG['concepts']

    SAVE_PATH = PJ('.', 'test_runs', DATASET, EXP_NAME)
    LOAD_MODEL = None
    # LOAD_MODEL = PJ(SAVE_PATH, 'epoch' + str(CONFIG['start_epoch'] - 1) + '.pkl')

    L_RATE = np.float64(CONFIG['l_rate'])

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build model
    if LOAD_MODEL is None:
        model = RESNET(freeze=CONFIG['freeze'], pretrained=True, k=CONFIG['k'], d=CONFIG['d'])
        model = model.to(DEVICE)

    else:
        print("Loading pretrained model")
        model = RESNET(freeze=True, pretrained=False, k=CONFIG['k'], d=CONFIG['d'])
        model.load_state_dict(torch.load(LOAD_MODEL))
        model = model.to(DEVICE)

    # state
    STATE = {
        'dataset': DATASET,
        'mode': 'train_test',
        'split_list': ['trainval', 'test_seen', 'test_unseen']
    }

    # data setting
    concepts = ConceptSets(STATE, CONCEPTS)

    datasets = ClassDatasets(STATE)

    train_loader = DataLoader(datasets['trainval'], batch_size=CONFIG['train_batch_size'], shuffle=True)
    test_loaders = {tn: DataLoader(datasets[tn], batch_size=CONFIG['test_batch_size'], shuffle=False)
                    for tn in STATE['split_list'][1:]}

    ##########################################################################################

    writer = SummaryWriter(PJ(SAVE_PATH))

    # optim setting
    params = model.classifier.parameters() if CONFIG['freeze'] else model.parameters()
    optimizer = optim.SGD(params, L_RATE, momentum=CONFIG['momentum'])

    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1, 8], gamma=0.1)

    for epoch in range(CONFIG['start_epoch'], CONFIG['end_epoch']):

        scheduler.step()

        # training
        train_metrics = model_epoch(loss_name="trainval", mode="train", epoch=epoch,
                                    model=model, k=CONFIG['k'], d=CONFIG['d'],
                                    data_loader=train_loader, concepts=concepts,
                                    optimizer=optimizer, writer=writer)

        torch.save(model.state_dict(), PJ(SAVE_PATH, 'epoch' + str(epoch) + '.pkl'))

        train_class, train_acc = cal_acc(train_metrics)
        writer.add_scalar('trainval_acc', train_acc * 100, epoch)

        ######################################################################################

        # test
        record = {tn: {'acc': 0.0, 'class': None} for tn in STATE['split_list'][1:]}
        record.update({tn + '_g': {'acc': 0.0, 'class': None} for tn in STATE['split_list'][1:]})

        for tn in STATE['split_list'][1:]:
            test_metric = model_epoch(mode="test", epoch=epoch, loss_name=tn,
                                      model=model, k=CONFIG['k'], d=CONFIG['d'],
                                      data_loader=test_loaders[tn], concepts=concepts,
                                      optimizer=optimizer, writer=writer)

            test_class, test_acc = cal_acc(test_metric)
            record[tn]['acc'] = test_acc
            record[tn]['class'] = test_class

            test_g_class, test_g_acc = cal_acc(test_metric, general=True)
            record[tn + '_g']['acc'] = test_g_acc
            record[tn + '_g']['class'] = test_g_class

            writer.add_scalar(tn + '_acc', test_acc * 100, epoch)
            writer.add_scalar(tn + '_g_acc', test_g_acc * 100, epoch)

        # per-class acc
        pc_class = record['test_seen']['class'].copy()
        pc_class.update(record['test_unseen']['class'])
        pc_acc = sum(list(pc_class.values())) / len(pc_class)
        writer.add_scalar('Acc_per_class', pc_acc * 100, epoch)

        # gzsl acc: H acc
        H_acc = 2 * record['test_seen_g']['acc'] * record['test_unseen_g']['acc'] / \
            (record['test_seen_g']['acc'] + record['test_unseen_g']['acc'] + 1e-10)
        writer.add_scalar('H_acc', H_acc * 100, epoch)

        ######################################################################################

        with open(PJ(SAVE_PATH, "val_table.txt"), "a+") as f:
            table = yaml.dump({str(epoch): record}, f)
