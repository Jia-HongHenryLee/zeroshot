import torch
import numpy as np
import random

torch.manual_seed(64)
torch.cuda.manual_seed(64)
np.random.seed(64)
random.seed(64)

torch.backends.cudnn.deterministic = True

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
    CONFIG = yaml.load(open("train_val_config.yaml"))

    EXP_NAME = CONFIG['exp_name']

    DATASET = CONFIG['dataset']
    CONCEPTS = CONFIG['concepts']

    SAVE_PATH = PJ('.', 'runs', DATASET, EXP_NAME)

    LOAD_MODEL = None
    # LOAD_MODEL = PJ(SAVE_PATH, '0_epoch7.pkl')

    L_RATE = np.float64(CONFIG['l_rate'])

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = None

    # state
    STATE = {
        'dataset': DATASET,
        'mode': 'train_val',
        'split_list': ['train', 'val']
    }

    ##########################################################################################

    # random validation for three times
    for val_times in range(3):

        # build model
        if LOAD_MODEL is None:
            model = RESNET(freeze=True, pretrained=True, k=CONFIG['k'], d=CONFIG['d'])
            model = model.to(DEVICE)

        else:
            print("Loading pretrained model")
            model = RESNET(freeze=False, pretrained=False, k=CONFIG['k'], d=CONFIG['d'])
            model.load_state_dict(torch.load(LOAD_MODEL))
            model = model.to(DEVICE)

        # data setting
        concepts = ConceptSets(STATE, CONCEPTS)

        datasets = ClassDatasets(STATE)

        train_loader = DataLoader(datasets['train'], batch_size=CONFIG['train_batch_size'], shuffle=True)
        val_loader = DataLoader(datasets['val'], batch_size=CONFIG['val_batch_size'], shuffle=False)

        ##########################################################################################

        # optim setting

        optimizer = optim.SGD(model.classifier.parameters(), lr=L_RATE, momentum=CONFIG['momentum'])

        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

        for epoch in range(CONFIG['start_epoch'], CONFIG['end_epoch']):

            writer = SummaryWriter(PJ(SAVE_PATH, 'val' + str(val_times)))

            scheduler.step()

            # training
            train_metrics = model_epoch(loss_name="train", mode="train", epoch=epoch,
                                        model=model, k=CONFIG['k'], d=CONFIG['d'],
                                        data_loader=train_loader, concepts=concepts,
                                        optimizer=optimizer, writer=writer)

            torch.save(model.state_dict(), PJ(SAVE_PATH, str(val_times) + '_epoch' + str(epoch) + '.pkl'))

            train_class, train_acc = cal_acc(train_metrics, concepts['train']['concept_label'])
            writer.add_scalar('train_acc', train_acc * 100, epoch)

            ######################################################################################

            # val
            val_metric = model_epoch(mode="test", epoch=epoch, loss_name="val",
                                     model=model, k=CONFIG['k'], d=CONFIG['d'],
                                     data_loader=val_loader, concepts=concepts,
                                     optimizer=optimizer, writer=writer)

            val_class, val_acc = cal_acc(val_metric)
            val_g_class, val_g_acc = cal_acc(val_metric, general=True)

            writer.add_scalar('val_acc', val_acc * 100, epoch)
            writer.add_scalar('val_g_acc', val_g_acc * 100, epoch)

            ######################################################################################

            with open(PJ(SAVE_PATH, "val_table.txt"), "a+") as f:
                table = yaml.dump({str(epoch): {
                    'val': {'acc': val_acc, 'class': val_class},
                    'val_g': {'acc': val_g_acc, 'class': val_g_class}
                }}, f)

            writer.close()
