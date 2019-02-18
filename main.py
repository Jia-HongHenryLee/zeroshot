import torch
import torch.optim as optim
from torch.autograd import Variable
import torchvision.transforms as transforms
import model
from tensorboardX import SummaryWriter

import numpy as np
from data_loader import ClassDataset, LabelSet
from torch.utils.data import DataLoader

from os.path import join as PJ
import json
import yaml

global DEVICE


class HingeRankLoss():

    def __init__(self, label_set, k=100, d=300):
        self.concept_len = len(label_set['concept_id'])
        self.concept_vectors = label_set['concept_vectors'].to(DEVICE)
        self.k = k
        self.d = d

    def _loss(self, outputs=None, labels=None, **kwargs):
        self.outputs = outputs
        self.labels = labels
        self.focal = kwargs

        self.loss = Variable(torch.cuda.FloatTensor([0.0]), requires_grad=True)

        for output, label in zip(self.outputs, self.labels):

            pos_inds = label
            neg_inds = torch.cuda.LongTensor([i for i in range(self.concept_len) if i != label.data[0]])

            p_vectors = torch.stack([self.concept_vectors[p] for p in pos_inds]).t()
            n_vectors = torch.stack([self.concept_vectors[n] for n in neg_inds]).t()

            p_transformeds = torch.mm(output, p_vectors).norm(p=2, dim=0)
            n_transformeds = torch.mm(output, n_vectors).norm(p=2, dim=0)

            subloss = torch.cuda.FloatTensor([1.0])

            for p_transformed in p_transformeds:
                for n_transformed in n_transformeds:
                    subloss = subloss.clone() + torch.exp(p_transformed - n_transformed)

            self.loss = self.loss.clone() + torch.log(subloss)

        return self.loss


def model_epoch(mode=None, epoch=0, model=None, data_loader=None, optimizer=None, labelset=None, writer=None):

    if mode == 'train':
        print("train")
        model.train()
        torch.set_grad_enabled(True)

    elif mode == 'test':
        print("test")
        model.eval()
        torch.set_grad_enabled(False)

    else:
        assert "Mode Error"

    concept_len = len(labelset['concept_id'])
    concept_vectors = labelset['concept_vectors'].to(DEVICE)

    metrics = {
        'total': torch.cuda.FloatTensor(concept_len).fill_(0),
        'correct': torch.cuda.FloatTensor(concept_len).fill_(0)
    }

    running_loss = 0.0

    for batch_i, batch_data in enumerate(data_loader, 1):

        if mode == 'train':
            optimizer.zero_grad()

        batch_img = batch_data['image'].to(DEVICE)
        batch_label = batch_data['label'].to(DEVICE)

        outputs = model(Variable(batch_img)).view(-1, K, D)

        if mode == 'train':
            loss = criterion._loss(outputs=outputs, labels=batch_label)
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0] / data_loader.batch_size

        label_vectors = concept_vectors.t().expand(outputs.shape[0], D, -1)
        p_rank = torch.bmm(outputs, label_vectors).norm(p=2, dim=1)
        p_label = torch.min(p_rank, dim=1)[1]

        for gt, p in zip(batch_label.reshape(-1), p_label):
            if gt == p:
                metrics['correct'][p] += 1
            metrics['total'][gt] += 1

        if batch_i % 10 == 0:
            if mode == 'train':
                tmp_loss = running_loss / 10
                writer.add_scalar('train_loss', tmp_loss, batch_i + (epoch - 1) * len(data_loader))
                print('[%d, %6d] loss: %.3f' % (epoch, batch_i * data_loader.batch_size, tmp_loss))
                running_loss = 0.0
            else:
                print('batch:', batch_i * data_loader.batch_size)

    return metrics


if __name__ == '__main__':

    # setting
    CONFIG = yaml.load(open("config.yaml"))

    MODEL = CONFIG['MODEL']
    DATASET = CONFIG['DATASET']
    EXP_NAME = CONFIG['EXP_NAME']
    MODE = CONFIG['MODE']

    SAVE_PATH = PJ('.', 'runs', DATASET, EXP_NAME)

    LOAD_MODEL = MODEL
    # LOAD_MODEL = PJ(SAVE_PATH + ['epoch4.pkl'])

    L_RATE = np.float64(CONFIG['L_RATE'])
    OPTIM = CONFIG['OPTIM']
    CONCEPTS = CONFIG['CONCEPTS']

    START_EPOCH = CONFIG['START_EPOCH']
    END_EPOCH = CONFIG['END_EPOCH']

    K = CONFIG['K']
    D = CONFIG['D']

    TRAIN_BATCH_SIZE = CONFIG['TRAIN_BATCH_SIZE']
    TEST_BATCH_SIZE = CONFIG['TEST_BATCH_SIZE']

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    # build model
    if LOAD_MODEL == "RESNET101":
        model = model.RESNET(freeze=True, pretrained=True, k=K, d=D)
        model = model.to(DEVICE)

    else:
        print("Loading pretrained model")
        model = torch.load(LOAD_MODEL)
        model = model.to(DEVICE)

    # label set
    labelset = LabelSet(DATASET, concepts=CONCEPTS)

    # dataset
    if MODE == 'train_and_val':
        print()

    elif MODE == 'trainval_and_test':

        # train_dataset
        train_dataset = ClassDataset(dataset=DATASET, mode='train_val',
                                     img_transform=train_transform, label_transform=labelset.train)

        # train_loader
        train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)

        # test_dataset

        test_dataset_names = ['test_seen', 'test_unseen', 'test_unseen_all']
        label_transforms = [labelset.all, labelset.test, labelset.all]

        test_datasets = {tn: ClassDataset(dataset=DATASET, mode='_'.join(tn.split('_')[:2]),
                                          img_transform=test_transform, label_transform=l)
                         for tn, l in zip(test_dataset_names, label_transforms)}

        # test_loader
        test_loaders = {tn: DataLoader(test_datasets[tn], batch_size=TEST_BATCH_SIZE, shuffle=False)
                        for tn in test_dataset_names}

    writer = SummaryWriter(SAVE_PATH)

    for epoch in range(START_EPOCH, END_EPOCH):

        # training
        criterion = HingeRankLoss(labelset.train, k=K)

        if OPTIM == "Adam":
            optimizer = optim.Adam(model.classifier.parameters(), lr=L_RATE)

        elif OPTIM == "SGD":
            optimizer = optim.SGD(model.classifier.parameters(), lr=L_RATE, momentum=CONFIG['MOMENTUM'])

        train_metrics = model_epoch(mode="train", epoch=epoch, model=model,
                                    data_loader=train_loader, labelset=labelset.train,
                                    optimizer=optimizer, writer=writer)

        torch.save(model, PJ(SAVE_PATH, 'epoch' + str(epoch) + '.pkl'))

        train_class = train_metrics['correct'] / (train_metrics['total'] + 1e-10).data
        train_acc = torch.sum(train_class) / len(labelset.train['concept_id'])
        writer.add_scalar('train_acc', train_acc * 100, epoch)

        # if epoch % 10 == 0:
        #     L_RATE = L_RATE * np.float64(0.5)
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = L_RATE
        #     writer.add_scalar('learning_rate', L_RATE, batch_i + (epoch - 1) * len(trainloader))

        # test
        test_accs = {}
        test_classes = {}
        test_sizes = [len(labelset.train['concept_id']),
                      len(labelset.test['concept_id']),
                      len(labelset.test['concept_id'])]

        for tn, l, ts in zip(test_dataset_names, label_transforms, test_sizes):
            metric = model_epoch(mode="test", epoch=epoch, model=model,
                                 data_loader=test_loaders[tn],
                                 labelset=l, writer=writer)

            test_class = metric['correct'] / (metric['total'] + 1e-10).data
            test_acc = torch.sum(test_class) / ts
            writer.add_scalar(tn + '_acc', test_acc * 100, epoch)

            test_accs[tn] = test_acc
            test_classes[tn] = test_class.cpu().numpy().tolist()

        H = 2 * test_accs['test_seen'] * test_accs['test_unseen_all'] / \
            (test_accs['test_seen'] + test_accs['test_unseen_all'])

        writer.add_scalar('H_acc', H * 100, epoch)

        with open(PJ(SAVE_PATH, "table.txt"), "a+") as f:
            table = json.dump({str(epoch): test_classes}, f)
