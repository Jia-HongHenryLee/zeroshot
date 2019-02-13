import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import numpy as np
from data_loader import ClassDataset, LabelSet
from gensim.models import KeyedVectors

from os.path import join as PJ
import json

global DEVICE


class VGG(nn.Module):

    def __init__(self, freeze=True, pretrained=True, k=100, d=300):
        super(VGG, self).__init__()
        model = torchvision.models.vgg16_bn(pretrained=pretrained)
        self.features = nn.Sequential(*list(model.features.children()))
        if freeze:
            for param in self.features.parameters():
                param.requires_grad = False
        self.classifier = nn.Sequential(*list(model.classifier.children())[:4])
        self.classifier._modules['3'] = nn.Linear(4096, k * d)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class RESNET(nn.Module):

    def __init__(self, freeze=True, pretrained=True, k=100, d=300):
        super(RESNET, self).__init__()
        model = torchvision.models.resnet101(pretrained=pretrained)
        self.features = nn.Sequential(*list(model.children())[:-1])
        if freeze:
            for param in self.features.parameters():
                param.requires_grad = False
        self.classifier = nn.Sequential(nn.Linear(2048, k * d))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class HingeRankLoss():

    def __init__(self, label_set, k=100, d=300):
        self.concept_len = len(label_set['concept_id'])
        self.concept_vectors = label_set['concept_vectors'].to(DEVICE)
        self.k = k
        self.d = d

    def _loss(self, outputs=None, labels=None, loss_form="pairwise", **kwargs):
        self.outputs = outputs
        self.labels = labels
        self.loss_form = loss_form
        self.focal = kwargs

        self.loss = Variable(torch.cuda.FloatTensor([0.0]), requires_grad=True)

        for output, label in zip(self.outputs, self.labels):

            if self.loss_form == 'pairwise':

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

            elif self.loss_form.find('ce') > -1:

                rank_vectors = torch.mm(output, self.concept_vectors.t()).norm(p=2, dim=0)
                value_range = torch.max(rank_vectors) - torch.min(rank_vectors)
                rank_vectors = 1 - (rank_vectors - min(rank_vectors)) / value_range

                target = torch.zeros(self.concept_len)
                target[label] = 1
                target = target.type(torch.cuda.FloatTensor)

                n_vectors = (1 - rank_vectors).clamp(10e-9, 1 - 10e-9)
                p_vectors = rank_vectors.clamp(10e-9, 1. - 10e-9)

                bce_loss = target * torch.log(p_vectors) + (1 - target) * torch.log(n_vectors)

                if self.loss_form == 'ce_focal':
                    focal_loss = self.focal["alpha"] * (1 - torch.exp(bce_loss)) ** self.focal["gamma"] * (-bce_loss)
                    self.loss = self.loss.clone() + torch.mean(focal_loss)

                elif self.loss_form == 'ce':
                    self.loss = self.loss.clone() + torch.mean(-bce_loss)

            else:
                assert "Error"

        return self.loss


def model_epoch(mode=None, epoch=0, model=None, data_loader=None, optimizer=None, labelset=None, writer=None, loss_form=""):

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
            loss = criterion._loss(outputs=outputs, labels=batch_label,
                                   loss_form=loss_form)
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
                writer.add_scalar('train_loss', tmp_loss, batch_i + (epoch - 1) * len(train_val_loader))
                print('[%d, %6d] loss: %.3f' % (epoch, batch_i * data_loader.batch_size, tmp_loss))
                running_loss = 0.0
            else:
                print('batch:', batch_i * data_loader.batch_size)

    return metrics


if __name__ == '__main__':

    # setting
    MODEL = "RESNET101"
    DATASET = "apy"
    EXP_NAME = "RESNET-pairwise-SGD-m9-freeze"
    LOAD_MODEL = MODEL
    # LOAD_MODEL = PJ('.', 'runs', DATASET, 'exp-' + str(EXP_NAME), 'epoch4.pkl')

    L_RATE = np.float64(10e-6)
    LOSS_FORM = "pairwise"
    OPTIM = "SGD"
    MOMENTUM = 0.9

    START_EPOCH = 1
    END_EPOCH = 15

    K = 100
    D = 300

    TRAIN_BATCH_SIZE = 256
    TEST_BATCH_SIZE = 256

    SAVE_PATH = PJ('.', 'runs', DATASET, 'exp-' + str(EXP_NAME))

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    # build model
    if LOAD_MODEL == "VGG16":
        model = VGG(freeze=True, pretrained=True, k=K, d=D)
        model = model.to(DEVICE)

    elif LOAD_MODEL == "RESNET101":
        model = RESNET(freeze=True, pretrained=True, k=K, d=D)
        model = model.to(DEVICE)

    else:
        print("Loading pretrained model")
        model = torch.load(LOAD_MODEL)
        model = model.to(DEVICE)

    # word2vec
    weight_path = PJ('.', 'dataset', 'google_news', 'GoogleNews-vectors-gensim-normed.bin')
    word2vec = KeyedVectors.load(weight_path, mmap='r')
    word2vec.vectors_norm = word2vec.vectors

    # label set
    labelset = LabelSet(word2vec=word2vec)

    # dataset
    train_val = ClassDataset(dataset=DATASET, mode='train_val',
                             img_transform=train_transform,
                             label_transform=labelset.train)

    test_seen = ClassDataset(dataset=DATASET, mode='test_seen',
                             img_transform=test_transform,
                             label_transform=labelset.all)

    test_unseen = ClassDataset(dataset=DATASET, mode='test_unseen',
                               img_transform=test_transform,
                               label_transform=labelset.test)

    test_unseen_all = ClassDataset(dataset=DATASET, mode='test_unseen',
                                   img_transform=test_transform,
                                   label_transform=labelset.all)

    # data_loader
    train_val_loader = torch.utils.data.DataLoader(train_val,
                                                   batch_size=TRAIN_BATCH_SIZE,
                                                   shuffle=True)

    test_seen_loader = torch.utils.data.DataLoader(test_seen,
                                                   batch_size=TEST_BATCH_SIZE,
                                                   shuffle=False)

    test_unseen_loader = torch.utils.data.DataLoader(test_unseen,
                                                     batch_size=TEST_BATCH_SIZE,
                                                     shuffle=False)

    test_unseen_all_loader = torch.utils.data.DataLoader(test_unseen_all,
                                                         batch_size=TEST_BATCH_SIZE,
                                                         shuffle=False)

    writer = SummaryWriter(SAVE_PATH)

    for epoch in range(START_EPOCH, END_EPOCH):

        # training
        criterion = HingeRankLoss(labelset.train, k=K)
        if OPTIM == "Adam":
            optimizer = optim.Adam(model.classifier.parameters(), lr=L_RATE)
        elif OPTIM == "SGD":
            optimizer = optim.SGD(model.classifier.parameters(), lr=L_RATE, momentum=MOMENTUM)

        train_metrics = model_epoch(mode="train", epoch=epoch, model=model,
                                    data_loader=train_val_loader,
                                    optimizer=optimizer, loss_form=LOSS_FORM,
                                    labelset=labelset.train, writer=writer)

        torch.save(model, PJ(SAVE_PATH, 'epoch' + str(epoch) + '.pkl'))

        train_class = train_metrics['correct'] / (train_metrics['total'] + 1e-10)
        train_acc = torch.sum(train_class) / len(labelset.train['concept_id'])
        writer.add_scalar('train_acc', train_acc * 100, epoch)

        # if epoch % 10 == 0:
        #     L_RATE = L_RATE * np.float64(0.5)
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = L_RATE
        #     writer.add_scalar('learning_rate', L_RATE, batch_i + (epoch - 1) * len(trainloader))

        # validation

        # test_seen
        test_seen_metrics = model_epoch(mode="test", epoch=epoch, model=model,
                                        data_loader=test_seen_loader,
                                        labelset=labelset.all, writer=writer)

        test_seen_class = test_seen_metrics['correct'] / (test_seen_metrics['total'] + 1e-10)
        test_seen_acc = torch.sum(test_seen_class) / len(labelset.train['concept_id'])
        writer.add_scalar('test_seen_acc', test_seen_acc * 100, epoch)

        # test_unseen
        test_unseen_metrics = model_epoch(mode="test", epoch=epoch, model=model,
                                          data_loader=test_unseen_loader,
                                          labelset=labelset.test, writer=writer)

        test_unseen_class = test_unseen_metrics['correct'] / (test_unseen_metrics['total'] + 1e-10)
        test_unseen_acc = torch.sum(test_unseen_class) / len(labelset.test['concept_id'])
        writer.add_scalar('test_unseen_acc', test_unseen_acc * 100, epoch)

        # test_unseen_all
        test_unseen_all_metrics = model_epoch(mode="test", epoch=epoch, model=model,
                                              data_loader=test_unseen_loader,
                                              labelset=labelset.all, writer=writer)

        test_unseen_all_class = test_unseen_all_metrics['correct'] / (test_unseen_all_metrics['total'] + 1e-10)
        test_unseen_all_acc = torch.sum(test_unseen_all_class) / len(labelset.test['concept_id'])
        writer.add_scalar('test_unseen_all_acc', test_unseen_all_acc * 100, epoch)

        H = 2 * test_seen_acc * test_unseen_all_acc / (test_seen_acc + test_unseen_all_acc)
        writer.add_scalar('H_acc', H, epoch)

        f = open(PJ(SAVE_PATH, "table.txt"), "a+")
        table = json.dump({str(epoch):
                           {'test_seen': test_seen_class.cpu().detach().numpy().tolist(),
                            'test_unseen_all': test_unseen_all_class.cpu().detach().numpy().tolist(),
                            'test_unseen': test_unseen_class.data.cpu().detach().numpy().tolist()}
                           }, f)
        f.close()

    writer.close()
