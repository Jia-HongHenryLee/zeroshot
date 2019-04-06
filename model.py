import utils
import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
from collections import deque
from torchvision.models import vgg16_bn as vgg16_bn
from torchvision.models import resnet101 as resnet101
from torchvision.models import resnet152 as resnet152

global DEVICE
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VGG(nn.Module):

    def __init__(self, freeze=True, pretrained=True, **kwarg):
        super(VGG, self).__init__()
        model = vgg16_bn(pretrained=pretrained)
        self.features = nn.Sequential(*list(model.features.children()))
        if freeze:
            for param in self.features.parameters():
                param.requires_grad = False
        self.classifier = nn.Sequential(*list(model.classifier.children())[:4])
        self.classifier._modules['3'] = nn.Linear(4096, kwarg['k'] * kwarg['d'])

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class RESNET(nn.Module):

    def __init__(self, freeze=True, pretrained=True, k=100, d=300):
        super(RESNET, self).__init__()
        model = resnet152(pretrained=pretrained)
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


def model_epoch(mode, epoch, loss_name, model, k, d, sample_rate, data_loader, concepts, optimizer, writer):

    # loss_function
    class HingeRankLoss():

        def __init__(self, concept, k, d):
            self.concept = concept
            self.k = k
            self.d = d
            self.sample_rate = sample_rate

        def _loss(self, outputs=None, labels=None, mode="train"):
            self.outputs = outputs
            self.labels = labels
            self.mode = mode

            if self.mode == "train":
                self.loss = Variable(torch.cuda.FloatTensor([0.0]), requires_grad=True)
            else:
                self.loss = Variable(torch.cuda.FloatTensor([0.0]), requires_grad=False)

            for output, label in zip(self.outputs, self.labels):

                label = label[self.concept['concept_label']]

                pos_inds = (label == 1).nonzero().reshape(-1).tolist()
                neg_inds = (label == 0).nonzero().reshape(-1).tolist()

                if sample_rate > 0 and self.mode == 'train':
                    neg_inds = np.random.choice(neg_inds, sample_rate, replace=False)

                p_vectors = self.concept['concept_vector'][pos_inds]
                n_vectors = self.concept['concept_vector'][neg_inds]

                if p_vectors.nelement() == 0:
                    continue

                p_trans = torch.mm(output, p_vectors.t()).norm(p=2, dim=0)
                n_trans = torch.mm(output, n_vectors.t()).norm(p=2, dim=0)

                subloss = torch.sum(torch.stack([torch.exp(p - n) for n in n_trans for p in p_trans]))
                subloss = torch.cuda.FloatTensor([1.0]) + subloss.clone()

                self.loss = self.loss.clone() + torch.log(subloss)

            return self.loss

    # epoch
    print(loss_name)

    if mode == 'train':
        model.train()
        torch.set_grad_enabled(True)

    elif mode == 'test':
        model.eval()
        torch.set_grad_enabled(False)

    else:
        assert "Mode Error"

    criterion = HingeRankLoss(concepts[loss_name], k, d)

    metrics = {
        'predicts_zsl': deque(),
        'predicts_gzsl': deque(),
        'gts_zsl': deque(),
        'gts_gzsl': deque()
    }

    running_loss = 0.0

    for batch_i, batch_data in enumerate(data_loader, 1):

        if mode == 'train':
            optimizer.zero_grad()

        batch_img = batch_data['image'].to(DEVICE)
        batch_label = batch_data['label'].to(DEVICE)

        outputs = model(Variable(batch_img)).view(-1, k, d)

        loss = criterion._loss(outputs=outputs, labels=batch_label, mode=mode)

        if mode == 'train':
            loss.backward()
            optimizer.step()

        running_loss += loss.item() / batch_label.shape[0]

        # tmp_metric
        tmp_metric = {
            'predicts_zsl': deque(),
            'predicts_gzsl': deque(),
            'gts_zsl': deque(),
            'gts_gzsl': deque()
        }

        # predict
        concept_vectors = concepts[loss_name]['concept_vector'].t()
        concept_vectors = concept_vectors.expand(outputs.shape[0], d, -1)
        p_ranks = torch.bmm(outputs, concept_vectors).norm(p=2, dim=1)

        for p_rank, gts in zip(p_ranks, batch_label):
            gts = gts[concepts[loss_name]['concept_label']]
            tmp_metric['predicts_zsl'].append(np.array(p_rank.tolist()))
            tmp_metric['gts_zsl'].append(np.array(gts.tolist()))
            metrics['predicts_zsl'].append(np.array(p_rank.tolist()))
            metrics['gts_zsl'].append(np.array(gts.tolist()))

        # general
        concept_vectors_g = concepts['general']['concept_vector'].t()
        concept_vectors_g = concept_vectors_g.expand(outputs.shape[0], d, -1)
        g_ranks = torch.bmm(outputs, concept_vectors_g).norm(p=2, dim=1)

        for g_rank, g_gts in zip(g_ranks, batch_label):
            tmp_metric['predicts_gzsl'].append(np.array(g_rank.tolist()))
            tmp_metric['gts_gzsl'].append(np.array(gts.tolist()))
            metrics['predicts_gzsl'].append(np.array(g_rank.tolist()))
            metrics['gts_gzsl'].append(np.array(g_gts.tolist()))

        # output loss
        tmp_loss = running_loss
        writer.add_scalar(loss_name + '_loss', tmp_loss, batch_i + (epoch - 1) * len(data_loader))
        print('[%d, %6d] loss: %.3f' % (epoch, batch_i * data_loader.batch_size, tmp_loss))
        running_loss = 0.0

        # record miap
        writer.add_scalar(loss_name + 'tmp_miap', utils.cal_miap(tmp_metric)[1], batch_i + (epoch - 1) * len(data_loader))
        writer.add_scalar(loss_name + 'tmp_g_miap', utils.cal_miap(tmp_metric, True)[1], batch_i + (epoch - 1) * len(data_loader))

    return metrics
