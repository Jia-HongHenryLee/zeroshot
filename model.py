import utils
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from collections import deque
from torchvision.models import vgg16_bn as vgg16_bn
from torchvision.models import resnet101 as resnet101
from torchvision.models import resnet152 as resnet152

from torchviz import make_dot

global DEVICE
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class visual_semantic_model(nn.Module):

    def __init__(self, pretrained=True, k=20, d=300):
        super(visual_semantic_model, self).__init__()
        self.k = k
        self.d = d
        self.non_linear_param = 0 if self.k == 1 else self.k + 1
        self.output_num = self.non_linear_param + self.k * (self.d + 1)

        resnet = resnet152(pretrained=pretrained)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        for param in self.features.parameters():
            param.requires_grad = False

        self.transform = nn.Sequential(
            nn.Dropout(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, self.output_num))

    def forward(self, visual, semantics):
        visual_feature = self.features(visual)
        visual_feature = visual_feature.view(visual_feature.size(0), -1)

        visual_matrix = self.transform(visual_feature)

        semantics = semantics.t()
        semantics = semantics.expand(visual_matrix.shape[0], self.d, -1)

        if self.non_linear_param == 0:
            matrix = visual_matrix[..., :-1].view(-1, self.k, self.d)
            bias = visual_matrix[..., -1].view(-1, self.k)[..., None]
            semantic_transforms = torch.matmul(matrix, semantics) + bias

        else:
            visual_outer = visual_matrix[..., :self.non_linear_param]
            visual_inner = visual_matrix[..., self.non_linear_param:]

            visual_outer_matrix = visual_outer[..., :-1].view(-1, 1, self.k)
            visual_outer_bias = visual_outer[..., -1].view(-1, 1)[..., None]

            visual_inner_matrix = visual_inner[..., :-self.k].view(-1, self.k, self.d)
            visual_inner_bias = visual_inner[..., -self.k:].view(-1, self.k)[..., None]

            semantic_transforms = torch.matmul(visual_inner_matrix, semantics) + visual_inner_bias
            semantic_transforms = torch.tanh(semantic_transforms)
            semantic_transforms = torch.matmul(visual_outer_matrix, semantic_transforms) + visual_outer_bias

        semantic_transforms = semantic_transforms.transpose(1, 2).contiguous()
        semantic_transforms = torch.sigmoid(semantic_transforms)

        return semantic_transforms


def model_epoch(mode, epoch, loss_name, model, loss_args, data_loader, concepts, optimizer, writer):

    # loss_function
    class HingeRankLoss():

        def __init__(self, concept, loss_args):
            self.concept = concept
            self.alpha = torch.cuda.FloatTensor([1 - loss_args['alpha'], loss_args['alpha']])
            self.gamma = loss_args['gamma']

        def _loss(self, outputs=None, semantics=None, labels=None, mode="train"):
            self.outputs = outputs
            self.labels = labels[:, self.concept['concept_label']][..., None]
            self.mode = mode

            bce_loss = F.binary_cross_entropy(self.outputs, self.labels, reduction='none').view(-1)
            pt = Variable((-bce_loss).data.exp())
            at = self.alpha.data.gather(0, self.labels.type(torch.cuda.LongTensor).data.view(-1))

            self.loss = torch.mean(Variable(at) * (1 - pt) ** self.gamma * bce_loss)

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

    criterion = HingeRankLoss(concepts[loss_name], loss_args)

    metrics = {
        'predicts_zsl': deque(),
        'predicts_gzsl': deque(),
        'gts_zsl': deque(),
        'gts_gzsl': deque()
    }

    running_loss = 0.0

    for batch_i, batch_data in enumerate(data_loader, 1):

        # tmp_metrics = {
        #     'predicts_zsl': deque(),
        #     'predicts_gzsl': deque(),
        #     'gts_zsl': deque(),
        #     'gts_gzsl': deque()
        # }

        if mode == 'train':
            optimizer.zero_grad()

        batch_img = batch_data['image'].to(DEVICE)
        batch_label = batch_data['label'].to(DEVICE)

        concept_vectors = concepts[loss_name]['concept_vector']
        outputs = model(Variable(batch_img), concept_vectors)

        if mode == 'train':
            loss = criterion._loss(outputs=outputs, semantics=concept_vectors, labels=batch_label, mode=mode)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # predict
        gts = batch_label[:, concepts[loss_name]['concept_label']][..., None]
        idx = torch.nonzero(torch.sum(gts, dim=1))[:, 0].tolist()
        outputs = torch.gt(outputs, loss_args['threshold'])

        metrics['predicts_zsl'].extend(np.array(outputs[idx].tolist()))
        metrics['gts_zsl'].extend(np.array(gts[idx].tolist()))
        # tmp_metrics['predicts_zsl'].extend(np.array(outputs[idx].tolist()))
        # tmp_metrics['gts_zsl'].extend(np.array(gts[idx].tolist()))

        # general
        concept_vectors_g = concepts['general']['concept_vector']
        outputs_g = model(Variable(batch_img), concept_vectors_g)
        outputs_g = torch.gt(outputs_g, loss_args['threshold'])
        gts_g = batch_label[:, concepts['general']['concept_label']][..., None]

        metrics['predicts_gzsl'].extend(np.array(outputs_g.tolist()))
        metrics['gts_gzsl'].extend(np.array(gts_g.tolist()))
        # tmp_metrics['predicts_gzsl'].extend(np.array(outputs_g.tolist()))
        # tmp_metrics['gts_gzsl'].extend(np.array(gts_g.tolist()))

        # output loss
        if mode == "train":
            tmp_loss = running_loss
            writer.add_scalar(loss_name + '_loss', tmp_loss, batch_i + (epoch - 1) * len(data_loader))
            print('[%d, %6d] loss: %.3f' % (epoch, batch_i * data_loader.batch_size, tmp_loss))
            running_loss = 0.0

        # # tmp_metric
        # train_miap = utils.cal_miap(tmp_metrics, False)
        # train_prf1 = utils.cal_prf1(tmp_metrics, general=False)
        # writer.add_scalar(loss_name + '_tmp_miap', train_miap * 100, batch_i + (epoch - 1) * len(data_loader))
        # writer.add_scalar(loss_name + '_tmp_of', train_prf1['o_f1'] * 100, batch_i + (epoch - 1) * len(data_loader))

        # if batch_i == 100:
        #    break

    return metrics
