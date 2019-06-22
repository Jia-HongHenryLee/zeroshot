import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from collections import deque
from torchvision.models import vgg16_bn as vgg16_bn
from torchvision.models import resnet50 as resnet50
from torchvision.models import resnet101 as resnet101
from torchvision.models import resnet152 as resnet152

global DEVICE
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class visual_semantic_model(nn.Module):

    def __init__(self, pretrained=True, freeze=True, k=20, d=300):
        super(visual_semantic_model, self).__init__()
        self.k = k
        self.d = d
        self.non_linear_param = 0 if self.k == 1 else self.k + 1
        self.output_num = self.non_linear_param + self.k * (self.d + 1)

        resnet = resnet152(pretrained=pretrained)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        if freeze:
            for param in self.features.parameters():
                param.requires_grad = False

        self.transform = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, self.output_num))

    def forward(self, visual, semantics):
        visual_feature = self.features(visual)
        visual_feature = visual_feature.view(visual_feature.size(0), -1)

        self.visual_matrix = self.transform(visual_feature)

        semantics = semantics.t()
        semantics = semantics.expand(self.visual_matrix.shape[0], self.d, -1)

        if self.non_linear_param == 0:
            self.matrix = self.visual_matrix[..., :-1].view(-1, self.k, self.d)
            self.bias = self.visual_matrix[..., -1].view(-1, self.k)[..., None]
            semantic_transforms = torch.matmul(self.matrix, semantics) + self.bias

        else:
            self.visual_outer = self.visual_matrix[..., :self.non_linear_param]
            self.visual_inner = self.visual_matrix[..., self.non_linear_param:]

            self.visual_outer_matrix = self.visual_outer[..., :-1].view(-1, 1, self.k)
            self.visual_outer_bias = self.visual_outer[..., -1].view(-1, 1)[..., None]

            self.visual_inner_matrix = self.visual_inner[..., :-self.k].view(-1, self.k, self.d)
            self.visual_inner_bias = self.visual_inner[..., -self.k:].view(-1, self.k)[..., None]

            semantic_transforms = torch.matmul(self.visual_inner_matrix, semantics) + self.visual_inner_bias
            semantic_transforms = torch.tanh(semantic_transforms)
            # semantic_transforms = F.softsign(semantic_transforms)
            semantic_transforms = torch.matmul(self.visual_outer_matrix, semantic_transforms) + self.visual_outer_bias

        self.semantic_transforms = semantic_transforms.transpose(1, 2).contiguous()
        output = torch.softmax(self.semantic_transforms, dim=1)

        return output


def model_epoch(mode, epoch, loss_name, model, loss_args, data_loader, concepts, optimizer, writer):

    # loss_function
    class HingeRankLoss():

        def __init__(self, concept, loss_args):
            self.concept = concept
            # self.alpha = torch.cuda.FloatTensor([loss_args['alpha'], 1 - loss_args['alpha']])
            # self.gamma = loss_args['gamma']

        def _loss(self, outputs=None, semantics=None, labels=None):
            self.outputs = outputs
            self.labels = labels

            if data_loader.dataset.dataset == "nus_wide":

                self.alpha = torch.cuda.FloatTensor([1, 1])

                tmp_outputs = []
                tmp_labels = []

                for output, label in zip(self.outputs, self.labels):

                    pos_ind = (label == 1).nonzero()[:, 0]
                    neg_ind = (label == 0).nonzero()[:, 0]
                    neg_ind = neg_ind[torch.randperm(neg_ind.size(0))[:pos_ind.shape[0]]]

                    tmp_outputs.append(output[pos_ind])
                    tmp_outputs.append(output[neg_ind])

                    tmp_labels.append(torch.ones(pos_ind.shape[0], 1))
                    tmp_labels.append(torch.zeros(pos_ind.shape[0], 1))

                self.outputs = torch.cat(tmp_outputs).type(torch.cuda.FloatTensor)
                self.labels = torch.cat(tmp_labels).type(torch.cuda.FloatTensor)

            self.loss = F.binary_cross_entropy(self.outputs, self.labels)

            # bce_loss = F.binary_cross_entropy(self.outputs, self.labels, reduction='none').view(-1)
            # pt = Variable((-bce_loss).data.exp())
            # at = self.alpha.data.gather(0, self.labels.type(torch.cuda.LongTensor).data.view(-1))

            # self.loss = torch.mean(Variable(at) * (1 - pt) ** self.gamma * bce_loss) * 2

            return self.loss

    # epoch
    print(loss_name)

    state = None
    if loss_name.find('train') != -1:
        state = "train"
        model.train()
        torch.set_grad_enabled(True)

    elif loss_name.find('test') != -1:
        state = "test"
        model.eval()
        torch.set_grad_enabled(False)

    else:
        assert "Mode Error"

    criterion = HingeRankLoss(concepts[loss_name], loss_args)

    if mode == "single":
        metrics = {
            'total': deque(),
            'total_g': deque(),
            'correct': deque(),
            'correct_g': deque()
        }
    elif mode == "multi":
        metrics = {
            'predicts_zsl': deque(),
            'predicts_gzsl': deque(),
            'gts_zsl': deque(),
            'gts_gzsl': deque()
        }

    running_loss = 0.0

    for batch_i, batch_data in enumerate(data_loader, 1):

        # input
        batch_img = batch_data['image'].to(DEVICE)
        batch_label = batch_data['label'].to(DEVICE)

        # model produce result
        gts = batch_label[:, concepts[loss_name]['concept_label']][..., None]
        concept_vectors = concepts[loss_name]['concept_vector']
        outputs = model(Variable(batch_img), concept_vectors)

        # cal loss
        if state == 'train':

            optimizer.zero_grad()

            loss = criterion._loss(outputs=outputs, semantics=concept_vectors, labels=gts)

            # l2_params_1 = model.visual_outer_matrix.contiguous().view(-1)
            # l2_params_2 = model.visual_inner_matrix.contiguous().view(-1)
            # l2_params = torch.cat([l2_params_1, l2_params_2])

            # # l2_params = model.visual_matrix.contiguous().view(-1)
            # l2_norm = 0.0005 * torch.stack([l2_p.norm(p=2) for l2_p in l2_params]).mean()
            # print(l2_norm)

            # loss += l2_norm

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        gts_g = batch_label[:, concepts['general']['concept_label']][..., None]
        concept_vectors_g = concepts['general']['concept_vector']
        outputs_g = model(Variable(batch_img), concept_vectors_g)

        # predict
        if mode == "single":
            maxs = torch.max(outputs, 1)[1][..., None]
            maxs_onehot = torch.zero_(outputs).scatter_(1, maxs, 1)

            maxs_g = torch.max(outputs_g, 1)[1][..., None]
            maxs_g_onehot = torch.zero_(outputs_g).scatter_(1, maxs_g, 1)

            metrics['total'].extend(np.array(gts.tolist()))
            metrics['total_g'].extend(np.array(gts_g.tolist()))

            metrics['correct'].extend(np.array(maxs_onehot.tolist()))
            metrics['correct_g'].extend(np.array(maxs_g_onehot.tolist()))

        else:
            idx = torch.nonzero(torch.sum(gts, dim=1))[:, 0].tolist()
            binary_outputs = torch.gt(binary_outputs, loss_args['threshold'])
            binary_outputs_g = torch.gt(binary_outputs_g, loss_args['threshold'])

            metrics['predicts_gzsl'].extend(np.array(binary_outputs_g.tolist()))
            metrics['gts_gzsl'].extend(np.array(gts_g.tolist()))

            metrics['predicts_zsl'].extend(np.array(binary_outputs[idx].tolist()))
            metrics['gts_zsl'].extend(np.array(gts[idx].tolist()))

        # output loss
        if state == "train":
            tmp_loss = running_loss
            writer.add_scalar(loss_name + '_loss', tmp_loss, batch_i + (epoch - 1) * len(data_loader))
            print('[%d, %6d] loss: %.4f' % (epoch, batch_i * data_loader.batch_size, tmp_loss))
            running_loss = 0.0

        # if batch_i == 100:
        #     break

    return metrics
