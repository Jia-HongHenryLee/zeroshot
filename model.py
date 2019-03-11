import torch
from torch.autograd import Variable
import torch.nn as nn
from torchvision.models import vgg16_bn as vgg16_bn
from torchvision.models import resnet101 as resnet101

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
        model = resnet101(pretrained=pretrained)
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


def model_epoch(mode, epoch, loss_name, model, k, d, data_loader, concepts, optimizer, writer):

    # loss_function
    class HingeRankLoss():

        def __init__(self, concept, k, d):
            self.concept = concept
            self.k = k
            self.d = d

        def _loss(self, outputs=None, labels=None, mode="train"):
            self.outputs = outputs
            self.labels = labels
            self.mode = mode

            if self.mode == "train":
                self.loss = Variable(torch.cuda.FloatTensor([0.0]), requires_grad=True)
            else:
                self.loss = Variable(torch.cuda.FloatTensor([0.0]), requires_grad=False)

            for output, label in zip(self.outputs, self.labels):

                pos_ind = [label.item()]
                neg_inds = [cl for cl in self.concept['concept_label'] if cl not in pos_ind]

                p_vectors = torch.stack([self.concept['concept_vector'][p] for p in pos_ind]).t()
                n_vectors = torch.stack([self.concept['concept_vector'][n] for n in neg_inds]).t()

                p_transformeds = torch.mm(output, p_vectors).norm(p=2, dim=0)
                n_transformeds = torch.mm(output, n_vectors).norm(p=2, dim=0)

                subloss = torch.cuda.FloatTensor([1.0])

                for p_transformed in p_transformeds:
                    for n_transformed in n_transformeds:
                        subloss = subloss.clone() + torch.exp(p_transformed - n_transformed)

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
        'total': {l: 0 for l in concepts[loss_name]['concept_label']},
        'correct': {l: 0 for l in concepts[loss_name]['concept_label']},
        'correct_g': {l: 0 for l in concepts['general']['concept_label']}
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

        concept_vectors = torch.stack(list(concepts[loss_name]['concept_vector'].values())).t()
        concept_vectors = concept_vectors.expand(outputs.shape[0], d, -1)

        p_rank = torch.bmm(outputs, concept_vectors).norm(p=2, dim=1)
        p_label = torch.min(p_rank, dim=1)[1]
        p_label = [concepts[loss_name]['id2label'][p.item()] for p in p_label]

        # general
        concept_vectors_g = torch.stack(list(concepts['general']['concept_vector'].values())).t()
        concept_vectors_g = concept_vectors_g.expand(outputs.shape[0], d, -1)

        g_rank = torch.bmm(outputs, concept_vectors_g).norm(p=2, dim=1)
        g_label = torch.min(g_rank, dim=1)[1]
        g_label = [concepts['general']['id2label'][g.item()] for g in g_label]

        for gt, p, g in zip(batch_label.reshape(-1), p_label, g_label):
            if gt == p:
                metrics['correct'][p] += 1
            if gt == g:
                metrics['correct_g'][g] += 1

            metrics['total'][gt.item()] += 1

        if batch_i % 3 == 0 and loss_name == 'trainval':
            tmp_loss = running_loss / 3
            writer.add_scalar('train_loss', tmp_loss, batch_i + (epoch - 1) * len(data_loader))
            print('[%d, %6d] loss: %.3f' % (epoch, batch_i * data_loader.batch_size, tmp_loss))
            running_loss = 0.0

        if loss_name is not 'trainval':
            tmp_loss = running_loss
            writer.add_scalar(loss_name + '_loss', tmp_loss, batch_i + (epoch - 1) * len(data_loader))
            print('[%d, %6d] loss: %.3f' % (epoch, batch_i * data_loader.batch_size, tmp_loss))
            running_loss = 0.0

    return metrics
