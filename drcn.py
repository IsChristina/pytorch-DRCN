# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 13:42:46 2019

@author: xiaBoss
"""
import random
from torchvision.utils import save_image
import torch.optim as optim
import os
import torchvision.utils as vutils
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torchvision import datasets
import torch.utils.data
from torch.autograd import Variable
source_dataset_name = 'SVHN'
target_dataset_name = 'MNIST'
source_dataset = os.path.join('.', 'dataset', 'svhn')
target_dataset = os.path.join('.', 'dataset', 'mnist')
batch_size = 64
image_size=32
model_root = 'models'   # directory to save trained models
lr=1e-4
m_lambda=0.7
weight_decay = 5e-6
n_epoch=50
cuda = True
cudnn.benchmark = True

img_transform_svhn = transforms.Compose([
    transforms.Grayscale(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    
])

img_transform_mnist = transforms.Compose([
    transforms.Resize([32,32]),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    
])
dataset_source = datasets.SVHN(
    root=source_dataset,
    split='train',
    transform=img_transform_svhn,
)
dataset_target = datasets.MNIST(
    root=target_dataset,
    train=True,
    transform=img_transform_mnist,
    download=True
)
datasetloader_source = torch.utils.data.DataLoader(
    dataset=dataset_source,
    batch_size=batch_size,
    shuffle=True,
    
)
datasetloader_target = torch.utils.data.DataLoader(
    dataset=dataset_target,
    batch_size=batch_size,
    shuffle=True,
    
)
len_source = len(datasetloader_source) #TOTAL BATCH NUMBER
len_target = len(datasetloader_target) #TOTAL BATCH NUMBER

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform(m.weight.data, gain=1)
        nn.init.constant(m.bias.data, 0.1)

manual_seed = random.randint(1, 10000)
random.seed(manual_seed)
torch.manual_seed(manual_seed)

class DRCN(nn.Module):
    def __init__(self, n_class):
        super(DRCN, self).__init__()

        # convolutional encoder

        self.enc_feat = nn.Sequential()
        self.enc_feat.add_module('conv1', nn.Conv2d(in_channels=1, out_channels=100, kernel_size=5,
                                                    padding=2))
        self.enc_feat.add_module('relu1', nn.ReLU(True))
        self.enc_feat.add_module('pool1', nn.MaxPool2d(kernel_size=2, stride=2))

        self.enc_feat.add_module('conv2', nn.Conv2d(in_channels=100, out_channels=150, kernel_size=5,
                                                    padding=2))
        self.enc_feat.add_module('relu2', nn.ReLU(True))
        self.enc_feat.add_module('pool2', nn.MaxPool2d(kernel_size=2, stride=2))

        self.enc_feat.add_module('conv3', nn.Conv2d(in_channels=150, out_channels=200, kernel_size=3,
                                                    padding=1))
        self.enc_feat.add_module('relu3', nn.ReLU(True))

        self.enc_dense = nn.Sequential()
        self.enc_dense.add_module('fc4', nn.Linear(in_features=200 * 8 * 8, out_features=1024))
        self.enc_dense.add_module('relu4', nn.ReLU(True))
        self.enc_dense.add_module('drop4', nn.Dropout2d())

        self.enc_dense.add_module('fc5', nn.Linear(in_features=1024, out_features=1024))
        self.enc_dense.add_module('relu5', nn.ReLU(True))

        # label predict layer
        self.pred = nn.Sequential()
        self.pred.add_module('dropout6', nn.Dropout2d())
        self.pred.add_module('predict6', nn.Linear(in_features=1024, out_features=n_class))

        # convolutional decoder

        self.rec_dense = nn.Sequential()
        self.rec_dense.add_module('fc5_', nn.Linear(in_features=1024, out_features=1024))
        self.rec_dense.add_module('relu5_', nn.ReLU(True))

        self.rec_dense.add_module('fc4_', nn.Linear(in_features=1024, out_features=200 * 8 * 8))
        self.rec_dense.add_module('relu4_', nn.ReLU(True))

        self.rec_feat = nn.Sequential()

        self.rec_feat.add_module('conv3_', nn.Conv2d(in_channels=200, out_channels=150,
                                                     kernel_size=3, padding=1))
        self.rec_feat.add_module('relu3_', nn.ReLU(True))
        self.rec_feat.add_module('pool3_', nn.Upsample(scale_factor=2))

        self.rec_feat.add_module('conv2_', nn.Conv2d(in_channels=150, out_channels=100,
                                                     kernel_size=5, padding=2))
        self.rec_feat.add_module('relu2_', nn.ReLU(True))
        self.rec_feat.add_module('pool2_', nn.Upsample(scale_factor=2))

        self.rec_feat.add_module('conv1_', nn.Conv2d(in_channels=100, out_channels=1,
                                                     kernel_size=5, padding=2))

    def forward(self, input_data):
        feat = self.enc_feat(input_data)
        feat = feat.view(-1, 200 * 8 * 8)
        feat_code = self.enc_dense(feat)

        pred_label = self.pred(feat_code)

        feat_encode = self.rec_dense(feat_code)
        feat_encode = feat_encode.view(-1, 200, 8, 8)
        img_rec = self.rec_feat(feat_encode)

        return pred_label, img_rec
# load models
my_net = DRCN(n_class=10)
my_net.apply(weights_init)

def rec_image(epoch):

    model_root = 'models'
    image_root = os.path.join('.','dataset', 'svhn')

    cuda = True
    cudnn.benchmark = True
    batch_size = 64
    image_size = 32

    # load data
    img_transfrom = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ])

    dataset = datasets.SVHN(
        root=image_root,
        split='test',
        transform=img_transfrom
    )

    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        
    )

    # test
    my_net = torch.load(os.path.join(
        model_root, 'svhn_mnist_model_epoch_' + str(epoch) + '.pth')
    )
    my_net = my_net.eval()
    if cuda:
        my_net = my_net.cuda()
  
    data_iter = iter(data_loader)
    data = data_iter.next()
    img, _ = data

    batch_size = len(img)

    input_img = torch.FloatTensor(batch_size, 1, image_size, image_size)

    if cuda:
        img = img.cuda()
        input_img = input_img.cuda()

    input_img.resize_as_(img).copy_(img)
    inputv_img = Variable(input_img)

    _, rec_img = my_net(input_data=inputv_img)
    vutils.save_image(input_img, './recovery_image/svhn_real_epoch_' + str(epoch) + '.png', nrow=8)
    vutils.save_image(rec_img.data, './recovery_image/svhn_rec_' + str(epoch) + '.png', nrow=8)

def test(epoch):

    model_root = 'models'
    image_root = os.path.join('.','dataset', 'mnist')

    cuda = True
    cudnn.benchmark = True
    batch_size = 64
    image_size = 32

    # load data
    img_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        
    ])

    dataset = datasets.MNIST(
        root=image_root,
        train=False,
        transform=img_transform,
        download=True
    )

    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        
    )

    # test
    my_net = torch.load(os.path.join(
        model_root, 'svhn_mnist_model_epoch_' + str(epoch) + '.pth')
    )

    my_net = my_net.eval()
    if cuda:
        my_net = my_net.cuda()

    len_dataloader = len(data_loader)
    data_iter = iter(data_loader)

    i = 0
    n_total = 0
    n_correct = 0

    while i < len_dataloader:
    
        data = data_iter.next()
        img, label = data

        batch_size = len(label)

        input_img = torch.FloatTensor(batch_size, 1, image_size, image_size)
        class_label = torch.LongTensor(batch_size)

        if cuda:
            img = img.cuda()
            label = label.cuda()
            input_img = input_img.cuda()
            class_label = class_label.cuda()

        input_img.resize_as_(img).copy_(img)
        class_label.resize_as_(label).copy_(label)
        inputv_img = Variable(input_img)
        classv_label = Variable(class_label)

        pred_label, _ = my_net(input_data=inputv_img)
        pred = pred_label.data.max(1)[1]
        n_correct += pred.eq(classv_label.data.view_as(pred)).sum()
        n_total += batch_size

        i += 1

    accu = 100. * n_correct / len(data_loader.dataset)
    print ('epoch: {}, accuracy:{:.2f}%'.format(epoch,accu))
# setup optimizer
optimizer_classify = optim.RMSprop([{'params': my_net.enc_feat.parameters()},
                                    {'params': my_net.enc_dense.parameters()},
                                    {'params': my_net.pred.parameters()}], lr=lr, weight_decay=weight_decay)

optimizer_rec = optim.RMSprop([{'params': my_net.enc_feat.parameters()},
                               {'params': my_net.enc_dense.parameters()},
                               {'params': my_net.rec_dense.parameters()},
                               {'params': my_net.rec_feat.parameters()}], lr=lr, weight_decay=weight_decay)

loss_class = nn.CrossEntropyLoss()
loss_rec = nn.MSELoss()

if cuda:
    my_net = my_net.cuda()
    loss_class = loss_class.cuda()
    loss_rec = loss_rec.cuda()
for p in my_net.parameters():
    p.requires_grad = True
    
 # training
for epoch in range(n_epoch):   
# train reconstruction
    
    dataset_target_iter = iter(datasetloader_target)
    i = 0

    while i < len_target:
        my_net.zero_grad()

        data_target = dataset_target_iter.next()
        t_img, _ = data_target

        batch_size = len(t_img)

        input_img = torch.FloatTensor(batch_size, 1, image_size, image_size)

        if cuda:
            t_img = t_img.cuda()
            input_img = input_img.cuda()

        input_img.resize_as_(t_img).copy_(t_img)
        inputv_img = Variable(input_img)

        _, rec_img = my_net(input_data=inputv_img) #预测标签和重构的图片
        save_image(rec_img.data, './recovery_image/mnist_rec' + str(epoch) + '.png', nrow=8)
        rec_img = rec_img.view(-1, 1 * image_size * image_size)
        inputv_img_img = inputv_img.contiguous().view(-1, 1 * image_size * image_size)
        err_rec = (1 - m_lambda) * loss_rec(rec_img,inputv_img_img)
        err_rec.backward()
        optimizer_rec.step()

        i += 1

    print ('epoch: %d, err_rec %f' \
          % (epoch, err_rec.cpu().data.numpy()))
    
    # training label classifier
    dataset_source_iter = iter(datasetloader_source)
    i = 0
    correct = 0
    while i < len_source:
        my_net.zero_grad()

        data_source = dataset_source_iter.next()
        s_img, s_label = data_source
        s_label = s_label.long().squeeze()#降维
        batch_size = len(s_label)
        
        input_img = torch.FloatTensor(batch_size, 1, image_size, image_size)
        class_label = torch.LongTensor(batch_size)
        
        
        if cuda:
            s_img = s_img.cuda()
            s_label = s_label.cuda()
            input_img = input_img.cuda()
            class_label = class_label.cuda()

        input_img.resize_as_(s_img).copy_(s_img)
        class_label.resize_as_(s_label).copy_(s_label)
        inputv_img = Variable(input_img)
        classv_label = Variable(class_label) 
        pred_label, _ = my_net(input_data=inputv_img)
        preds = pred_label.data.max(1, keepdim=True)[1]

        correct += preds.eq(classv_label.data.view_as(preds)).sum()
        
        err_class = m_lambda * loss_class(pred_label, classv_label)
        err_class.backward()
        optimizer_classify.step()
        i += 1
    acc_train = float(correct) * 100. / len(datasetloader_source.dataset)
    print ('epoch: %d, err_class: %f' \
          % (epoch, err_class.cpu().data.numpy()))
    print('source-Accuracy: {}/{} ({:.2f}%)'.format(correct, len(datasetloader_source.dataset), acc_train))

    torch.save(my_net, '{0}/svhn_mnist_model_epoch_{1}.pth'.format(model_root, epoch))
    rec_image(epoch)
    test(epoch)
print ('done')
