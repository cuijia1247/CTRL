# Author: cuijia1247
# Date: 2014-1-6
# version: 1.0

import os
import time

import torch
import torchvision.models as models
import tqdm
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from SCLdataSet_NEW import TCFLDataset
import numpy as np
from MlpNet import MlpNet1
from torchvision.datasets import CIFAR10
import datetime
import shutil
from model import StackedAutoEncoder_img as CLAE_IMG
from model import StackedAutoEncoder_vgg16 as CLAE_VGG16
# from model import StackedAutoEncoder_resnet50 as CLAE_RESNET50
from model import StyleLearningAutoEncoder as SLAE
import pickle
import random

if not os.path.exists('./imgs'):
    os.mkdir('./imgs')

def to_img(x):
    x = x.view(x.size(0), 3, 128, 128) # customized codes
    return x


def runStyleConsensusLearning(num_epochs=1, batch_size=1, isImage_=True):
    img_transform = transforms.Compose([
        transforms.RandomRotation(360),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0, hue=0),
        transforms.ToTensor(),
    ])

    ### painting91dataset def __init__(self, labelFile, featureName, imageFolder, resize_height=512, resize_width=512, repeat=1)
    dataset = TCFLDataset(
                                ### Painting91
                                labelFile='./data/Painting91/Labels/test.txt',
                                featureName='./features/painting91_resnet50_test.npy',  #painting91
                                imageFolder='./data/Painting91/Images/',

                                ### Pandora
                                # labelFile='./data/Pandora/Labels/test.txt',
                                # featureName='./features/Pandora_vgg16_test.npy',  # painting91
                                # # featureName='./features/pandora_resnet50_test.npy', #pandora
                                # imageFolder='./data/Pandora/Images/',

                                # ### FashionStyle
                                # labelFile='./data/FashionStyle14/Labels/test.txt',
                                # featureName='./features/FashionStyle14_vgg16_test.npy',  # vgg16
                                # # featureName='./features/FashionStyle14_resnet50_test.npy',  #resnet50
                                # imageFolder='./data/FashionStyle14/Images/',

                                ### Arch
                                # labelFile='./data/Arch/Labels/test.txt',
                                # featureName='./features/Arch_vgg16_test.npy',  # vgg16
                                # # featureName='./features/Arch_resnet50_test.npy',  #resnet50
                                # imageFolder='./data/Arch/Images/',

                                ### WikiArt3
                                # labelFile='./data/WikiArt3/Labels/test.txt',
                                # featureName='./features/WikiArt3_vgg16_test.npy',  # painting91
                                # # featureName='./features/WikiArt3_resnet50_test.npy',  #painting91
                                # imageFolder='./data/WikiArt3/Images/',

                                ### AVAstyle
                                # labelFile='./data/AVAstyle/Labels/test.txt',
                                # featureName='./features/AVAstyle_vgg16_test.npy',  # vgg16
                                # # featureName='./features/AVAstyle_resnet50_test.npy',  # resnet50
                                # imageFolder='./data/AVAstyle/Images/',

                                isImage=isImage_,
                                resize_height=128,
                                resize_width=128,
                                # batch_size=512,
                                # isLeveled=isLeveled_,
                                # Levels=Levels_,
                                )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    layernum = [8, 256, 256, 256]
    model_SLAE = SLAE(layernum_=layernum, slType_='diamond').cuda() # for our new codes
    model_SLAE.load_state_dict(torch.load('YOUR_ROOT_PATH' + 'Painting91_resnet_SCL.pth'))
    model_classifier = nn.Sequential(
        nn.Linear(8 * 3136, 256),  # for vgg16
        nn.ReLU(),
        nn.Linear(256, 13),
    ).cuda()
    model_classifier.load_state_dict(torch.load('YOUR_ROOT_PATH' + 'Painting91_resnet_CLS.pth'))
    criterion = nn.CrossEntropyLoss()

    cur_accuracy = 0.0
    cur_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    model_SLAE.eval()
    correct = 0

    for epoch in range(num_epochs):
        for i, (img, label) in enumerate(dataloader):
            img = np.reshape(img, (img.shape[0], 8, 56, 56))  # special for vgg16 conv5
            target = label - 1  # for custom data
            target = Variable(target).cuda()
            img = Variable(img).cuda()
            features = model_SLAE(img, 0, 0, 0, 0)# without l2, l3
            features = features.detach()
            prediction = model_classifier(features.view(features.size(0), -1))
            value, idx = prediction.topk(1) # topk(1) for top1, topk(2) for top2
            if target in idx:
                correct += 1

    temp = 100*float(correct) / (len(dataloader)*batch_size)
    print('The test CLS_accuracy = {}'.format(temp))

if __name__ == '__main__':
    isImage = False
    runStyleConsensusLearning(isImage_=isImage)


