# Author: cuijia1247
# Date: 2014-1-6
# version: 1.0

import os
import torch
import tqdm
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from SCLdataSet_NEW import TCFLDataset
import numpy as np
import datetime
import shutil

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
                                featureName='./features/Painting91_vgg16_test.npy',  # painting91
                                # featureName='./features/painting91_resnet50_test.npy',  # painting91
                                imageFolder='./data/Painting91/Images/',


                                isImage=isImage_,
                                resize_height=128,
                                resize_width=128,
                                # batch_size=512,
                                # isLeveled=isLeveled_,
                                # Levels=Levels_,
                                )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    layernum = [8, 256, 256, 256]

    model_SLAE = torch.load('/home/cuijia1247/Codes/style-consensus-learning/github_codes/models/painting91-vgg16/2024-06-21-19-24-15_67.6470588235294_300_SCL.pth')

    model_classifier = torch.load('/home/cuijia1247/Codes/style-consensus-learning/github_codes/models/painting91-vgg16/2024-06-21-19-24-15_67.6470588235294_300_CLS.pth')

    model_SLAE.eval()
    correct = 0

    for epoch in range(num_epochs):
        for i, (img, label) in enumerate(dataloader):
            img = np.reshape(img, (img.shape[0], 8, 56, 56))  # special for vgg16 conv5
            target = label - 1  # for custom data
            target = Variable(target).cuda()
            img = Variable(img).cuda()
            # level1 = np.reshape(level1, (img.shape[0], 32, 56, 56))  # special for resnet50
            features = model_SLAE(img, 0, 0, 0, 0)# without l2, l3
            features = features.detach()
            prediction = model_classifier(features.view(features.size(0), -1))
            value, idx = prediction.topk(1)

            if target in idx:
                correct += 1


    temp = 100*float(correct) / (len(dataloader)*batch_size)
    print('The test CLS_accuracy = {}'.format(temp))

def copyImagsToSubDirs():
    labelFile = './data/WikiArt3/Labels/test.txt'
    sourceImgs = '/home/cuijia1247/Codes/style-consensus-learning/data/WikiArt3/Images/'
    targetImgs = '/home/cuijia1247/Codes/style-consensus-learning/data/WikiArt3/subImages/test/'
    with open(labelFile, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in tqdm.tqdm(lines):
            content = line.rstrip().split(',')
            filename = content[0][1:-1]
            dir = ''
            for i in range(content.__len__()):
                if i > 0:
                    temp = content[i].lstrip()
                    if temp=='1':
                        dir = str(i)
                        break
            sourceFile = sourceImgs + filename
            targetFile = targetImgs + dir + '/' + filename
            targetDir = targetImgs + dir
            is_dir = os.path.exists(targetDir)
            if not is_dir:
                os.makedirs(targetDir)
            shutil.copy(sourceFile, targetFile)
            # print(filename)

if __name__ == '__main__':
    isImage = False
    runStyleConsensusLearning(isImage_=isImage)



