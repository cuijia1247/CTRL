# The codes are learnt from https://github.com/ShayanPersonal/stacked-autoencoder-pytorch
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
from model_SSCAE import StyleLearningAutoEncoder as SLAE
import pickle
import random
from PIL import Image

os.environ['TORCH_HOME'] = './pretrainModels'  #指定预训练模型下载地址
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists('./imgs'):
    os.mkdir('./imgs')

def to_img(x):
    # x = x.view(x.size(0), 3, 32, 32) # for cifar10
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
                                # labelFile='./data/Painting91/Labels/test.txt',
                                # featureName='./features/Painting91_vgg16_test.npy', #painting91
                                # # featureName='./features/painting91_resnet50_test.npy',  #painting91
                                # imageFolder='./data/Painting91/Images/',

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
                                labelFile='./data/WikiArt3/Labels/test.txt',
                                featureName='./features/WikiArt3_vgg16_test.npy',  # painting91
                                # featureName='./features/WikiArt3_resnet50_test.npy',  #painting91
                                imageFolder='./data/WikiArt3/Images/',

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
    # dataset = CIFAR10('./data/cifar10/', transform=img_transform, download=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # layernum = [8, 128, 256, 512] # for vgg16
    # layernum = [32, 128, 256, 512]  # for resnet50
    layernum = [8, 32, 32, 128]
    model_SLAE = SLAE(layernum_=layernum, slType_='diamond').cuda() # for our new codes
    model_SLAE.load_state_dict(torch.load('/home/cuijia1247/Codes/style-consensus-learning/temp/2025-04-01-13-26-41_81.42400568181819_99_SCL.pth'))
    model_classifier = nn.Sequential(
        nn.Linear(6272, 256),  # for resnet50
        # nn.Linear(8*4*3136, 256), #for resnet50
        # nn.Linear(8 * 3136, 256),  # for vgg16
        # nn.Dropout(p=0.7),
        nn.ReLU(),
        # nn.Linear(512, 128),
        # nn.ReLU(),
        # nn.Linear(256, 64),
        # nn.ReLU(),
        nn.Linear(256, 15),
        # nn.ReLU(),
    ).cuda()
    model_classifier.load_state_dict(torch.load('/home/cuijia1247/Codes/style-consensus-learning/temp/2025-04-01-13-26-41_81.42400568181819_99_CLS.pth'))
    criterion = nn.CrossEntropyLoss()

    cur_accuracy = 0.0
    cur_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    model_SLAE.eval()
    correct = 0
    # with open('./features/painting91_ds_resnet50_train.pkl', 'rb') as pickle_f:
    #     styleSet = pickle.load(pickle_f)
    for epoch in range(num_epochs):
        for i, (img, label) in enumerate(dataloader):
            img = np.reshape(img, (img.shape[0], 8, 56, 56))  # special for vgg16 conv5
            # level1 = np.reshape(level1, (img.shape[0], 8, 56, 56))  # special for vgg16 conv5
            # img = np.reshape(img, (img.shape[0], 32, 56, 56))  # special for resnet50
            # level1 = np.reshape(level1, (img.shape[0], 32, 56, 56))  # special for resnet50
            target = label - 1  # for custom data
            # target = label # for cifar10
            # level1 = []
            # for num_ in target:
            #     number = styleSet[num_].__len__()
            #     index = random.randint(1, number - 1)
            #     # print('num_ is {}, len is {}, index is {}'.format(num_, number, index))
            #     level1.append(styleSet[num_][index])
            # level1 = torch.tensor(np.array(level1))
            target = Variable(target).cuda()
            img = Variable(img).cuda()
            # level1 = np.reshape(level1, (img.shape[0], 32, 56, 56))  # special for resnet50
            features, _ = model_SLAE(img, 0, 0, 0, 0)# without l2, l3
            features = features.detach()
            prediction = model_classifier(features.view(features.size(0), -1))
            value, idx = prediction.topk(1)
            # pred = prediction.data.max(1, keepdim=True)[1]
            if target in idx:
                correct += 1
            # correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            # print('The {} image is predicted as {}:{}'.format(img_name, pred, target))

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

def make_resnet_models(target_layer=4):
    resnet = models.resnet50(pretrained=True).to(device)

    # 获取模型的特定层的输出
    layers = [
        nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1),
        nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1, resnet.layer2),
        nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1, resnet.layer2,
                      resnet.layer3),
        nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1, resnet.layer2,
                      resnet.layer3, resnet.layer4)
    ]

    if target_layer < 1 or target_layer > 4:
        raise ValueError(f"Invalid target layer: {target_layer}. Target layer should be between 1 and 4.")

    # 截取模型到指定层
    feature_extractor = nn.Sequential(*layers[target_layer - 1]).eval()
    return feature_extractor

def make_vgg16_model(layers=31): #conv1 5; conv2 10, conv3 17, conv4 24, conv5 31
    model = models.vgg16(pretrained=True).features[:layers].to(device)
    model = model.eval()
    return model

def sscae_style_predict_and_return(img_path, model_SSCAE, model_classifier):
    # get the pre-trained feature from resnet50
    # layernum = [1, 32, 32, 128]  # for vgg16
    # model = SLAE(layernum_=layernum, slType_='diamond').cuda()  # for our new codes
    model = make_vgg16_model()
    model.eval()
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])
    img = Image.open(img_path).convert('RGB')
    tensor = train_transform(img).to(device)
    tensor = torch.unsqueeze(tensor, dim=0)
    feature_pre = model(tensor).to(device)
    feature_pre = feature_pre.flatten()

    # define CLS and SCL models


    feature_sty = Variable(feature_pre).cuda()
    feature_sty = feature_sty.reshape(1, 8, 56, 56) # special for vgg16
    # level1 = np.reshape(level1, (img.shape[0], 32, 56, 56))  # special for resnet50
    feature_sty = model_SSCAE(feature_sty, 0, 0, 0, 0)  # without l2, l3
    feature_sty = feature_sty.detach()
    prediction = model_classifier(feature_sty.view(feature_sty.size(0), -1))
    value, idx = prediction.topk(1)
    idx = idx.cpu().numpy()
    pre_result = -999
    if idx.size == 1:
        pre_result = idx[0][0]
    else:
        pre_result = idx[0]
    return pre_result


def output_SSCAE_reference_label(source_folder, label_path, output_file):
    correct = 0
    count = 0
    # read labels and save into dict
    label_dict = {}
    layernum = [8, 32, 32, 128]
    model_SSCAE = SLAE(layernum_=layernum, slType_='diamond').cuda()  # for our new codes
    model_SSCAE.load_state_dict(
        torch.load('/home/cuijia1247/Codes/style-consensus-learning/temp/2025-04-05-16-09-39_59.37295885042456_1870_SCL.pth'))
    model_classifier = nn.Sequential(
        nn.Linear(6272, 256),  # for resnet50
        # nn.Linear(8*4*3136, 256), #for resnet50
        # nn.Linear(8 * 3136, 256),  # for vgg16
        # nn.Dropout(p=0.7),
        nn.ReLU(),
        # nn.Linear(512, 128),
        # nn.ReLU(),
        # nn.Linear(256, 64),
        # nn.ReLU(),
        nn.Linear(256, 15),
        # nn.ReLU(),
    ).cuda()
    model_classifier.load_state_dict(
        torch.load('/home/cuijia1247/Codes/style-consensus-learning/temp/2025-04-05-16-09-39_59.37295885042456_1870_CLS.pth'))
    with open(label_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.replace(" ", "")
            line = line.replace("'", "")
            temp = line.strip().split(',')
            key = temp[0]
            value = -999
            for i in range(1, len(temp)):
                if temp[i] == '1':
                    value = i-1
                    break
            if value != -999:
                label_dict[key] = value
            else:
                print('value ERROR!!')
    with open(output_file, 'w', encoding='utf-8') as w:
        w.write('filename,prediction,label')
        for root, dirs, files in os.walk(source_folder):
            for file in tqdm.tqdm(files):
                # file = '2516.jpg'
                img_path = os.path.join(root, file)
                pre_ = sscae_style_predict_and_return(img_path, model_SSCAE, model_classifier)
                lab_ = label_dict[file]
                w.write(file + ',' + str(pre_) + ',' + str(lab_) + '\n')

def dataset_image_filter(image_folder, label_path):
    label_dict = {}
    with open(label_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.replace(" ", "")
            line = line.replace("'", "")
            temp = line.strip().split(',')
            key = temp[0]
            value = -999
            for i in range(1, len(temp)):
                if temp[i] == '1':
                    value = i
                    break
            if value != -999:
                label_dict[key] = value
            else:
                print('value ERROR!!')
    for root, dirs, files in os.walk(image_folder):
        for file in tqdm.tqdm(files):
            if file not in label_dict.keys():
                os.remove(os.path.join(root, file))
                print('The file {} is removed'.format(file))

if __name__ == '__main__':
    # num_epochs = 1000
    # batch_size = 128
    # isImage = False
    # runStyleConsensusLearning(isImage_=isImage)
    # copyImagsToSubDirs()
    ######
    source_folder = '/home/cuijia1247/Codes/style-consensus-learning/data/WikiArt3/Images'
    label_path = '/home/cuijia1247/Codes/style-consensus-learning/data/WikiArt3/Labels/label.txt'
    output_file = '/home/cuijia1247/Codes/style-consensus-learning/data/WikiArt3/output_for_HR/WikiArt3.txt'
    # dataset_image_filter(source_folder, label_path)
    output_SSCAE_reference_label(source_folder, label_path, output_file)


