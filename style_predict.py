# Author: cuijia1247
# Date: 2024-12-13
# version: 1.0
# this code is used to predict the webui style from one single image and image folders

import os
import time
import torch
import tqdm
from torch import nn
import torchvision.models as models
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from SCLdataSet_NEW import TCFLDataset
import numpy as np
from MlpNet import MlpNet1
from torchvision.datasets import CIFAR10
import datetime
from model import StackedAutoEncoder_img as CLAE_IMG
from model import StackedAutoEncoder_vgg16 as CLAE_VGG16
# from model import StackedAutoEncoder_resnet50 as CLAE_RESNET50
from model_NEW import StyleLearningAutoEncoder as SLAE
import pickle
import random
# from utils.CustomLoss import Gram_loss as Gram
from PIL import Image
import shutil

os.environ['TORCH_HOME'] = './pretrainModels/resnet50'  #指定预训练模型下载地址
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

style_list_webui = ['1_3D',
                    '2_扁平插画',
                    '3_复古',
                    '4_渐变',
                    '5_黑暗',
                    '6_素色极简',
                    '7_黑白',
                    '8_玻璃拟化',
                    '9_杂志风',
                    '10_拟态风']

def make_models(target_layer=4):
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

def style_predict_for_one_image(img_path):
    #get the pre-trained feature from resnet50
    model = make_models()
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

    #define CLS and SCL models
    model_SLAE = torch.load(
        '/home/cuijia1247/Codes/style-consensus-learning/models/62.551103843008995_2642/2024-12-12-23-22-15_62.551103843008995_2642_SCL.pth')
    model_classifier = torch.load(
        '/home/cuijia1247/Codes/style-consensus-learning/models/62.551103843008995_2642/2024-12-12-23-22-15_62.551103843008995_2642_CLS.pth')
    feature_sty = Variable(feature_pre).cuda()
    feature_sty = feature_sty.reshape(1, 32, 56, 56)
    # level1 = np.reshape(level1, (img.shape[0], 32, 56, 56))  # special for resnet50
    feature_sty = model_SLAE(feature_sty, 0, 0, 0, 0)  # without l2, l3
    feature_sty = feature_sty.detach()
    prediction = model_classifier(feature_sty.view(feature_sty.size(0), -1))
    value, idx = prediction.topk(1)
    idx = idx.cpu().numpy()
    if idx.size == 1:
        for i in range(10):#webui style class no. is 10
            if i == idx[0][0]:
                print('Image {} is {} style'.format(img_path, style_list_webui[i]))
    # print('STYLE PREDICT DONE.')

def style_predict_img_and_return(img_path):
    # get the pre-trained feature from resnet50
    model = make_models()
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
    model_SLAE = torch.load(
        '/home/cuijia1247/Codes/style-consensus-learning/models/62.551103843008995_2642/2024-12-12-23-22-15_62.551103843008995_2642_SCL.pth')
    model_classifier = torch.load(
        '/home/cuijia1247/Codes/style-consensus-learning/models/62.551103843008995_2642/2024-12-12-23-22-15_62.551103843008995_2642_CLS.pth')
    feature_sty = Variable(feature_pre).cuda()
    feature_sty = feature_sty.reshape(1, 32, 56, 56)
    # level1 = np.reshape(level1, (img.shape[0], 32, 56, 56))  # special for resnet50
    feature_sty = model_SLAE(feature_sty, 0, 0, 0, 0)  # without l2, l3
    feature_sty = feature_sty.detach()
    prediction = model_classifier(feature_sty.view(feature_sty.size(0), -1))
    value, idx = prediction.topk(3)
    idx = idx.cpu().numpy()
    pre_result = -999
    if idx.size == 1:
        pre_result = idx[0][0]
    else:
        pre_result = idx[0]
    return pre_result

def style_predict_for_files_in_one_folder(source_folder, target_folder, class_num):
    if not os.path.exists(target_folder):
        for i in range(class_num):
            os.makedirs(target_folder + '/' + str(i), exist_ok=True)

    for root, dirs, files in os.walk(source_folder):
        for file in tqdm.tqdm(files):
            img_path = os.path.join(root, file)
            pre_ = style_predict_img_and_return(img_path)
            shutil.copy(img_path, target_folder + '/' + str(pre_) + '/')

def style_predict_accuracy(source_folder, label):
    # if not os.path.exists(target_folder):
    #     for i in range(class_num):
    #         os.makedirs(target_folder + '/' + str(i), exist_ok=True)
    correct = 0
    count = 0
    for root, dirs, files in os.walk(source_folder):
        for file in tqdm.tqdm(files):
            img_path = os.path.join(root, file)
            pre_ = style_predict_img_and_return(img_path)
            if label in pre_:
                correct += 1
                count += 1
            else:
                count += 1
    accracy = float(correct / count)
    print('The accuracy is {}'.format(accracy))
            # shutil.copy(img_path, target_folder + '/' + str(pre_) + '/')




if __name__ == '__main__':
    # img_path = '/home/cuijia1247/Codes/ui_understanding/data/style/webui_style/1_3D/DM_20240929170336_009.PNG'
    # style_predict_for_one_image(img_path)
    source_folder = '/home/cuijia1247/Codes/style-consensus-learning/data/webui/webstyle/subImages/test/1'
    # target_folder = '/home/cuijia1247/Codes/style-consensus-learning/data/webui/webstyle_prediction'
    # class_num = 10
    # style_predict_for_files_in_one_folder(source_folder, target_folder, class_num)
    label = 0
    style_predict_accuracy(source_folder, label)
