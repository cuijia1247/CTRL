# Author: cuijia1247
# Date: 2014-1-6
# version: 1.0
# the features are extracted as a dict form (name: feature) by pretrained vgg16 model

import os

import torch.cuda
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
import numpy as np
import tqdm
import pickle

os.environ['TORCH_HOME'] = '../../pretrainModels/vgg16' #指定预训练模型下载地址
os.environ["KMP_DOUPLICATE_LIB_OK"]="TURE"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_model(layers=31): #conv1 5; conv2 10, conv3 17, conv4 24, conv5 31
    model = models.vgg16(pretrained=True).features[:layers].to(device)
    print(device)
    print(model)
    model = model.eval()
    return model

def featureE(model, imgPath, labelPath, featurePath, featureName):
    #正则化，通过ImageNet参数
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    featureTransform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ]
    )
    # features = []
    # names = []
    final_features = {}
    # current_path = os.path.abspath(os.path.join(os.getcwd(), "../../")) #set the path to the root of the project
    with open(os.path.join(getRootPath(),labelPath), 'r') as f:
        lines = f.readlines()
        for line in tqdm.tqdm(lines):
            content = line.rstrip().split(',')
            name = content[0][1:-1]
            imgP = os.path.join(imgPath, name)
            img = Image.open(os.path.join(getRootPath(),imgP)).convert('RGB')
            tensor = featureTransform(img).to(device)
            tensor = torch.unsqueeze(tensor, dim=0)
            feature = model(Variable(tensor.float(), requires_grad=False)).data.cpu().numpy().flatten()
            final_features[name] = feature
    outputP = os.path.join(featurePath, featureName)
    np.save(os.path.join(getRootPath(), outputP), final_features)
    print('The VGG16 Feature Extraction DONE.')

def getRootPath():
    return os.path.abspath(os.path.join(os.getcwd(), "../../"))

def styleDataset(model, imgFolder, targetPath, featureName, classNum):
    model.eval()
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])
    styleList = []
    for subFolder in range(classNum):
        newPath = imgFolder + '/' + str(subFolder+1) + '/'
        styleList_ = []
        for filename in tqdm.tqdm(os.listdir(newPath)):
            imgPath = newPath + filename
            img = Image.open(imgPath).convert('RGB')
            tensor = train_transform(img).to(device)
            tensor = torch.unsqueeze(tensor, dim=0)
            feature = model(tensor).to(device).data.cpu().numpy()
            feature = feature.flatten()
            styleList_.append(feature)
        styleList.append(styleList_)
    outputPath = targetPath + featureName
    with open(outputPath, 'wb') as file:
        pickle.dump(styleList, file)
    print('The VGG16 Style DataSet Feature Extraction DONE.')

if __name__ == '__main__':
    model = make_model()
    # imgPath = './data/WikiArt3/Images'
    # labelPath = './data/WikiArt3/Labels/test.txt'
    # featurePath = './features'
    # featureName = 'WikiArt3_vgg16_test.npy'
    # featureE(model, imgPath, labelPath, featurePath, featureName)
    imgPath = '../../data/Painting91/subImages/train'
    targetPath = '../../features/'
    featureName = 'Painting91_ds_vgg16_train.pkl'
    styleDataset(model, imgPath, targetPath, featureName, 13)