# Author: cuijia1247
# Date: 2014-1-6
# version: 1.0
import os.path

# To customize the dataset class for various dataset
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils import image_processing as ip
from PIL import Image
import torch
import torchvision.transforms as transforms
import random
import tqdm
from scipy import spatial
from utils.styleLevelCal import getDatasetAnalysis as getDA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset for painting91
class TCFLDataset(Dataset):
    def __init__(self, labelFile, featureName, imageFolder, transform=False, isImage=False, resize_height=256, resize_width=256, repeat=1):
        """
        :param labelFile: label文件 xxx.txt or xxx.npy
        :param isImage: the returned is image or not
        :param featureName: feature file name , xxx.npy if the file exists
        :param imageFolder: image dir
        :param resize_height: 为None时，不缩放
        :param resize_width: 为None时，不缩放
        :param repeat: 默认不循环
        :param isLeveled: 是否加入风格强度因素
        :param Levels: 风格强度等级总数
        transform: 数据预处理
        """
        self.isImage = isImage
        self.transform = transform
        self.image_label_list, self.image_name_list, label_list = self.readLabel(labelFile)
        # self.isLeveled = isLeveled
        # self.Levels = Levels
        self.repeat = repeat
        self.imageFolder = imageFolder
        self.len = self.__len__()
        self.resize_height = resize_height
        self.resize_width = resize_width

        norm_mean = [0.485, 0.456, 0.406]
        norm_std = [0.229, 0.224, 0.225]
        # preprocessing
        self.toTransform = transforms.Compose([
            transforms.Resize((self.resize_height, self.resize_width)),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),

        ])
        ### isImage
        if self.isImage == False:
            self.features = np.load(featureName, allow_pickle=True).item()
            ### adding the style levels here
            ### analyse the dataset status in different classes
            self.classNum, self.eu_dist, self.style_level_dict = getDA(label_list, self.features, self.image_label_list)
            self.s_levels = {}
            for key in self.features.keys():
                ### set style levels
                s_level = self.setLevels(key, styleLevels='Euclidean')
                self.s_levels[key] = s_level
        else:
            self.images = self.readImage(labelFile)


    def getNNFeature(self, x_, sl_, label):#calculate the NN features in sl_
        NN_temp = []
        distance_ = 999
        for key in self.features.keys():
            if self.s_levels[key] == sl_ and label==self.image_label_list[key]:
                temp = spatial.distance.cosine(x_, self.features[key])
                if distance_ > temp: #NN
                    NN_temp = self.features[key]
                    distance_ = temp
        return NN_temp


    def readImage(self, labelFile):
        image_label_list = []
        with open(labelFile, 'r') as f:
            lines = f.readlines()
            for line in lines:
                # delete the extra signs at the ending of line
                content = line.rstrip().split(',')
                # get the name away from its suffix
                name = content[0][1:-1]
                label = -1
                count = 1
                for value in content[1:]:  # change label from one-hot into digits: 1, 0, 0, 0, 0 --> 1
                    if int(value) == 1:
                        label = count
                        break
                    else:
                        count = count + 1
                image_label_list.append((self.data_preprocess(os.path.join(self.imageFolder, name)), label))
        return image_label_list


    def __getitem__(self, i):
        index = i % self.len
        # image_name, label = self.image_label_list[index]
        image_name = self.image_name_list[index]
        label = -999
        s_level = -999
        feature = None
        # if self.transform is not None:
        #     img = self.transform(Image.open())
        if self.isImage==False: #返回的是特征，不是图像
            feature = []
            key = self.features.keys()
            for key_ in key:
                if key_ == image_name:
                    feature = self.features[key_]
                    label = self.image_label_list[key_]
                    break
            if feature.__len__()==0:
                raise ValueError('Feature Retrival ERROR.')
            return feature, label
        else:#返回的是图像, 图像不设置Levels
            img, label = self.images[index]
            return img, label

    def setLevels(self, key, styleLevels='Random'):#设置风格强度，默认采用随机方式
        if styleLevels=='Random':#随机设定风格强度
            return random.randint(0, self.Levels-1)
        elif styleLevels=='Euclidean':#欧氏距离设定风格强度
            for key_ in self.style_level_dict.keys():
                if self.image_label_list[key] == key_:
                    l4 = self.style_level_dict[key_]['level4']
                    l3 = self.style_level_dict[key_]['level3']
                    l2 = self.style_level_dict[key_]['level2']
                    if self.eu_dist[key] < l4:
                        return 3
                    elif self.eu_dist[key] >= l4 and self.eu_dist[key] < l3:
                        return 2
                    elif self.eu_dist[key] >= l3 and self.eu_dist[key] < l2:
                        return 1
                    else:
                        return 0

    def __len__(self):
        data_len = -999
        if self.repeat == None:
            data_len = 0
        else:
            data_len = len(self.image_label_list) * self.repeat
        if data_len == -999:
            raise ValueError('Dataset Length ERROR.')
        return data_len

    def readLabel(self, labelFile): #read labels from labelFile
        image_label_list = {}
        image_name_list = []
        label_list = []
        with open(labelFile, 'r') as f:
            lines = f.readlines()
            for line in lines:
                # delete the extra signs at the ending of line
                content = line.rstrip().split(',')
                # get the name away from its suffix
                name = content[0][1:-1]
                label = -1
                count = 1
                for value in content[1:]: #change label from one-hot into digits: 1, 0, 0, 0, 0 --> 1
                    if int(value) == 1:
                        label = count
                        break
                    else:
                        count = count + 1
                # image_label_list.append((name, label))
                image_name_list.append(name)
                image_label_list[name] = label
                label_list.append(label)
        return image_label_list, image_name_list, label_list

    def load_data(self, path, resize_height, resize_width, normalization=False):
        """
        :param path: data path
        :param resize_height:
        :param resize_width:
        :param normalization: whether the normalization is required.
        :return:
        """
        img = ip.read_image(path, resize_height, resize_width, normalization=False)
        return img


    def data_preprocess(self, data):
        """
        :param data: normally the data are images
        :return:
        """
        img = Image.open(data).convert('RGB')
        data = self.toTransform(img)
        return data



if __name__ == '__main__':
    pass