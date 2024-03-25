# The codes are learnt from https://github.com/ShayanPersonal/stacked-autoencoder-pytorch
# Author: cuijia1247
# Date: 2014-1-6
# version: 1.0



import os
import time

import torch
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
from model import StackedAutoEncoder_img as CLAE_IMG
from model import StackedAutoEncoder_vgg16 as CLAE_VGG16
# from model import StackedAutoEncoder_resnet50 as CLAE_RESNET50
from model_NEW import StyleLearningAutoEncoder as SLAE
import pickle
import random
# from utils.CustomLoss import Gram_loss as Gram

if not os.path.exists('./imgs'):
    os.mkdir('./imgs')

# device1 = torch.device("cuda:0")
# device2 = torch.device("cuda:1")

def gram_matrix(x):
    features = x.flatten(2, -1)
    img_num, channel_num, _ = features.shape
    gram_mat = []
    for i in range(img_num): #compute gram matrix every images
        fea_ = features[i, :]
        gram_ = torch.mm(fea_, fea_.T)
        gram_ = gram_.unsqueeze(0)
        if i == 0:
            gram_mat = gram_
        else:
            gram_mat = torch.cat((gram_mat, gram_), dim=0)
    # gram_mat = torch.mm(features, features.T)
    return gram_mat

def to_img(x):
    # x = x.view(x.size(0), 3, 32, 32) # for cifar10
    x = x.view(x.size(0), 3, 128, 128) # customized codes
    return x


def runStyleConsensusLearning(num_epochs=2000, batch_size=256, isImage_=True):
    img_transform = transforms.Compose([
        transforms.RandomRotation(360),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0, hue=0),
        transforms.ToTensor(),
    ])

    ### painting91dataset def __init__(self, labelFile, featureName, imageFolder, resize_height=512, resize_width=512, repeat=1)
    dataset = TCFLDataset(
                                ### Painting91
                                # labelFile='./data/Painting91/Labels/train.txt',
                                # featureName='./features/Painting91_vgg16_train.npy', #painting91
                                # # featureName='./features/painting91_resnet50_train.npy',  #painting91
                                # imageFolder='./data/Painting91/Images/',

                                ### Pandora
                                labelFile='./data/Pandora/Labels/train.txt',
                                featureName='./features/Pandora_vgg16_train.npy', #pandora
                                # featureName='./features/pandora_resnet50_train.npy',  # painting91
                                imageFolder='./data/Pandora/Images/',

                                ### FashionStyle14
                                # labelFile='./data/FashionStyle14/Labels/train.txt',
                                # featureName='./features/FashionStyle14_vgg16_train.npy',  # vgg16
                                # # featureName='./features/FashionStyle14_resnet50_train.npy',  #resnet50
                                # imageFolder='./data/FashionStyle14/Images/',

                                ### WikiArt3
                                # labelFile='./data/WikiArt3/Labels/train.txt',
                                # featureName='./features/WikiArt3_vgg16_train.npy',  # vgg16
                                # # featureName='./features/WikiArt3_resnet50_train.npy',  #resnet
                                # imageFolder='./data/WikiArt3/Images/',

                                ### AVAstyle multi_labels
                                # labelFile='./data/AVAstyle/Labels/train.txt',
                                # featureName='./features/AVAstyle_vgg16_train.npy',  # vgg16
                                # # featureName='./features/AVAstyle_resnet50_train.npy',  #resnet50
                                # imageFolder='./data/AVAstyle/Images/',

                                ### Arch
                                # labelFile='./data/Arch/Labels/train.txt',
                                # featureName='./features/Arch_vgg16_train.npy',  # vgg16
                                # # featureName='./features/Arch_resnet50_train.npy',  #resnet50
                                # imageFolder='./data/Arch/Images/',

                                isImage=isImage_,
                                resize_height=128,
                                resize_width=128,
                                # batch_size=512,
                                # isLeveled=isLeveled_,
                                # Levels=Levels_,
                                )
    val_dataset = TCFLDataset(
                                ### Painting91
                                # labelFile='./data/Painting91/Labels/test.txt',
                                # featureName='./features/Painting91_vgg16_test.npy', #painting91
                                # # featureName='./features/painting91_resnet50_test.npy',  # painting91
                                # imageFolder='./data/Painting91/Images/',

                                ### Pandora
                                labelFile='./data/Pandora/Labels/test.txt',
                                featureName='./features/Pandora_vgg16_test.npy',  # painting91
                                # featureName='./features/pandora_resnet50_test.npy', #pandora
                                imageFolder='./data/Pandora/Images/',

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
    # dataset = CIFAR10('./data/cifar10/', transform=img_transform, download=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=0)
    # load style dataset


    # model = CLAE_IMG().cuda() # for img
    # model = CLAE_VGG16().cuda() # for vgg16 feature
    # model = CLAE_RESNET50().cuda()  # for vgg16 feature
    # layernum = [32, 256, 256, 256] # for resnet50
    layernum = [8, 256, 256, 256] # for vgg16
    model = SLAE(layernum_=layernum, slType_='diamond').cuda() # for our new codes
    # model = SLAE(layernum_=layernum, slType_='diamond').to(device1)  # for our new codes
    # model = SLAE(layernum_=layernum).cuda()  # for our new codes

    cur_accuracy = 0.0
    cur_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    f = open(cur_time + '-log.txt', 'a')


    # total = sum([param.nelement() for param in model.parameters() if param.requires_grad])
    # print("Number of parameters: %2fM" % (total/1e6))

    # def count_params(module):
    #     return sum([param.nelement() for param in model.parameters() if param.requires_grad])
    #
    # total_params = 0
    # for name, child in model.named_children():
    #     params = count_params(child)
    #     total_params += params
    #     print("{}的参数量: {}".format(name, params))
    # print("总参数量: {}".format(total_params))
    with open('features/Pandora_ds_vgg16_train.pkl', 'rb') as pickle_f:
        styleSet = pickle.load(pickle_f)
    for epoch in range(num_epochs):
        print('epoch is {}'.format(epoch))
        # if epoch % 20 == 0: #origianl
        if epoch == 0: #classifier do not initilization
            # Test the quality of our features with a randomly initialzed linear classifier.
            # classifier = nn.Linear(512 * 16, 10).cuda() #cifa10
            # classifier = nn.Linear(512 * 4, 13).cuda()#customized codes simple classifier
            # classifier = MlpNet1(512*1024, 512*256, 512*64, 512*16, 512, 13).cuda()
            # classifier = MlpNet1().cuda()
            classifier = nn.Sequential(
                nn.Linear(12544  , 256),  # for resnet50
                # nn.Linear(6272, 256),  # for resnet50
                # nn.Linear(3136, 256),  # for resnet50
                # nn.Linear(32 * 3136, 256),  # for resnet50
                # nn.Linear(8 * 3136, 256), # for vgg16
                # nn.Dropout(p=0.7),
                nn.ReLU(),
                # nn.Linear(512, 128),
                # nn.ReLU(),
                # nn.Linear(256, 64),
                # nn.ReLU(),
                nn.Linear(256, 12),
                # nn.ReLU(),
            ).cuda()
            # ).to(device2)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
        model.train()
        loss = torch.zeros(1).cuda()
        loss_val = torch.ones(1).cuda()
        total_time = time.time()
        correct = 0
        # temp = enumerate(dataloader)
        if isImage_ == True:
            for i, (img, label) in enumerate(dataloader):
                # print(' i is {} step 1'.format(i))
                # img, target = images
                img = img
                target = label - 1 # for custom data
                # target = label # for cifar10
                target = Variable(target).cuda()
                img = Variable(img).cuda()
                # print(' i is {} step 1.5'.format(i))
                features = model(img).detach()
                # print(' i is {} step 2'.format(i))
                prediction = classifier(features.view(features.size(0), -1))
                loss = criterion(prediction, target)


                optimizer.zero_grad()
                # print('run backward')
                loss.backward()
                optimizer.step()
                pred = prediction.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        else: # isImage_=False
            for i, (img, label) in enumerate(dataloader):  # img is the feature
                target = label - 1  # for custom data
                # target = label # for cifar10
                level1 = []
                for num_ in target:
                    number = styleSet[num_].__len__()
                    index = random.randint(1, number-1)
                    # print('num_ is {}, len is {}, index is {}'.format(num_, number, index))
                    level1.append(styleSet[num_][index])
                level1 = torch.tensor(np.array(level1))
                # target = Variable(target).to(device2)
                # level = level
                img = np.reshape(img, (img.shape[0], 8, 56, 56)) # special for vgg16 conv5
                level1 = np.reshape(level1, (img.shape[0], 8, 56, 56)) # special for vgg16 conv5
                # level2 = np.reshape(level2, (img.shape[0], 8, 56, 56))  # special for vgg16 conv5
                # level3 = np.reshape(level3, (img.shape[0], 8, 56, 56))  # special for vgg16 conv5
                # img = np.reshape(img, (img.shape[0], 32, 56, 56))  # special for resnet50
                # number = styleSet[target].__len__()
                # index = random.randint(1, number)
                # level1 = np.reshape(styleSet[target][index], 32, 56, 56) # for resnet50
                # level1 = np.reshape(styleSet[target][index], 8, 56, 56)  # for vgg16
                # level1 = np.reshape(level1, (img.shape[0], 32, 56, 56)) # special for resnet50
                # level2 = np.reshape(level2, (img.shape[0], 32, 56, 56))  # special for resnet50
                # level3 = np.reshape(level3, (img.shape[0], 32, 56, 56))  # special for resnet50
                # img = np.reshape(img, (img.shape[0], 32, 56, 56))  # special for resnet50
                target = Variable(target).cuda()
                img = Variable(img).cuda()  # for features of vgg16 the dimensions are 8*56*56
                # img = Variable(img).to(device1)  # for features of vgg16 the dimensions are 8*56*56
                level1 = Variable(level1).cuda()
                # level1 = Variable(level1).to(device1)
                # level2 = Variable(level2).cuda()
                # level3 = Variable(level3).cuda()
                # print(' i is {} step 1.5'.format(i))
                features = model(img, level1, 0, 0, loss.detach()).detach()  # combine cla_loss and reconstruction loss no level2, level3
                # features = model(img, level1, level2, level3, loss.detach()).detach() #combine cla_loss and reconstruction loss
                # features = model(img, level1, level2, level3).detach() #original
                # print(' i is {} step 2'.format(i))
                # features.to(device2)
                # print("features is in {}".format(features.device))
                # features = gram_matrix(features)
                prediction = classifier(features.view(features.size(0), -1))
                loss = criterion(prediction, target)
                # loss = 0.5*loss + 0.5*loss_val

                optimizer.zero_grad()
                # print('run backward')
                loss.backward()
                optimizer.step()
                pred = prediction.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        total_time = time.time() - total_time

        model.eval()
        # # img, _ = data
        # img = Variable(img).cuda()
        # # img = Variable(img).to(device1)
        # # features, x_reconstructed = model(img, level1, level2, level3) # original with l1, l2, l3
        # features, x_reconstructed = model(img, level1, 0, 0) # without l2, l3
        # reconstruction_loss = torch.mean((x_reconstructed.data - img.data)**2)
        accuracy_val = -999
        if epoch % 1 == 0: #for validation
            correct_val = 0
            for epoch_val in range(1):
                for i, (img_val, label_val) in enumerate(val_dataloader):
                    img_val = np.reshape(img_val, (img_val.shape[0], 8, 56, 56))  # special for vgg16 conv5
                    # level1 = np.reshape(level1, (img.shape[0], 8, 56, 56))  # special for vgg16 conv5
                    # img_val = np.reshape(img_val, (img_val.shape[0], 32, 56, 56))  # special for resnet50
                    # level1 = np.reshape(level1, (img.shape[0], 32, 56, 56))  # special for resnet50
                    target_val = label_val - 1  # for custom data
                    target_val = Variable(target_val).cuda()
                    img_val = Variable(img_val).cuda()
                    # level1 = np.reshape(level1, (img.shape[0], 32, 56, 56))  # special for resnet50
                    features_val = model(img_val, 0, 0, 0, 0)  # without l2, l3
                    features_val = features_val.detach()
                    # features_val = gram_matrix(features_val)
                    prediction_val = classifier(features_val.view(features_val.size(0), -1))
                    # loss_val = criterion(prediction_val, target_val)
                    value, idx_val = prediction_val.topk(1)
                    # pred = prediction.data.max(1, keepdim=True)[1]
                    if target_val in idx_val:
                        correct_val += 1
            accuracy_val = 100 * float(correct_val) / (len(val_dataloader))

        #output the log file to both screen and logfiles
        print("Epoch {} complete\tTime: {:.4f}s\t\taccuracy_val: {:.4f}\t\tClassification Loss: {:.4f}".format(epoch, total_time, accuracy_val, loss.item()))
        print("Epoch {} complete\tTime: {:.4f}s\t\taccuracy_val: {:.4f}\t\tClassification Loss: {:.4f}".format(epoch, total_time, accuracy_val, loss.item()), file=f)
        print("Feature Statistics\tMean: {:.4f}\t\tMax: {:.4f}\t\tSparsity: {:.4f}%".format(
            torch.mean(features.data), torch.max(features.data), torch.sum(features.data == 0.0)*100 / features.data.numel())
        )

        print("Feature Statistics\tMean: {:.4f}\t\tMax: {:.4f}\t\tSparsity: {:.4f}%".format(
            torch.mean(features.data), torch.max(features.data),
            torch.sum(features.data == 0.0) * 100 / features.data.numel()), file=f
        )
        print("Linear classifier performance: {}/{} = {:.2f}%".format(correct, len(dataloader)*batch_size, 100*float(correct) / (len(dataloader)*batch_size)))
        print("Linear classifier performance: {}/{} = {:.2f}%".format(correct, len(dataloader) * batch_size,
                                                                      100 * float(correct) / (len(dataloader) * batch_size)), file=f)
        print("="*80)
        print("=" * 80, file=f)
        ### for custom data
        # temp = 100*float(correct) / (len(dataloader)*batch_size)
        if cur_accuracy < accuracy_val or epoch==(num_epochs-1):
            cur_accuracy = accuracy_val
            CDAE_name = './temp/' + cur_time + '_' + str(cur_accuracy) + '_' + str(epoch) + '_SCL.pth'
            torch.save(model.state_dict(), CDAE_name)
            CLS_name = './temp/' + cur_time + '_' + str(cur_accuracy) + '_' + str(epoch) + '_CLS.pth'
            torch.save(classifier.state_dict(), CLS_name)
    f.close()
    # torch.save(model.state_dict(), './CDAE.pth') # for cifar10

if __name__ == '__main__':
    # num_epochs = 1000
    # batch_size = 128
    isImage = False
    runStyleConsensusLearning(isImage_=isImage)


