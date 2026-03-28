# Author: cuijia1247
# Date: 2014-1-6
# version: 1.0
import tqdm
import os
import shutil
import random
import contextlib
import numpy as np
import cv2
from PIL import Image

@contextlib.contextmanager
def open_files(file1, file2):
    f1 = open(file1, 'r', encoding='utf-8')
    f2 = open(file2, 'r', encoding='utf-8')
    yield(f1, f2)

def getClassNum(filename, classNum):
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        # numList = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        numList = np.zeros(classNum)
        for line in tqdm.tqdm(lines):
            content = line.rstrip().split(',')
            for i in range(content.__len__()):
                if i > 0:
                    temp = int(content[i])
                    if temp == 1:
                        # if i==2:
                        #     print('Hello World')
                        numList[i - 1] = numList[i - 1] + 1
                        break
            # print(line)
    return numList

def copyImagsToSubDirs():
    labelFile = './data/AVAstyle/Labels/test.txt'
    sourceImgs = '/home/cuijia1247/Codes/style-consensus-learning/data/AVAstyle/Images/'
    targetImgs = '/home/cuijia1247/Codes/style-consensus-learning/data/AVAstyle/subImages/test/'
    with open(labelFile, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in tqdm.tqdm(lines):
            content = line.rstrip().split(',')
            filename = content[0][1:-1]
            dir = ''
            for i in range(15):
                temp = content[i+1]
                if temp==' 1':
                    dir = str(i+1)
                    break
            sourceFile = sourceImgs + filename
            targetFile = targetImgs + dir + '/' + filename
            targetDir = targetImgs + dir
            is_dir = os.path.exists(targetDir)
            if not is_dir:
                os.makedirs(targetDir)
            shutil.copy(sourceFile, targetFile)

def generateLabelToTargetFolder(labelFile, targetDir, classList):
    os.makedirs(targetDir, exist_ok=True)
    writerName1 = targetDir + 'train.txt'
    writer1 = open(writerName1, 'a')
    writerName2 = targetDir + 'test.txt'
    writer2 = open(writerName2, 'a')
    writerName3 = targetDir + 'val.txt'
    writer3 = open(writerName3, 'a')
    with open(labelFile, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in tqdm.tqdm(lines):
            content = line.rstrip().split(',')
            for i in range(content.__len__()):
                if i > 0:
                    temp = int(content[i])
                    if temp==1:
                        contentStr = ','.join(content)
                        # print(contentStr)
                        seed = random.randint(1, classList[i-1])
                        if seed <= 0.7*classList[i-1]: #train
                            writer1.write(contentStr + os.linesep)
                            # print('train' + contentStr)
                        elif seed > 0.7*classList[i-1] and seed <= 0.8*classList[i-1]:
                            writer3.write(contentStr + os.linesep)
                        else:
                            writer2.write(contentStr + os.linesep)
                            # print('test' + contentStr)
                        break
        # print(content[0])
    writer1.close()
    writer2.close()
    writer3.close()

def copyImagsToSubDirs(labelDir, sourceImgs, targetDir):
    dir_list = ['train', 'val', 'test']
    for lableFolder in dir_list:
        labelFile = labelDir + lableFolder + '.txt'
        targetFolder = targetDir + lableFolder + '/'

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
                targetDir_ = targetFolder + dir
                os.makedirs(targetDir_, exist_ok=True)
                targetFile = targetDir_ + '/' + filename
                # img = cv2.imread(sourceFile)
                img = Image.open(sourceFile)
                img = img.convert("RGB")

                # print(sourceFile)
                # target_img = cv2.resize(img, (256, 256))
                target_img = img.resize((256, 256))
                # cv2.imwrite(targetFile, target_img)
                target_img.save(targetFile)
            # print(filename)


if __name__ == "__main__":
    dataset = 'FashionStyle14'
    labelfile = '/home/cuijia1247/Codes/style-consensus-learning/data/' + dataset + '/Labels/label.txt'
    # classNum = 13  # for painting91
    # classNum = 12 # for pandora
    # classNum = 15 # for WikiArt3
    # classNum = 25 # for Arch
    classNum = 14 # for FashionStyle14 & AVAstyle
    classList = getClassNum(labelfile, classNum)
    # labelFile = '/home/cuijia1247/Codes/style-consensus-learning/data/Painting91/Labels/label.txt'
    targetDir = '/home/cuijia1247/Codes/style-consensus-learning/data/ST-SACLF/' + dataset + '/Labels' + '/'
    generateLabelToTargetFolder(labelfile, targetDir, classList)
    ### after the train, test, val txt generated, resize and copy to the target folders
    print('The train, val, test lablefiles are generated. Then, the copy and resize process begins')
    labelDir = '/home/cuijia1247/Codes/style-consensus-learning/data/ST-SACLF/' + dataset + '/Labels/'
    sourceImgs = '/home/cuijia1247/Codes/style-consensus-learning/data/' + dataset + '/Images/'
    targetDir = '/home/cuijia1247/Codes/style-consensus-learning/data/ST-SACLF/' + dataset + '/'
    copyImagsToSubDirs(labelDir, sourceImgs, targetDir)