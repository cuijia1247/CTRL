# Author: cuijia1247
# Date: 2014-1-6
# version: 1.0
import tqdm
import os
import shutil
import random
import contextlib
import numpy as np

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

def generateLabelToTargetFolder(labelFile, targetDir, classList, classNum):
    writerName1 = targetDir + 'train.txt'
    writer1 = open(writerName1, 'a')
    writerName2 = targetDir + 'test.txt'
    writer2 = open(writerName2, 'a')
    with open(labelFile, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in tqdm.tqdm(lines):
            isTrain = True
            content = line.rstrip().split(',')
            for i in range(content.__len__()):
                # count = 0
                if i > 0:
                    temp = int(content[i])
                    if temp==1:
                        # if i==2:
                        #     print('Hello')
                        contentStr = ','.join(content)
                        print(contentStr)
                        seed = random.randint(1, classList[i-1])
                        if seed <= 0.8*classList[i-1]: #train
                            writer1.write(contentStr + os.linesep)
                            # print('train' + contentStr)
                        else:
                            writer2.write(contentStr + os.linesep)
                            # print('test' + contentStr)
                        break
        # print(content[0])
    writer1.close()
    writer2.close()

if __name__ == "__main__":
    labelfile = '/home/cuijia1247/Codes/style-consensus-learning/data/WikiArt3/Labels/label.txt'
    # classNum = 25 #for Arch
    # classNum = 13  # for painting91
    classNum = 15
    classList = getClassNum(labelfile, classNum)
    # labelFile = '/home/cuijia1247/Codes/style-consensus-learning/data/Painting91/Labels/label.txt'
    targetDir = '/home/cuijia1247/Codes/style-consensus-learning/data/WikiArt3/Labels' + '/'
    generateLabelToTargetFolder(labelfile, targetDir, classList, classNum)