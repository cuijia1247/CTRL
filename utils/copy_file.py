# Author: cuijia1247
# Date: 2024-12-11
# version: 1.0

import os
import tqdm
import shutil
import random

def copy_images_for_training(source_folder, source_sub_folder, target_folder, target_sub_folder):
    #prepare all folders in the target_folder
    if not os.path.exists(target_folder+target_sub_folder):
        os.makedirs(target_folder+target_sub_folder, exist_ok=True)
        # os.makedirs(target_folder+target_sub_folder + '/Images', exist_ok=True)
        os.makedirs(target_folder+target_sub_folder + '/Labels', exist_ok=True)
        # os.makedirs(target_folder+target_sub_folder + '/subImages', exist_ok=True)

    #copy Images
    shutil.copytree(source_folder+'/style_data_20241205_all_in_one/', target_folder + target_sub_folder + '/Images')

    #copy subImages
    shutil.copytree(source_folder+source_sub_folder, target_folder+target_sub_folder+'/subImages')

def copy_all_images_into_one(root_folder, sub_folder):
    if not os.path.exists(root_folder + '/style_data_20241205_all_in_one'):
        os.makedirs(root_folder + '/style_data_20241205_all_in_one', exist_ok=True)
    source_folder = root_folder + sub_folder
    for root, dirs, files in os.walk(source_folder):
        for file in tqdm.tqdm(files):
            # print(os.path.join(root, file))
            shutil.copy(os.path.join(root, file), root_folder + '/style_data_20241205_all_in_one/')

def generateLabelToTargetFolder(targetDir, sourceDir, subImage_dir, classNum):
    if not os.path.exists(targetDir):
        os.makedirs(targetDir, exist_ok=True)
    if not os.path.exists(subImage_dir):
        os.makedirs(subImage_dir, exist_ok=True)
        os.makedirs(subImage_dir + 'train/', exist_ok=True)
        os.makedirs(subImage_dir + 'test/', exist_ok=True)
    writerName1 = targetDir + 'train.txt'
    writer1 = open(writerName1, 'w')
    writerName2 = targetDir + 'test.txt'
    writer2 = open(writerName2, 'w')
    for root, dirs, _ in os.walk(sourceDir):
        for dir in dirs:
            for _, _, files in os.walk(os.path.join(root, dir)):
                for file in tqdm.tqdm(files):
                    str = "'" + file + "'"
                    for i in range(1, classNum+1):
                        if i == int(dir):
                            str += ',1'
                        else:
                            str += ',0'
                    seed = random.randint(1, classNum)
                    if seed <= 8: #train
                        writer1.write(str + os.linesep)
                        sub_target_path = subImage_dir + 'train/' + dir
                        if not os.path.exists(sub_target_path):
                            os.makedirs(sub_target_path, exist_ok=True)
                        sub_target_path = sub_target_path + '/' + file
                        root_ = root + dir + '/'
                        shutil.copy(os.path.join(root_, file), sub_target_path)
                    else:
                        writer2.write(str + os.linesep)
                        sub_target_path = subImage_dir + 'test/' + dir
                        if not os.path.exists(sub_target_path):
                            os.makedirs(sub_target_path, exist_ok=True)
                        sub_target_path = sub_target_path + '/' + file
                        root_ = root + dir + '/'
                        shutil.copy(os.path.join(root_, file), sub_target_path)
    writer1.close()
    writer2.close()


if __name__ == '__main__':
    # source_folder = '../data/webui'
    # source_sub_folder = '/style_data_20241205'
    # target_folder = '../data/webui'
    # target_sub_folder = '/webstyle'
    # copy_all_images_into_one(source_folder, source_sub_folder)
    # copy_images_for_training(source_folder, source_sub_folder, target_folder, target_sub_folder)
    source_dir = '/home/cuijia1247/Codes/style-consensus-learning/data/webui/style_data_20241205/'
    target_dir = '/home/cuijia1247/Codes/style-consensus-learning/data/webui/webstyle/Labels/'
    subImage_dir = '/home/cuijia1247/Codes/style-consensus-learning/data/webui/webstyle/subImages/'
    classNum = 10
    generateLabelToTargetFolder(target_dir, source_dir, subImage_dir, classNum)