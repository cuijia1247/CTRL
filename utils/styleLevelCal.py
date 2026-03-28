# The codes are used to calculate the style level for each data
# Author: cuijia1247
# Date: 2014-1-20
# version: 1.0
from collections import Counter
import numpy as np

def getDatasetAnalysis(label_list, feature_dict, label_dict): #get the analysis data
    '''
    Args:
        label_list: the label dict should be arranged in dict form
        feature_dict: the features of data in dict form
        label_dict: the label of data in dict form
    Returns:
    '''
    dataA = getDataClassNumber(label_list)
    classNum = dataA.__len__()
    centroid_dict = getCentroid(dataA, feature_dict, label_dict)
    eu_dist, eu_dist_min, eu_dist_max = getFeatureDistance(centroid_dict, feature_dict, label_dict)
    #generate style level stardard
    style_level_dict = {}
    threshold = [0.1, 0.3, 0.5]
    for key in centroid_dict.keys():
        inner_dict = {}
        value_range = abs(eu_dist_max[key] - eu_dist_min[key])
        inner_dict['level4'] = eu_dist_min[key] + value_range * threshold[0]
        inner_dict['level3'] = eu_dist_min[key] + value_range * threshold[1]
        inner_dict['level2'] = eu_dist_min[key] + value_range * threshold[2]
        style_level_dict[key] = inner_dict

    return classNum, eu_dist, style_level_dict

def getDataClassNumber(label_list):
    dataA_ = Counter(label_list)
    return dataA_

def getCentroid(dataA_, feature_dict, label_dict):
    '''

    Args:
        dataA_: data analysis result
        label_dict: lable dict
        feature_dict: feature dict
     Returns:
    '''
    centroid_dict = {}
    # num = dataA_.__len__()
    for key_ in dataA_.keys():
        # value_sum = np.zeros(feature_dict.popitem()[1].shape)
        for temp_key in feature_dict.keys():
            temp = feature_dict[temp_key]
            value_sum = np.zeros(temp.shape)
            break
        for key in feature_dict.keys():
            if key_ == label_dict[key]:
                value_sum = np.array(value_sum) + np.array(feature_dict[key])
        centroid_dict[key_] = value_sum / dataA_[key_]
    return centroid_dict

def getFeatureDistance(centroid_dict, feature_dict, label_dict): #get feature distance away from centroid for the whole dataset feature space
    eu_dist = {}
    eu_dist_min = {}
    eu_dist_max = {}
    for key in feature_dict.keys():
        for cen_key in centroid_dict:
            if label_dict[key]==cen_key:#找到该类别的质心
                distan_ = np.sqrt(np.sum(np.square(np.array(centroid_dict[cen_key]-feature_dict[key])))) #eu_distance
                #update the min dict
                if cen_key in eu_dist_min: #cen_key已存在
                    if eu_dist_min[cen_key] >= distan_: #新的更小
                        eu_dist_min[cen_key] = distan_
                else: #cen_key不存在
                    eu_dist_min[cen_key] = distan_
                # update the max dict
                if cen_key in eu_dist_max:  # cen_key已存在
                    if eu_dist_max[cen_key] <= distan_:  # 新的更大
                        eu_dist_max[cen_key] = distan_
                else:  # cen_key不存在
                    eu_dist_max[cen_key] = distan_
                eu_dist[key] = distan_
    return eu_dist, eu_dist_min, eu_dist_max


if __name__ == '__main__':
    # a = {'1':1, '2':1, '3':3}
    # b = [1, 1, 1, 2, 5]
    # a_num = Counter(a)
    # b_num = Counter(b)
    # print(a_num)
    # print(b_num)
    a = []
    b = [2, 3, 4]
    c = np.array(a) + np.array(b)
    print(c)
    pass
