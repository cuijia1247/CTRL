# The codes are used to calculate different losses
# Author: cuijia1247
# Date: 2014-1-6
# version: 1.0

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import cityblock
from scipy.spatial.distance import chebyshev
from scipy import spatial
import math

class KL_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.kl_loss = nn.KLDivLoss() #越大越接近

    def forward(self, x, y):
        x_ = x.flatten(0)
        y_ = y.flatten(0)
        temp = self.kl_loss(x_, y_)
        temp = torch.sub(1, torch.log(temp))
        return temp

class Cosine_loss(nn.Module):
    def __init__(self):
        super().__init__()
        # self.consine_loss = F.cosine_similarity(dim=-1)

    def forward(self, x, y):
        x_ = x.flatten(0)
        y_ = y.flatten(0)
        temp = F.cosine_similarity(x_, y_, dim=-1)
        temp = torch.mean(temp)
        return temp

class Cityblock_loss(nn.Module):
    def __init__(self):
        super().__init__()
        # self.consine_loss = F.cosine_similarity(dim=-1)

    def forward(self, x, y):
        x_ = x.flatten(0).cpu().detach().numpy()
        y_ = y.flatten(0).cpu().detach().numpy()
        temp = cityblock(x_, y_)
        # temp = torch.mean(temp)
        return temp

class Chebyshev_loss(nn.Module):
    def __init__(self):
        super().__init__()
        # self.consine_loss = F.cosine_similarity(dim=-1)

    def forward(self, x, y):
        x_ = x.flatten(0).cpu().detach().numpy()
        y_ = y.flatten(0).cpu().detach().numpy()
        temp = chebyshev(x_, y_)
        # temp = torch.mean(temp)
        return temp

class DotProductSimilarity(nn.Module):

    def __init__(self, scale_output=False):
        super(DotProductSimilarity, self).__init__()
        self.scale_output = scale_output

    def forward(self, tensor_1, tensor_2):
        result = (tensor_1.flatten(0) * tensor_2.flatten(0)).mean()
        if self.scale_output:
            # TODO why allennlp do multiplication at here ?
            result /= math.sqrt(tensor_1.size(-1))
        return result

class ProjectedDotProductSimilarity(nn.Module):

    def __init__(self, tensor_1_dim, tensor_2_dim, projected_dim,
                 reuse_weight=False, bias=False, activation=None):
        super(ProjectedDotProductSimilarity, self).__init__()
        self.reuse_weight = reuse_weight
        self.projecting_weight_1 = nn.Parameter(torch.Tensor(tensor_1_dim, projected_dim))
        if self.reuse_weight:
            if tensor_1_dim != tensor_2_dim:
                raise ValueError('if reuse_weight=True, tensor_1_dim must equal tensor_2_dim')
        else:
            self.projecting_weight_2 = nn.Parameter(torch.Tensor(tensor_2_dim, projected_dim))
        self.bias = nn.Parameter(torch.Tensor(1)) if bias else None
        self.activation = activation

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.projecting_weight_1)
        if not self.reuse_weight:
            nn.init.xavier_uniform_(self.projecting_weight_2)
        if self.bias is not None:
            self.bias.data.fill_(0)

    def forward(self, tensor_1, tensor_2):
        projected_tensor_1 = torch.matmul(tensor_1, self.projecting_weight_1)
        if self.reuse_weight:
            projected_tensor_2 = torch.matmul(tensor_2, self.projecting_weight_1)
        else:
            projected_tensor_2 = torch.matmul(tensor_2, self.projecting_weight_2)
        result = (projected_tensor_1 * projected_tensor_2).sum(dim=-1)
        if self.bias is not None:
            result = result + self.bias
        if self.activation is not None:
            result = self.activation(result)
        return result

class Gram_loss(nn.Module):
    def __init__(self):
        super().__init__()
        # self.consine_loss = F.cosine_similarity(dim=-1)

    def forward(self, x, y):
        x_ = self.gram_matrix(x)
        y_ = self.gram_matrix(y)
        x_c, x_h, x_w = x_.shape
        y_c, y_h, y_w = y_.shape
        base_ = 4.* x_c * x_h * x_w * y_c * y_h * y_w
        result = torch.sum(torch.square(torch.sub(x_, y_))/(base_ * base_))
        return result

    def gram_matrix(self, x):
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

if __name__ == '__main__':
    pass