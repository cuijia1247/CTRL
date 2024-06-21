# Author: cuijia1247
# Date: 2014-1-6
# version: 1.0

import torch
from torch import nn
from torch.autograd import Variable
from utils.CustomLoss import KL_loss
from utils.CustomLoss import Cosine_loss
from utils.CustomLoss import Cityblock_loss
from utils.CustomLoss import DotProductSimilarity
from utils.CustomLoss import ProjectedDotProductSimilarity
from utils.CustomLoss import Gram_loss

class DiamondStyleLearningCell(nn.Module):
    r"""
    Convolutional denoising autoencoder layer for styleLearning.
    This module is automatically trained when in model.training is True.

    Args:
        input_size: The number of features in the input
        output_size: The number of features to output
        stride: Stride of the convolutional layers.
        padding: Padding pattern of the convolutional layer
        stylevel, the style strongness level
    """

    def __init__(self, input_size, output_size, stylevel, stride=4, padding=0):
        super(DiamondStyleLearningCell, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.stride = stride
        self.padding = padding
        self.sl = stylevel
        ####################
        self.forward_pass = nn.Sequential(
            # nn.LayerNorm([32, 56, 56]),# for resnet50
            nn.LayerNorm([8, 56, 56]),  # for vgg16
            nn.Conv2d(input_size, int(output_size / 4), kernel_size=2, stride=stride, padding=0),
            nn.SiLU(),
            nn.Conv2d(int(output_size/4), output_size, kernel_size=2, stride=stride, padding=0),
            nn.SiLU(),
            nn.ConvTranspose2d(output_size, int(output_size / 4), kernel_size=2, stride=stride, padding=0),
            nn.SiLU(),
            nn.ConvTranspose2d(int(output_size / 4), int(input_size), kernel_size=2, stride=stride, padding=0),
        )
        self.backward_pass = nn.Sequential(
            nn.Conv2d(int(input_size), int(output_size / 4), kernel_size=2, stride=stride, padding=0),
            nn.SiLU(),
            nn.Conv2d(int(output_size / 4), output_size, kernel_size=2, stride=stride, padding=0),
            nn.SiLU(),
            nn.ConvTranspose2d(output_size, int(output_size / 4), kernel_size=2, stride=stride, padding=0),
            nn.SiLU(),
            nn.ConvTranspose2d(int(output_size / 4), input_size, kernel_size=2, stride=stride, padding=0),
        )
        self.criterion = Gram_loss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.1) #smaller variable

    def forward(self, x, l1, l2=0.0, l3=0.0, loss_=0.0, r_g=False):
        # Train each autoencoder individually
        x = x.detach()
        y = self.forward_pass(x)
        lossTotal = -999

        if self.training:
            x_reconstruct = self.backward_pass(y)
            l1_ = self.forward_pass(l1)
            loss1 = self.criterion(y, l1_)
            loss2 = self.criterion(x_reconstruct, x)
            lossTotal = 0.3*loss1 + 0.6*loss2 + loss_ #the final version
            self.optimizer.zero_grad()
            lossTotal.backward(retain_graph=r_g)
            self.optimizer.step()

        return y.detach(), lossTotal

    def reconstruct(self, x):
        return self.backward_pass(x)

class StyleLearningAutoEncoder(nn.Module):
    r"""
    A style learning autoencoder made from the convolutional denoising autoencoders above.
    Each autoencoder is trained independently and at the same time.
    """

    def __init__(self, layernum_=[], slType_='triangle', ae_num_=3, stride_=2, padding_=0, CLS_loss = 0):
        super(StyleLearningAutoEncoder, self).__init__()

        self.layernumberList = layernum_
        self.ae_num = ae_num_
        self.slType = slType_
        self.stride_ = stride_
        self.padding_ = padding_
        self.CLS_loss = CLS_loss
        if self.layernumberCheck()==True:
            if self.slType == 'triangle': #三角形网络结构，传统的U-net形式
                pass
            elif self.slType == 'diamond': #钻石型结构，为了便于后面的特征迭代
                self.ae1 = DiamondStyleLearningCell(self.layernumberList[0], self.layernumberList[1], stride=stride_, padding=padding_, stylevel=1)

        else:
            raise ValueError('SLAE initialization is FAILED.')

    def layernumberCheck(self): # check whether the layer number is workable
        num_ = len(self.layernumberList)
        if num_ == 0:
            raise ValueError('SLAE layernumber is null.')
        elif num_ < self.ae_num+1:
            raise ValueError('SLAE layernumber is illegal.')
        elif num_ == self.ae_num+1:
            return True

    def forward(self, x, l1, l2, l3, loss_=0):
        if self.slType == 'triangle':  # 三角形网络结构，传统的U-net形式
            a1, al2, al3 = self.ae1(x, l1, l2, l3, loss_) #cell1
            a2, al3 = self.ae2(a1, al2, al2, loss_) #cell2
            a3 = self.ae3(a2, al3, loss_) #cell3
        elif self.slType == 'diamond':  # 钻石型结构，为了便于后面的特征迭代
            #########parallel mode
            a1, L_ = self.ae1(x, l1, l2, l3, loss_, r_g=False)  # cell1

        if self.training:
            if self.slType == 'triangle':  # 三角形网络结构，传统的U-net形式
                return a3
            elif self.slType == 'diamond':  # 钻石型结构，为了便于后面的特征迭代
                return a1

        else:
            if self.slType == 'triangle':  # 三角形网络结构，传统的U-net形式
                return a3, self.reconstruct(a3)
            elif self.slType == 'diamond':  # 钻石型结构，为了便于后面的特征迭代
                return a1

