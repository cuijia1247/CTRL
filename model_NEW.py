# The codes are learnt from https://github.com/ShayanPersonal/stacked-autoencoder-pytorch
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

class CDAutoEncoder(nn.Module):
    r"""
    Convolutional denoising autoencoder layer for stacked autoencoders.
    This module is automatically trained when in model.training is True.

    Args:
        input_size: The number of features in the input
        output_size: The number of features to output
        stride: Stride of the convolutional layers.
    """
    def __init__(self, input_size, output_size, stride):
        super(CDAutoEncoder, self).__init__()
        intermediate = int(output_size/2)
        padding_ = 0
        # if stride == 1:
        #     padding_ = 1
        # self.forward_pass = nn.Sequential(
        #     nn.Conv2d(input_size, intermediate, kernel_size=2, stride=stride, padding=padding_),
        #     nn.ReLU(),
        #     nn.Conv2d(intermediate, output_size, kernel_size=2, stride=stride, padding=0),
        #     nn.ReLU(),
        # )
        # self.backward_pass = nn.Sequential(
        #     nn.ConvTranspose2d(output_size, intermediate, kernel_size=2, stride=stride, padding=0),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(intermediate, input_size, kernel_size=2, stride=stride, padding=padding_),
        #     nn.ReLU(),
        # )
        self.forward_pass = nn.Sequential(
            nn.Conv2d(input_size, output_size, kernel_size=2, stride=stride, padding=0),
            nn.ReLU(),
        )
        self.backward_pass = nn.Sequential(
            nn.ConvTranspose2d(output_size, input_size, kernel_size=2, stride=stride, padding=0),
            nn.ReLU(),
        )
        # self.iteration = iterations
        self.criterion = nn.MSELoss() # L2-norm
        # self.criterion = nn.SmoothL1Loss() # Smooth L1 loss
        # self.criterion = nn.HuberLoss() # Huber Loss
        self.criterion2 = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.1)

    def forward(self, x):
        # Train each autoencoder individually
        x = x.detach()
        # Add noise, but use the original lossless input as the target.
        # x_noisy = x * (Variable(x.data.new(x.size()).normal_(0, 0.1)) > -.1).type_as(x) # this is noisy adding code
        x_noisy = x
        y = self.forward_pass(x_noisy)

        if self.training:
            x_reconstruct = self.backward_pass(y)
            loss = self.criterion(x_reconstruct, Variable(x.data, requires_grad=False))
            self.optimizer.zero_grad()
            # print('CDAutoEncoder backward')
            # if self.iteration % 10 == 0:
            #     loss.backward()
            loss.backward()
            self.optimizer.step()
            
        return y.detach()

    def reconstruct(self, x):
        return self.backward_pass(x)

class StackedAutoEncoder_img(nn.Module):
    r"""
    A stacked autoencoder made from the convolutional denoising autoencoders above.
    Each autoencoder is trained independently and at the same time.
    """

    def __init__(self):
        super(StackedAutoEncoder_img, self).__init__()
        # self.iterations = iterations
        self.ae1 = CDAutoEncoder(3, 128, 2)
        self.ae2 = CDAutoEncoder(128, 256, 2)
        self.ae3 = CDAutoEncoder(256, 512, 2)
        # self.ae4 = CDAutoEncoder(512, 1024, 2)
        # self.ae5 = CDAutoEncoder(1024, 2048, 2)

    def forward(self, x):
        a1 = self.ae1(x)
        a2 = self.ae2(a1)
        a3 = self.ae3(a2)
        # a4 = self.ae4(a3)
        # a5 = self.ae5(a4)

        if self.training:
            return a3

        else:
            return a3, self.reconstruct(a3)

    def reconstruct(self, x):
        # a4_reconstruct = self.ae5.reconstruct(x)
        # a3_reconstruct = self.ae4.reconstruct(a4_reconstruct)
        a2_reconstruct = self.ae3.reconstruct(x)
        a1_reconstruct = self.ae2.reconstruct(a2_reconstruct)
        x_reconstruct = self.ae1.reconstruct(a1_reconstruct)
        return x_reconstruct

class StackedAutoEncoder_vgg16(nn.Module):
    r"""
    A stacked autoencoder made from the convolutional denoising autoencoders above.
    Each autoencoder is trained independently and at the same time.
    """

    def __init__(self):
        super(StackedAutoEncoder_vgg16, self).__init__()
        # self.iterations = iterations
        self.ae1 = CDAutoEncoder(8, 128, 2)
        self.ae2 = CDAutoEncoder(128, 256, 2)
        self.ae3 = CDAutoEncoder(256, 512, 2)
        # self.ae4 = CDAutoEncoder(512, 1024, 2)
        # self.ae5 = CDAutoEncoder(1024, 2048, 2)

    def forward(self, x):
        a1 = self.ae1(x)
        a2 = self.ae2(a1)
        a3 = self.ae3(a2)
        # a4 = self.ae4(a3)
        # a5 = self.ae5(a4)

        if self.training:
            return a3

        else:
            return a3, self.reconstruct(a3)

    def reconstruct(self, x):
        # a4_reconstruct = self.ae5.reconstruct(x)
        # a3_reconstruct = self.ae4.reconstruct(a4_reconstruct)
        a2_reconstruct = self.ae3.reconstruct(x)
        a1_reconstruct = self.ae2.reconstruct(a2_reconstruct)
        x_reconstruct = self.ae1.reconstruct(a1_reconstruct)
        return x_reconstruct

class StackedAutoEncoder_resnet50(nn.Module):
    r"""
    A stacked autoencoder made from the convolutional denoising autoencoders above.
    Each autoencoder is trained independently and at the same time.
    """

    def __init__(self):
        super(StackedAutoEncoder_resnet50, self).__init__()
        # self.iterations = iterations
        self.ae1 = CDAutoEncoder(32, 128, 2)
        self.ae2 = CDAutoEncoder(128, 256, 2)
        self.ae3 = CDAutoEncoder(256, 512, 2)
        # self.ae4 = CDAutoEncoder(512, 1024, 2)
        # self.ae5 = CDAutoEncoder(1024, 2048, 2)

    def forward(self, x):
        a1 = self.ae1(x)
        a2 = self.ae2(a1)
        a3 = self.ae3(a2)
        # a4 = self.ae4(a3)
        # a5 = self.ae5(a4)

        if self.training:
            return a3

        else:
            return a3, self.reconstruct(a3)

    def reconstruct(self, x):
        # a4_reconstruct = self.ae5.reconstruct(x)
        # a3_reconstruct = self.ae4.reconstruct(a4_reconstruct)
        a2_reconstruct = self.ae3.reconstruct(x)
        a1_reconstruct = self.ae2.reconstruct(a2_reconstruct)
        x_reconstruct = self.ae1.reconstruct(a1_reconstruct)
        return x_reconstruct


class StyleLearningCell1(nn.Module):
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

    def __init__(self, input_size, output_size, stylevel, stride=2, padding=0):
        super(StyleLearningCell1, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.stride = stride
        self.padding = padding
        self.sl = stylevel
        # self.cls_loss = cls_loss
        self.forward_pass = nn.Sequential(
            nn.Conv2d(input_size, output_size, kernel_size=2, stride=stride, padding=0),
            nn.ReLU(),
        )
        self.backward_pass = nn.Sequential(
            nn.ConvTranspose2d(output_size, input_size, kernel_size=2, stride=stride, padding=0),
            nn.ReLU(),
        )

        # self.iteration = iterations
        # self.criterion = DotProductSimilarity()
        self.criterion = Cityblock_loss()
        # self.criterion = Cosine_loss()
        # self.criterion = KL_loss()
        # self.criterion = nn.MSELoss()  # L2-norm
        # self.criterion = nn.SmoothL1Loss() # Smooth L1 loss
        # self.criterion = nn.HuberLoss() # Huber Loss
        # self.criterion2 = nn.CrossEntropyLoss()
        # self.optimizer = torch.optim.SGD(self.parameters(), lr=0.001) #smaller variable
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.1) #original
        # self.optimizer = torch.optim.SGD(self.parameters(), lr=0.1) #variable
        # self.optimizer = torch.optim.SGD(self.parameters(), lr=0.00001)  # extremely smaller variable

    def forward(self, x, l1, l2, l3, loss_=0.0):
        # Train each autoencoder individually
        x = x.detach()
        y = self.forward_pass(x)
        l2_ = self.forward_pass(l2)
        l3_ = self.forward_pass(l3)

        if self.training:
            x_reconstruct = self.backward_pass(y)
            loss = self.criterion(x_reconstruct, Variable(l1.data, requires_grad=False))
            lossTotal = loss + loss_
            self.optimizer.zero_grad()
            # torch.autograd.grad(lossTotal)
            lossTotal.requires_grad_() #for cityblock
            lossTotal.backward()
            self.optimizer.step()

        return y.detach(), l2_.detach(), l3_.detach()

    def reconstruct(self, x):
        return self.backward_pass(x)

class StyleLearningCell2(nn.Module):
    r"""
    Convolutional denoising autoencoder layer for stacked autoencoders.
    This module is automatically trained when in model.training is True.

    Args:
        input_size: The number of features in the input
        output_size: The number of features to output
        stride: Stride of the convolutional layers.
    """

    def __init__(self, input_size, output_size, stylevel, stride=2, padding=0):
        super(StyleLearningCell2, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.stride = stride
        self.padding = padding
        self.sl = stylevel
        # self.cls_loss = cls_loss
        self.forward_pass = nn.Sequential(
            nn.Conv2d(input_size, output_size, kernel_size=2, stride=stride, padding=0),
            nn.ReLU(),
        )
        self.backward_pass = nn.Sequential(
            nn.ConvTranspose2d(output_size, input_size, kernel_size=2, stride=stride, padding=0),
            nn.ReLU(),
        )
        # self.iteration = iterations
        # self.criterion = DotProductSimilarity()
        self.criterion = Cityblock_loss()
        # self.criterion = Cosine_loss()
        # self.criterion = KL_loss()
        # self.criterion = nn.MSELoss()  # L2-norm
        # self.criterion = nn.SmoothL1Loss() # Smooth L1 loss
        # self.criterion = nn.HuberLoss() # Huber Loss
        self.criterion2 = nn.CrossEntropyLoss()
        # self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01) #smaller variable
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.1)  # original
        # self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01) # variable
        # self.optimizer = torch.optim.SGD(self.parameters(), lr=0.00001)  # extremely smaller variable

    def forward(self, x, l2, l3, loss_=0):
        # Train each autoencoder individually
        x = x.detach()
        y = self.forward_pass(x)
        # l2_ = self.forward_pass(l2)
        l3_ = self.forward_pass(l3)

        if self.training:
            x_reconstruct = self.backward_pass(y)
            loss = self.criterion(x_reconstruct, Variable(l2.data, requires_grad=False))
            lossTotal = loss + loss_
            self.optimizer.zero_grad()
            lossTotal.requires_grad_()  # for cityblock
            lossTotal.backward()
            self.optimizer.step()

        return y.detach(), l3_.detach()

    def reconstruct(self, x):
        return self.backward_pass(x)

class StyleLearningCell3(nn.Module):
    r"""
    Convolutional denoising autoencoder layer for stacked autoencoders.
    This module is automatically trained when in model.training is True.

    Args:
        input_size: The number of features in the input
        output_size: The number of features to output
        stride: Stride of the convolutional layers.
    """

    def __init__(self, input_size, output_size, stylevel, stride=2, padding=0):
        super(StyleLearningCell3, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.stride = stride
        self.padding = padding
        self.sl = stylevel
        # self.cls_loss = cls_loss
        self.forward_pass = nn.Sequential(
            nn.Conv2d(input_size, output_size, kernel_size=2, stride=stride, padding=0),
            nn.ReLU(),
        )
        self.backward_pass = nn.Sequential(
            nn.ConvTranspose2d(output_size, input_size, kernel_size=2, stride=stride, padding=0),
            nn.ReLU(),
        )
        # self.iteration = iterations
        # self.criterion = DotProductSimilarity()
        self.criterion = Cityblock_loss()
        # self.criterion = Cosine_loss()
        # self.criterion = KL_loss()
        # self.criterion = nn.MSELoss()  # L2-norm
        # self.criterion = nn.SmoothL1Loss() # Smooth L1 loss
        # self.criterion = nn.HuberLoss() # Huber Loss
        self.criterion2 = nn.CrossEntropyLoss()
        # self.optimizer = torch.optim.SGD(self.parameters(), lr=0.1) #smaller variable
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01) #original
        # self.optimizer = torch.optim.SGD(self.parameters(), lr=0.001)  # variable
        # self.optimizer = torch.optim.SGD(self.parameters(), lr=0.00001) #extremely smaller variable

    def forward(self, x, l3, loss_=0):
        # Train each autoencoder individually
        x = x.detach()
        y = self.forward_pass(x)

        if self.training:
            x_reconstruct = self.backward_pass(y)
            loss = self.criterion(x_reconstruct, Variable(l3.data, requires_grad=False))
            lossTotal = loss + loss_
            self.optimizer.zero_grad()
            lossTotal.requires_grad_()  # for cityblock
            lossTotal.backward()
            self.optimizer.step()

        return y.detach()

    def reconstruct(self, x):
        return self.backward_pass(x)

class DiamondStyleLearningCell1(nn.Module):
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

    def __init__(self, input_size, output_size, stylevel, stride=2, padding=0):
        super(DiamondStyleLearningCell1, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.stride = stride
        self.padding = padding
        self.sl = stylevel
        # self.cls_loss = cls_loss
        self.forward_pass = nn.Sequential(
            nn.Conv2d(input_size, output_size, kernel_size=2, stride=stride, padding=0),
            # nn.LeakyReLU(),
            # nn.SiLU(),
            nn.SiLU(),
            nn.ConvTranspose2d(output_size, input_size, kernel_size=2, stride=stride, padding=0),
            nn.SiLU(),
        )
        self.backward_pass = nn.Sequential(
            nn.Conv2d(input_size, output_size, kernel_size=2, stride=stride, padding=0),
            nn.SiLU(),
            nn.ConvTranspose2d(output_size, input_size, kernel_size=2, stride=stride, padding=0),
            nn.SiLU(),
        )

        # self.iteration = iterations
        # self.criterion = DotProductSimilarity()
        # self.criterion = Cityblock_loss()
        self.criterion = Cosine_loss()
        # self.criterion = KL_loss()
        # self.criterion = nn.MSELoss()  # L2-norm
        # self.criterion = nn.SmoothL1Loss() # Smooth L1 loss
        # self.criterion = nn.HuberLoss() # Huber Loss
        # self.criterion2 = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.001) #smaller variable
        # self.optimizer = torch.optim.SGD(self.parameters(), lr=0.1) #original
        # self.optimizer = torch.optim.SGD(self.parameters(), lr=0.1) #variable
        # self.optimizer = torch.optim.SGD(self.parameters(), lr=0.00001)  # extremely smaller variable

    def forward(self, x, l1, l2, l3, loss_=0.0):
        # Train each autoencoder individually
        x = x.detach()
        y = self.forward_pass(x)
        l2_ = self.forward_pass(l2)
        l3_ = self.forward_pass(l3)

        if self.training:
            x_reconstruct = self.backward_pass(y)
            loss = self.criterion(x_reconstruct, Variable(l1.data, requires_grad=False))
            lossTotal = loss + loss_
            # lossTotal = torch.div(torch.add(loss, loss), 2)
            self.optimizer.zero_grad()
            # torch.autograd.grad(lossTotal)
            # lossTotal.requires_grad_() #for cityblock
            lossTotal.backward()
            self.optimizer.step()

        return y.detach(), l2_.detach(), l3_.detach()

    def reconstruct(self, x):
        return self.backward_pass(x)

class DiamondStyleLearningCell2(nn.Module):
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

    def __init__(self, input_size, output_size, stylevel, stride=2, padding=0):
        super(DiamondStyleLearningCell2, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.stride = stride
        self.padding = padding
        self.sl = stylevel
        # self.cls_loss = cls_loss
        intermediate = int((input_size + output_size)/2)
        self.forward_pass = nn.Sequential(
            nn.Conv2d(input_size, intermediate, kernel_size=2, stride=stride, padding=0),
            nn.SiLU(),
            nn.Conv2d(intermediate, output_size, kernel_size=2, stride=stride, padding=0),
            nn.SiLU(),
            nn.ConvTranspose2d(output_size, intermediate, kernel_size=2, stride=stride, padding=0),
            nn.SiLU(),
            nn.ConvTranspose2d(intermediate, input_size, kernel_size=2, stride=stride, padding=0),
            nn.SiLU(),
        )
        self.backward_pass = nn.Sequential(
            nn.Conv2d(input_size, intermediate, kernel_size=2, stride=stride, padding=0),
            nn.SiLU(),
            nn.Conv2d(intermediate, output_size, kernel_size=2, stride=stride, padding=0),
            nn.SiLU(),
            nn.ConvTranspose2d(output_size, intermediate, kernel_size=2, stride=stride, padding=0),
            nn.SiLU(),
            nn.ConvTranspose2d(intermediate, input_size, kernel_size=2, stride=stride, padding=0),
            nn.SiLU(),
        )

        # self.iteration = iterations
        # self.criterion = DotProductSimilarity()
        # self.criterion = Cityblock_loss()
        # self.criterion = Cosine_loss()
        # self.criterion = KL_loss()
        # self.criterion = nn.MSELoss()  # L2-norm
        # self.criterion = nn.SmoothL1Loss() # Smooth L1 loss
        self.criterion = nn.HuberLoss() # Huber Loss
        # self.criterion2 = nn.CrossEntropyLoss()
        # self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01) #smaller variable
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.1) #original
        # self.optimizer = torch.optim.SGD(self.parameters(), lr=0.1) #variable
        # self.optimizer = torch.optim.SGD(self.parameters(), lr=0.00001)  # extremely smaller variable

    def forward(self, x, l2, l3, loss_=0):
        # Train each autoencoder individually
        x = x.detach()
        y = self.forward_pass(x)
        # l2_ = self.forward_pass(l2)
        l3_ = self.forward_pass(l3)

        if self.training:
            x_reconstruct = self.backward_pass(y)
            loss = self.criterion(x_reconstruct, Variable(l2.data, requires_grad=False))
            # lossTotal = loss + loss_
            self.optimizer.zero_grad()
            # torch.autograd.grad(lossTotal)
            # loss.requires_grad_() #for cityblock
            loss.backward()
            self.optimizer.step()

        return y.detach(), l3_.detach()

    def reconstruct(self, x):
        return self.backward_pass(x)

class DiamondStyleLearningCell3(nn.Module):
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

    def __init__(self, input_size, output_size, stylevel, stride=2, padding=0):
        super(DiamondStyleLearningCell3, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.stride = stride
        self.padding = padding
        self.sl = stylevel
        # self.cls_loss = cls_loss
        intermediate1 = int((input_size + output_size)/2)
        intermediate2 = int((input_size + intermediate1) / 2)
        self.forward_pass = nn.Sequential(
            nn.Conv2d(input_size, intermediate1, kernel_size=2, stride=stride, padding=0),
            nn.SiLU(),
            nn.Conv2d(intermediate1, intermediate2, kernel_size=2, stride=stride, padding=0),
            nn.SiLU(),
            nn.Conv2d(intermediate2, output_size, kernel_size=2, stride=stride, padding=0),
            nn.SiLU(),
            nn.ConvTranspose2d(output_size, intermediate2, kernel_size=2, stride=stride, padding=0),
            nn.SiLU(),
            nn.ConvTranspose2d(intermediate2, intermediate1, kernel_size=2, stride=stride, padding=0),
            nn.SiLU(),
            nn.ConvTranspose2d(intermediate1, input_size, kernel_size=2, stride=stride, padding=0),
            nn.SiLU(),
        )
        self.backward_pass = nn.Sequential(
            nn.Conv2d(input_size, intermediate1, kernel_size=2, stride=stride, padding=0),
            nn.SiLU(),
            nn.Conv2d(intermediate1, intermediate2, kernel_size=2, stride=stride, padding=0),
            nn.SiLU(),
            nn.Conv2d(intermediate2, output_size, kernel_size=2, stride=stride, padding=0),
            nn.SiLU(),
            nn.ConvTranspose2d(output_size, intermediate2, kernel_size=2, stride=stride, padding=0),
            nn.SiLU(),
            nn.ConvTranspose2d(intermediate2, intermediate1, kernel_size=2, stride=stride, padding=0),
            nn.SiLU(),
            nn.ConvTranspose2d(intermediate1, input_size, kernel_size=2, stride=stride, padding=0),
            nn.SiLU(),
        )

        # self.iteration = iterations
        # self.criterion = DotProductSimilarity()
        # self.criterion = Cityblock_loss()
        # self.criterion = Cosine_loss()
        # self.criterion = KL_loss()
        # self.criterion = nn.MSELoss()  # L2-norm
        # self.criterion = nn.SmoothL1Loss() # Smooth L1 loss
        self.criterion = nn.HuberLoss() # Huber Loss
        # self.criterion2 = nn.CrossEntropyLoss()
        # self.optimizer = torch.optim.SGD(self.parameters(), lr=0.1) #smaller variable
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.1) #original
        # self.optimizer = torch.optim.SGD(self.parameters(), lr=0.1) #variable
        # self.optimizer = torch.optim.SGD(self.parameters(), lr=0.00001)  # extremely smaller variable

    def forward(self, x, l3, loss_=0):
        # Train each autoencoder individually
        x = x.detach()
        y = self.forward_pass(x)
        # l2_ = self.forward_pass(l2)
        # l3_ = self.forward_pass(l3)

        if self.training:
            x_reconstruct = self.backward_pass(y)
            loss = self.criterion(x_reconstruct, Variable(l3.data, requires_grad=False))
            # lossTotal = loss + loss_
            self.optimizer.zero_grad()
            # torch.autograd.grad(lossTotal)
            # loss.requires_grad_() #for cityblock
            loss.backward()
            self.optimizer.step()

        return y.detach()

    def reconstruct(self, x):
        return self.backward_pass(x)

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
        # self.norm = nn.LayerNorm([8, 56, 56])
        # self.cls_loss = cls_loss
        ######################
        # self.forward_pass = nn.Sequential(
        #     # nn.LayerNorm([32, 56, 56]),# for resnet50
        #     nn.LayerNorm([8, 56, 56]),  # for vgg16
        #     nn.Conv2d(input_size, int(output_size / 4), kernel_size=2, stride=stride, padding=0),
        #     nn.SiLU(),
        #     # nn.Conv2d(int(output_size / 4), int(output_size / 4), kernel_size=2, stride=stride, padding=0),
        #     # nn.SiLU(),
        #     nn.Conv2d(int(output_size/4), int(output_size/2), kernel_size=2, stride=stride, padding=0),
        #     nn.SiLU(),
        #     nn.Conv2d(int(output_size/2), output_size, kernel_size=2, stride=stride, padding=0),
        #     # nn.SiLU(),
        # )
        # self.backward_pass = nn.Sequential(
        #     # nn.LayerNorm([32, 56, 56]), # for resnet50
        #     # nn.LayerNorm([8, 56, 56]),  # for vgg16
        #     nn.ConvTranspose2d(output_size, int(output_size/2), kernel_size=2, stride=stride, padding=0),
        #     nn.SiLU(),
        #     nn.ConvTranspose2d(int(output_size/2), int(output_size/4), kernel_size=2, stride=stride, padding=0),
        #     nn.SiLU(),
        #     # nn.ConvTranspose2d(int(output_size / 4), int(output_size / 4), kernel_size=2, stride=stride, padding=0),
        #     # nn.SiLU(),
        #     nn.ConvTranspose2d(int(output_size/4), input_size, kernel_size=2, stride=stride, padding=0),
        #     # nn.SiLU(),
        #
        # )
        ####################
        self.forward_pass = nn.Sequential(
            # nn.LayerNorm([32, 56, 56]),# for resnet50
            nn.LayerNorm([8, 56, 56]),  # for vgg16
            nn.Conv2d(input_size, int(output_size / 2), kernel_size=2, stride=stride, padding=0),
            nn.SiLU(),
            # nn.Conv2d(int(output_size / 4), int(output_size / 4), kernel_size=2, stride=stride, padding=0),
            # nn.SiLU(),
            nn.Conv2d(int(output_size/2), int(output_size/2), kernel_size=2, stride=stride, padding=0),
            nn.SiLU(),
            nn.Conv2d(int(output_size/2), output_size, kernel_size=2, stride=stride, padding=0),
            nn.SiLU(),
            nn.ConvTranspose2d(output_size, int(output_size / 2), kernel_size=2, stride=stride, padding=0),
            nn.SiLU(),
            nn.ConvTranspose2d(int(output_size / 2), int(output_size / 4), kernel_size=2, stride=stride, padding=0),
            nn.SiLU(),
            nn.ConvTranspose2d(int(output_size / 4), int(input_size / 2), kernel_size=2, stride=stride, padding=0),
        )
        self.backward_pass = nn.Sequential(
            # nn.LayerNorm([8, 56, 56]),  # for vgg16
            nn.Conv2d(int(input_size / 2), int(output_size / 2), kernel_size=2, stride=stride, padding=0),
            nn.SiLU(),
            # nn.Conv2d(int(output_size / 4), int(output_size / 4), kernel_size=2, stride=stride, padding=0),
            # nn.SiLU(),
            nn.Conv2d(int(output_size / 2), int(output_size / 2), kernel_size=2, stride=stride, padding=0),
            nn.SiLU(),
            nn.Conv2d(int(output_size / 2), output_size, kernel_size=2, stride=stride, padding=0),
            nn.SiLU(),
            nn.ConvTranspose2d(output_size, int(output_size / 2), kernel_size=2, stride=stride, padding=0),
            nn.SiLU(),
            nn.ConvTranspose2d(int(output_size / 2), int(output_size / 4), kernel_size=2, stride=stride, padding=0),
            nn.SiLU(),
            nn.ConvTranspose2d(int(output_size / 4), input_size, kernel_size=2, stride=stride, padding=0),
        )

        # self.iteration = iterations
        # self.criterion = DotProductSimilarity()
        # self.criterion = Cityblock_loss()
        # self.criterion = Cosine_loss()
        self.criterion = Gram_loss()
        # self.criterion = KL_loss()
        # self.criterion = nn.MSELoss()  # L2-norm
        # self.criterion = nn.SmoothL1Loss() # Smooth L1 loss
        # self.criterion = nn.HuberLoss() # Huber Loss
        # self.criterion2 = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01) #smaller variable
        # self.optimizer = torch.optim.SGD(self.parameters(), lr=0.1) #original
        # self.optimizer = torch.optim.SGD(self.parameters(), lr=0.1) #variable
        # self.optimizer = torch.optim.SGD(self.parameters(), lr=0.00001)  # extremely smaller variable

    def forward(self, x, l1, l2=0.0, l3=0.0, loss_=0.0, r_g=False):
        # Train each autoencoder individually
        x = x.detach()
        y = self.forward_pass(x)
        # l2_ = self.forward_pass(l2)
        # l3_ = self.forward_pass(l3)
        lossTotal = -999

        if self.training:
            x_reconstruct = self.backward_pass(y)
            l1_ = self.forward_pass(l1)
            loss1 = self.criterion(y, l1_)
            loss2 = self.criterion(x_reconstruct, x)
            # print("loss1 is {}, loss 2 is {}".format(loss1, loss2))
            # lossTotal = self.criterion(x, x_reconstruct)
            # loss_.to(loss1.device)
            # print("loss1 is {}, loss2 is {}, loss_ is {}".format(loss1.device, loss2.device, loss_.device))
            lossTotal = 0.3*loss1 + 0.6*loss2 + loss_ #the final version
            # lossTotal = loss1*0 + loss2
            # loss_result = lossTotal
            # if r_g == False:
            #     print('The current insider loss is {}'.format(loss))
            # lossTotal = torch.div(torch.add(loss, loss), 2)
            self.optimizer.zero_grad()
            # torch.autograd.grad(lossTotal)
            # lossTotal.requires_grad_() #for cityblock
            # lossTotal.backward(retain_graph=True)
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
        # sltType is 'triangle', 'diamond'
        # ae_num is the number of dae class
        # layernumlist is the channel number in different dae class
        # if the dae number is n, the necessary layer number is n+1
        # self.iterations = iterations
        self.layernumberList = layernum_
        self.ae_num = ae_num_
        self.slType = slType_
        self.stride_ = stride_
        self.padding_ = padding_
        self.CLS_loss = CLS_loss
        if self.layernumberCheck()==True:
            if self.slType == 'triangle': #三角形网络结构，传统的U-net形式
                self.ae1 = StyleLearningCell1(self.layernumberList[0], self.layernumberList[1], stride=stride_, padding=padding_, stylevel=1)
                self.ae2 = StyleLearningCell2(self.layernumberList[1], self.layernumberList[2], stride=stride_, padding=padding_, stylevel=2)
                self.ae3 = StyleLearningCell3(self.layernumberList[2], self.layernumberList[3], stride=stride_, padding=padding_, stylevel=3)
            elif self.slType == 'diamond': #钻石型结构，为了便于后面的特征迭代
                self.ae1 = DiamondStyleLearningCell(self.layernumberList[0], self.layernumberList[1], stride=stride_, padding=padding_, stylevel=1)
                self.ae11 = DiamondStyleLearningCell(self.layernumberList[0], self.layernumberList[2], stride=stride_, padding=padding_, stylevel=1)
                self.ae111 = DiamondStyleLearningCell(self.layernumberList[0], self.layernumberList[3], stride=stride_, padding=padding_, stylevel=1)
                # self.ae1 = DiamondStyleLearningCell1(self.layernumberList[0], self.layernumberList[1], stride=stride_,  padding=padding_, stylevel=1)
                # self.ae2 = DiamondStyleLearningCell2(self.layernumberList[0], self.layernumberList[2], stride=stride_, padding=padding_, stylevel=2)
                # self.ae3 = DiamondStyleLearningCell3(self.layernumberList[0], self.layernumberList[3], stride=stride_, padding=padding_, stylevel=3)

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
            # a4 = self.ae4(a3)
            # a5 = self.ae5(a4)
        elif self.slType == 'diamond':  # 钻石型结构，为了便于后面的特征迭代
            #########parallel mode
            a1, L_ = self.ae1(x, l1, l2, l3, loss_, r_g=False)  # cell1
            # x = torch.div(torch.add(a1, x), 2)
            # a1, L_ = self.ae1(x, l1, l2, l3, loss_, r_g=False)  # cell1
            # x = torch.div(torch.add(a1, x), 2)
            a11, L_ = self.ae11(x, l1, l2, l3, loss_, r_g=False)  # cell1
            # x = torch.div(torch.add(a11, x), 2)
            # a11, L_ = self.ae11(x, l1, l2, l3, loss_, r_g=False)  # cell1
            # x = torch.div(torch.add(a11, x), 2)
            a111, L_ = self.ae11(x, l1, l2, l3, loss_)  # cell1
            # x = torch.div(torch.add(a111, x), 2)
            # a111, L_ = self.ae11(x, l1, l2, l3, loss_)  # cell1
            # print('The style loss is {}'.format(L_))
            #########dslc-dslc-dslc
            # a1, L_ = self.ae1(x, l1, l2, l3, loss_)  # cell1
            # a11 = self.ae1(a1, l1, l2, l3, loss_, r_g=True)  # cell1
            # a111 = self.ae1(a11, l1, l2, l3, loss_)  # cell1
            #########dslc-dslc[2]-dslc[3]
            # a1, L_ = self.ae1(x, l1, l2, l3, loss_, r_g=True)  # cell1
            # a11 = self.ae11(a1, l1, l2, l3, loss_, r_g=True)  # cell1
            # a111 = self.ae111(a11, l1, l2, l3, loss_)  # cell1
            #########
            # a2, al3 = self.ae2(a1, al2, al2, loss_)  # cell2
            # a3 = self.ae3(a2, al3, loss_)  # cell3
            # a4 = self.ae4(a3)
            # a5 = self.ae5(a4)

        if self.training:
            if self.slType == 'triangle':  # 三角形网络结构，传统的U-net形式
                return a3
            elif self.slType == 'diamond':  # 钻石型结构，为了便于后面的特征迭代
                return a111

        else:
            if self.slType == 'triangle':  # 三角形网络结构，传统的U-net形式
                return a3, self.reconstruct(a3)
            elif self.slType == 'diamond':  # 钻石型结构，为了便于后面的特征迭代
                return a111

    # def reconstruct(self, x):
    #     if self.slType == 'triangle':  # 三角形网络结构，传统的U-net形式
    #         # a4_reconstruct = self.ae5.reconstruct(x)
    #         # a3_reconstruct = self.ae4.reconstruct(a4_reconstruct)
    #         a2_reconstruct = self.ae3.reconstruct(x)
    #         a1_reconstruct = self.ae2.reconstruct(a2_reconstruct)
    #         x_reconstruct = self.ae1.reconstruct(a1_reconstruct)
    #         return x_reconstruct
    #     elif self.slType == 'diamond':  # 钻石型结构，为了便于后面的特征迭代
    #         # a4_reconstruct = self.ae5.reconstruct(x)
    #         # a3_reconstruct = self.ae4.reconstruct(a4_reconstruct)
    #         # a2_reconstruct = self.ae3.reconstruct(x)
    #         ###dsc1-2-3
    #         # a11_reconstruct = self.ae111.reconstruct(x)
    #         # a1_reconstruct = self.ae11.reconstruct(a11_reconstruct)
    #         # x_reconstruct = self.ae1.reconstruct(x)
    #         ###dsc-dsc-dsc
    #         a11_reconstruct = self.ae1.reconstruct(x)
    #         a1_reconstruct = self.ae1.reconstruct(a11_reconstruct)
    #         x_reconstruct = self.ae1.reconstruct(a1_reconstruct)
    #         return x_reconstruct