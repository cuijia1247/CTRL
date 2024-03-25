# The codes are used to build a classification through MLP in pytorch
# Author: cuijia1247
# Date: 2014-1-10
# version: 1.0

import torch
import torch.nn.functional as F

class MlpNet1(torch.nn.Module):
    def __int__(self, n1, n2, n3, n4, n5, n6):
        print('MlpNet')
        super(MlpNet, self).__init__()
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.n4 = n4
        self.n5 = n5
        self.n6 = n6
        self.l1 = torch.nn.Linear(self.n1, self.n2)
        self.l2 = torch.nn.Linear(self.n2, self.n3)
        self.l3 = torch.nn.Linear(self.n3, self.n4)
        self.l4 = torch.nn.Linear(self.n4, self.n5)
        self.l5 = torch.nn.Linear(self.n5, self.n6)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)


if __name__ == '__main__':
    pass