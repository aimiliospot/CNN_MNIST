import torch.nn as nn
import torch.nn.functional as F


#useful article --> https://jhui.github.io/2018/02/09/PyTorch-neural-networks/
class CnnMnist(nn.Module):
    def __init__(self):
        super(CnnMnist, self).__init__()
        #kernel size refers to the matrix size of the filter in convolution
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding= 1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size= 3, padding= 1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size= 3, padding= 1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size= 3, padding= 1)
        self.fc1 = nn.Linear(7*7*64, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        #Convolution 1
        x = self.conv1(x)
        x = F.relu(x)

        #Convolution 2
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        #Convolution 3
        x = self.conv3(x)
        x = F.relu(x)

        #Convolution 4
        x = self.conv4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        #Fully Connected 1
        x = x.view(-1, 7*7*64)
        x = self.fc1(x)
        x = F.relu(x)

        #Fully Connected 2
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x






