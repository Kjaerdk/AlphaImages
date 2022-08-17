# credit
# https://github.com/joansj/pytorch-intro/blob/master/src/cifar_model.py
# https://github.com/vdumoulin/welcome_tutorials/blob/master/pytorch/4.%20Image%20Classification%20with%20Convnets%20and%20ResNets.ipynb

import copy
import torch
import torch.nn as nn


class CNN1(nn.Module):
    """CNN Classifier"""

    def __init__(self, n_periods, height_pixels):
        super().__init__()

        self.n_periods = n_periods
        self.height_pixels = height_pixels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(4, 3), padding='valid', stride=(2, 1)), #, dilation=(2, 1)),  # dilation
            nn.LeakyReLU(),
            nn.BatchNorm2d(num_features=16),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Flatten()
        )
        self.head = nn.Sequential(
            nn.Linear(self._get_dim(), 1)
            #nn.LeakyReLU(),
            #nn.Dropout(p=0.5),
            #nn.Linear(100, 1)
        )

    def _get_dim(self):
        c_layer = copy.deepcopy(self.conv)
        dim = c_layer(torch.empty((1, 1, self.height_pixels, self.n_periods * 3))).shape[-1]
        return dim

    def forward(self, x):
        t = self.conv(x)
        return torch.sigmoid(self.head(t.view(t.size(0), -1)).squeeze())


class CNN2(nn.Module):
    """CNN Classifier"""

    def __init__(self, n_periods, height_pixels):
        super().__init__()

        self.n_periods = n_periods
        self.height_pixels = height_pixels
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(5, 3), padding='valid', stride=(1, 1), dilation=(1, 1)),  # dilation
            nn.LeakyReLU(),
            nn.BatchNorm2d(num_features=16),
            nn.MaxPool2d(kernel_size=(2, 1))
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2, 2), padding='valid', stride=(1, 1)),  # dilation
            nn.LeakyReLU(),
            nn.BatchNorm2d(num_features=32),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Flatten()
        )
        self.head = nn.Sequential(
            nn.Linear(self._get_dim(), 1)
            #nn.LeakyReLU(),
            #nn.Dropout(p=0.5),
            #nn.Linear(100, 1)
        )

    def _get_dim(self):
        c_layer1 = copy.deepcopy(self.conv1)
        c_layer2 = copy.deepcopy(self.conv2)

        dim = c_layer2(c_layer1(torch.empty((1, 1, self.height_pixels, self.n_periods * 3)))).shape[-1]
        return dim

    def forward(self, x):
        t = self.conv2(self.conv1(x))
        return torch.sigmoid(self.head(t.view(t.size(0), -1)).squeeze())


class CNN3(nn.Module):
    """CNN Classifier"""

    def __init__(self, n_periods, height_pixels):
        super(CNN3, self).__init__()

        self.n_periods = n_periods
        self.height_pixels = height_pixels

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 2), padding='valid', stride=(1, 1), dilation=(1, 1)),  # dilation
            nn.LeakyReLU(),
            nn.BatchNorm2d(num_features=16),
            nn.MaxPool2d(kernel_size=(2, 1))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2, 2), padding='valid', stride=(1, 1)),  # dilation
            nn.LeakyReLU(),
            nn.BatchNorm2d(num_features=32),
            nn.MaxPool2d(kernel_size=(2, 1))
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2, 1), padding='valid', stride=(1, 1)),  # dilation
            nn.LeakyReLU(),
            nn.BatchNorm2d(num_features=64),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Flatten()
        )
        self.head = nn.Sequential(
            nn.Linear(self._get_dim(), 1)
            #nn.LeakyReLU(),
            #nn.Dropout(p=0.5),
            #nn.Linear(100, 1)
        )

    def _get_dim(self):
        c_layer1 = copy.deepcopy(self.conv1)
        c_layer2 = copy.deepcopy(self.conv2)
        c_layer3 = copy.deepcopy(self.conv3)

        dim = c_layer3(c_layer2(c_layer1(torch.empty((1, 1, self.height_pixels, self.n_periods * 3))))).shape[-1]
        return dim

    def forward(self, x):
        t = self.conv3(self.conv2(self.conv1(x)))

        return torch.sigmoid(self.head(t.view(t.size(0), -1)).squeeze())
