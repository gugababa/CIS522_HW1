import torch
import torch.nn as nn


class Model(torch.nn.Module):
    """This class instantiates and propagates a convolutional neural network (CNN) built for the CIFAR dataset.
    3 convolutional, 3 batch norm, and 2 linear layers.
    """

    def __init__(self, num_channels: int, num_classes: int) -> None:
        """This constructor initializes the architecture of the CNN"""
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=num_channels, out_channels=3, kernel_size=3, padding=0
        )
        self.relu1 = nn.ReLU()
        self.batch1 = nn.BatchNorm2d(3)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, padding=0)
        self.relu2 = nn.ReLU()
        self.batch2 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3, padding=0)
        self.relu3 = nn.ReLU()
        self.batch3 = nn.BatchNorm2d(12)
        self.fc1 = nn.Linear(in_features=12 * 12 * 12, out_features=512)
        self.relu5 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=512, out_features=128)
        self.relu6 = nn.ReLU()
        self.fc3 = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The forward method propagates the CNN, allowing us to attain our final classification predictions"""
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.batch1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.batch2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.batch3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu5(x)
        x = self.fc2(x)
        x = self.relu6(x)
        x = self.fc3(x)
        return x
