import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, num_of_classes: int, input_size: tuple):
        super(CNN, self).__init__()
        self.size = input_size

        self.conv_layer1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_layer2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_layer3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_layer4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        def _get_flattened_size(self):
            with torch.no_grad():
                dummy_input = torch.zeros(1, *self.size)
                out = self.conv_layer1(dummy_input)
                out = self.pool1(out)
                out = self.conv_layer2(out)
                out = self.pool2(out)
                out = self.conv_layer3(out)
                out = self.pool3(out)
                out = self.conv_layer4(out)
                out = self.pool4(out)
                flattened_size = out.view(1, -1).size(1)
            return flattened_size

        self.fc1 = nn.Linear(_get_flattened_size(self), 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, num_of_classes)


    def _get_flattened_size(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *self.size)
            out = self.conv_layer1(dummy_input)
            out = self.pool1(out)
            out = self.conv_layer2(out)
            out = self.pool2(out)
            out = self.conv_layer3(out)
            out = self.pool3(out)
            out = self.conv_layer4(out)
            out = self.pool4(out)
            flattened_size = out.view(1, -1).size(1)
        return flattened_size



    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.pool1(out)

        out = self.conv_layer2(out)
        out = self.pool2(out)

        out = self.conv_layer3(out)
        out = self.pool3(out)

        out = self.conv_layer4(out)
        out = self.pool4(out)

        out = out.reshape(out.size(0), -1)

        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out
