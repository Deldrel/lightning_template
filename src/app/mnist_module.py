import torch.nn as nn
import torch.optim as optim

from app.base_module import BaseModule
from app.settings import AppSettings


class MNISTModule(BaseModule):
    def __init__(self, settings: AppSettings):
        super(MNISTModule, self).__init__(settings)

        # Make model architecture configurable
        conv1_channels = 32
        conv2_channels = 64
        fc1_units = 256
        fc2_units = 128
        dropout_rate = 0.1

        self.conv_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=conv1_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),  # 28x28x1 -> 28x28x32
            nn.ReLU(),
            nn.BatchNorm2d(conv1_channels),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28x28x32 -> 14x14x32
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(
                in_channels=conv1_channels,
                out_channels=conv2_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),  # 14x14x32 -> 14x14x64
            nn.ReLU(),
            nn.BatchNorm2d(conv2_channels),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 14x14x64 -> 7x7x64
            nn.Dropout2d(dropout_rate),
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(7 * 7 * conv2_channels, fc1_units),
            nn.ReLU(),
            nn.BatchNorm1d(fc1_units),
            nn.Dropout(dropout_rate),
            nn.Linear(fc1_units, fc2_units),
            nn.ReLU(),
            nn.BatchNorm1d(fc2_units),
            nn.Dropout(dropout_rate),
            nn.Linear(fc2_units, 10),
        )

    def get_loss_func(self):
        return nn.CrossEntropyLoss()

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.settings.train.learning_rate,
            weight_decay=self.settings.train.weight_decay,
        )
        return optimizer

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x
