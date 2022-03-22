import config

import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        hp = config.Hyperparameter

        # c, 14, 14
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=hp.NUM_CHANNEL,
                out_channels=hp.NUM_D_FILTER,
                kernel_size=(4, 4),
                stride=(2, 2),
                padding=(1, 1)
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False)
        )
        # c, 7, 7
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=hp.NUM_D_FILTER,
                out_channels=hp.NUM_D_FILTER * 2,
                kernel_size=(4, 4),
                stride=(2, 2),
                padding=(1, 1)
            ),
            nn.BatchNorm2d(hp.NUM_D_FILTER * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False)
        )
        # c, 3, 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=hp.NUM_D_FILTER * 2,
                out_channels=hp.NUM_D_FILTER * 4,
                kernel_size=(4, 4),
                stride=(2, 2),
                padding=(1, 1)
            ),
            nn.BatchNorm2d(hp.NUM_D_FILTER * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False)
        )
        self.fc_dis = nn.Linear(3 * 3 * hp.NUM_D_FILTER * 4, 1)
        self.fc_aux = nn.Linear(3 * 3 * hp.NUM_D_FILTER * 4, hp.NUM_CLASS)
        self.softmax = nn.LogSoftmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        fc_dis = self.fc_dis(x)
        fc_aux = self.fc_aux(x)
        real_fake = self.sigmoid(fc_dis)
        classes = self.softmax(fc_aux)
        return real_fake, classes
