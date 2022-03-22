import config

import torch.nn as nn


# Generate 28 * 28 image with 10 class (0~9)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        hp = config.Hyperparameter

        # c, 4, 4
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=hp.NUM_NOISE + hp.NUM_CLASS,
                out_channels=hp.NUM_G_FILTER * 8,
                kernel_size=(4, 4),
                stride=(1, 1)
            ),
            nn.ReLU(True)
        )
        # c, 8, 8
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=hp.NUM_G_FILTER * 8,
                out_channels=hp.NUM_G_FILTER * 4,
                kernel_size=(4, 4),
                stride=(2, 2),
                padding=(1, 1),
                bias=False
            ),
            nn.BatchNorm2d(hp.NUM_G_FILTER * 4),
            nn.ReLU(True)
        )
        # c, 16, 16
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=hp.NUM_G_FILTER * 4,
                out_channels=hp.NUM_G_FILTER * 2,
                kernel_size=(4, 4),
                stride=(2, 2),
                padding=(1, 1),
                bias=False
            ),
            nn.BatchNorm2d(hp.NUM_G_FILTER * 2),
            nn.ReLU(True)
        )
        # c, 32, 32
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=hp.NUM_G_FILTER * 2,
                out_channels=hp.NUM_G_FILTER,
                kernel_size=(4, 4),
                stride=(2, 2),
                padding=(1, 1),
                bias=False
            ),
            nn.BatchNorm2d(hp.NUM_G_FILTER),
            nn.ReLU(True)
        )
        # c, 28, 28
        self.deconv5 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=hp.NUM_G_FILTER,
                out_channels=hp.NUM_CHANNEL,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(2, 2),
                bias=False
            ),
            nn.Tanh()
        )

    def forward(self, x):
        x = x[:, :, None, None]
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.deconv5(x)
        return x
