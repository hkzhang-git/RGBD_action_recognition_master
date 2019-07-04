import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def channel_shuffle(x, groups):
    if groups == 1:
        return x
    else:
        batchsize, num_channels, frame_num, height, width = x.data.size()
        channels_per_group = num_channels // groups
        x = x.view(batchsize, groups, channels_per_group, frame_num, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batchsize, -1, frame_num, height, width)
        return x


class BasicConv3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              bias=False)  # verify bias false

        self.bn = nn.BatchNorm3d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Mixed_3b(nn.Module):
    def __init__(self):
        super(Mixed_3b, self).__init__()
        self.groups=[48, 96, 24, 24]
        self.index_0 = np.arange(0, 48)
        self.index_1 = np.arange(48, 144)
        self.index_2 = np.arange(144, 168)
        self.index_3 = np.arange(168, 192)
        self.branch0 = nn.Sequential(
            BasicConv3d(self.groups[0], 64, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(self.groups[1], 96, kernel_size=1, stride=1),
            BasicConv3d(96, 96, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)),
            BasicConv3d(96, 128, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0)),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(self.groups[2], 16, kernel_size=1, stride=1),
            BasicConv3d(16, 16, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)),
            BasicConv3d(16, 32, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0)),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(self.groups[3], 32, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x = channel_shuffle(x, 8)
        x0 = self.branch0(x[:, self.index_0])
        x1 = self.branch1(x[:, self.index_1])
        x2 = self.branch2(x[:, self.index_2])
        x3 = self.branch3(x[:, self.index_3])
        out = torch.cat((x0, x1, x2, x3), 1)

        return out


class Mixed_3c(nn.Module):
    def __init__(self):
        super(Mixed_3c, self).__init__()
        self.groups = [68, 102, 52, 34]
        self.index_0 = np.arange(0, 68)
        self.index_1 = np.arange(68, 170)
        self.index_2 = np.arange(170, 222)
        self.index_3 = np.arange(222, 256)
        self.branch0 = nn.Sequential(
            BasicConv3d(self.groups[0], 128, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(self.groups[1], 128, kernel_size=1, stride=1),
            BasicConv3d(128, 128, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)),
            BasicConv3d(128, 192, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0)),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(self.groups[2], 32, kernel_size=1, stride=1),
            BasicConv3d(32, 32, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)),
            BasicConv3d(32, 96, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0)),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(self.groups[3], 64, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x = channel_shuffle(x, 16)
        x0 = self.branch0(x[:, self.index_0])
        x1 = self.branch1(x[:, self.index_1])
        x2 = self.branch2(x[:, self.index_2])
        x3 = self.branch3(x[:, self.index_3])
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_4b(nn.Module):
    def __init__(self):
        super(Mixed_4b, self).__init__()
        self.groups = [180, 195, 45, 60]
        self.index_0 = np.arange(0, 180)
        self.index_1 = np.arange(180, 375)
        self.index_2 = np.arange(375, 420)
        self.index_3 = np.arange(420, 480)
        self.branch0 = nn.Sequential(
            BasicConv3d(self.groups[0], 192, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(self.groups[1], 96, kernel_size=1, stride=1),
            BasicConv3d(96, 96, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)),
            BasicConv3d(96, 208, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0)),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(self.groups[2], 16, kernel_size=1, stride=1),
            BasicConv3d(16, 16, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)),
            BasicConv3d(16, 48, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0)),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(self.groups[3], 64, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x = channel_shuffle(x, 16)
        x0 = self.branch0(x[:, self.index_0])
        x1 = self.branch1(x[:, self.index_1])
        x2 = self.branch2(x[:, self.index_2])
        x3 = self.branch3(x[:, self.index_3])
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_4c(nn.Module):
    def __init__(self):
        super(Mixed_4c, self).__init__()
        self.groups = [160, 224, 64, 64]
        self.index_0 = np.arange(0, 160)
        self.index_1 = np.arange(160, 384)
        self.index_2 = np.arange(384, 448)
        self.index_3 = np.arange(448, 512)

        self.branch0 = nn.Sequential(
            BasicConv3d(self.groups[0], 160, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(self.groups[1], 112, kernel_size=1, stride=1),
            BasicConv3d(112, 112, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)),
            BasicConv3d(112, 224, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0)),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(self.groups[2], 24, kernel_size=1, stride=1),
            BasicConv3d(24, 24, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)),
            BasicConv3d(24, 64, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0)),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(self.groups[3], 64, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x = channel_shuffle(x, 16)
        x0 = self.branch0(x[:, self.index_0])
        x1 = self.branch1(x[:, self.index_1])
        x2 = self.branch2(x[:, self.index_2])
        x3 = self.branch3(x[:, self.index_3])
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_4d(nn.Module):
    def __init__(self):
        super(Mixed_4d, self).__init__()
        self.groups = [128, 256, 64, 64]
        self.index_0 = np.arange(0, 128)
        self.index_1 = np.arange(128, 384)
        self.index_2 = np.arange(384, 448)
        self.index_3 = np.arange(448, 512)
        self.branch0 = nn.Sequential(
            BasicConv3d(self.groups[0], 128, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(self.groups[1], 128, kernel_size=1, stride=1),
            BasicConv3d(128, 128, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)),
            BasicConv3d(128, 256, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0)),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(self.groups[2], 24, kernel_size=1, stride=1),
            BasicConv3d(24, 24, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)),
            BasicConv3d(24, 64, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0)),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(self.groups[3], 64, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x = channel_shuffle(x, 16)
        x0 = self.branch0(x[:, self.index_0])
        x1 = self.branch1(x[:, self.index_1])
        x2 = self.branch2(x[:, self.index_2])
        x3 = self.branch3(x[:, self.index_3])
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_4e(nn.Module):
    def __init__(self):
        super(Mixed_4e, self).__init__()
        self.groups = [108, 280, 62, 62]
        self.index_0 = np.arange(0, 108)
        self.index_1 = np.arange(108, 388)
        self.index_2 = np.arange(388, 450)
        self.index_3 = np.arange(450, 512)
        self.branch0 = nn.Sequential(
            BasicConv3d(self.groups[0], 112, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(self.groups[1], 144, kernel_size=1, stride=1),
            BasicConv3d(144, 144, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)),
            BasicConv3d(144, 288, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0)),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(self.groups[2], 32, kernel_size=1, stride=1),
            BasicConv3d(32, 32, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)),
            BasicConv3d(32, 64, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0)),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(self.groups[3], 64, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x = channel_shuffle(x, 16)
        x0 = self.branch0(x[:, self.index_0])
        x1 = self.branch1(x[:, self.index_1])
        x2 = self.branch2(x[:, self.index_2])
        x3 = self.branch3(x[:, self.index_3])
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_4f(nn.Module):
    def __init__(self):
        super(Mixed_4f, self).__init__()
        self.groups = [162, 202, 82, 82]
        self.index_0 = np.arange(0, 162)
        self.index_1 = np.arange(162, 364)
        self.index_2 = np.arange(364, 446)
        self.index_3 = np.arange(446, 528)
        self.branch0 = nn.Sequential(
            BasicConv3d(self.groups[0], 256, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(self.groups[1], 160, kernel_size=1, stride=1),
            BasicConv3d(160, 160, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)),
            BasicConv3d(160, 320, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0)),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(self.groups[2], 32, kernel_size=1, stride=1),
            BasicConv3d(32, 32, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)),
            BasicConv3d(32, 128, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0)),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(self.groups[3], 128, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x = channel_shuffle(x, 16)
        x0 = self.branch0(x[:, self.index_0])
        x1 = self.branch1(x[:, self.index_1])
        x2 = self.branch2(x[:, self.index_2])
        x3 = self.branch3(x[:, self.index_3])
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_5b(nn.Module):
    def __init__(self):
        super(Mixed_5b, self).__init__()
        self.groups = [256, 320, 128, 128]
        self.index_0 = np.arange(0, 256)
        self.index_1 = np.arange(256, 576)
        self.index_2 = np.arange(576, 704)
        self.index_3 = np.arange(704, 832)
        self.branch0 = nn.Sequential(
            BasicConv3d(self.groups[0], 256, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(self.groups[1], 160, kernel_size=1, stride=1),
            BasicConv3d(160, 160, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)),
            BasicConv3d(160, 320, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0)),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(self.groups[2], 32, kernel_size=1, stride=1),
            BasicConv3d(32, 32, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)),
            BasicConv3d(32, 128, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0)),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(self.groups[3], 128, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x = channel_shuffle(x, 13)
        x0 = self.branch0(x[:, self.index_0])
        x1 = self.branch1(x[:, self.index_1])
        x2 = self.branch2(x[:, self.index_2])
        x3 = self.branch3(x[:, self.index_3])
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_5c(nn.Module):
    def __init__(self):
        super(Mixed_5c, self).__init__()
        self.groups = [312, 312, 104, 104]
        self.index_0 = np.arange(0, 312)
        self.index_1 = np.arange(312, 624)
        self.index_2 = np.arange(624, 728)
        self.index_3 = np.arange(728, 832)
        self.branch0 = nn.Sequential(
            BasicConv3d(self.groups[0], 384, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(self.groups[1], 192, kernel_size=1, stride=1),
            BasicConv3d(192, 192, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)),
            BasicConv3d(192, 384, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0)),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(self.groups[2], 48, kernel_size=1, stride=1),
            BasicConv3d(48, 48, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)),
            BasicConv3d(48, 128, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0)),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(self.groups[3], 128, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x = channel_shuffle(x, 8)
        x0 = self.branch0(x[:, self.index_0])
        x1 = self.branch1(x[:, self.index_1])
        x2 = self.branch2(x[:, self.index_2])
        x3 = self.branch3(x[:, self.index_3])
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class SS3D_P(nn.Module):

    def __init__(self, num_classes=400, input_channel=3, dropout_keep_prob=0.0):
        super(SS3D_P, self).__init__()
        self.features = nn.Sequential(
            BasicConv3d(input_channel, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3)), # (64, 32, 112, 112)
            BasicConv3d(64, 64, kernel_size=(7, 1, 1), stride=(2, 1, 1), padding=(3, 0, 0)),
            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),  # (64, 32, 56, 56)
            BasicConv3d(64, 64, kernel_size=1, stride=1), # (64, 32, 56, 56)
            BasicConv3d(64, 64, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)),  # (192, 32, 56, 56)
            BasicConv3d(64, 192, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0)),
            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),  # (192, 32, 28, 28)
            Mixed_3b(), # (256, 32, 28, 28)
            Mixed_3c(), # (480, 32, 28, 28)
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)), # (480, 16, 14, 14)
            Mixed_4b(),# (512, 16, 14, 14)
            Mixed_4c(),# (512, 16, 14, 14)
            Mixed_4d(),# (512, 16, 14, 14)
            Mixed_4e(),# (528, 16, 14, 14)
            Mixed_4f(),# (832, 16, 14, 14)
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0)), # (832, 8, 7, 7)
            Mixed_5b(), # (832, 8, 7, 7)
            Mixed_5c(), # (1024, 8, 7, 7)
            nn.AvgPool3d(kernel_size=(2, 7, 7), stride=1),# (1024, 8, 1, 1)
            nn.Dropout3d(dropout_keep_prob),
        )
        self.classifier = nn.Conv3d(1024, num_classes, kernel_size=1, stride=1, bias=True)

    def forward(self, x):
        x = self.features(x)
        logits = self.classifier(x)
        logits = logits.squeeze(3)
        logits = logits.squeeze(3)

        averaged_logits = torch.mean(logits, 2)
        predictions = F.log_softmax(averaged_logits, dim=1)
        return predictions


