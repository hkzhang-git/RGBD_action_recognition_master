import torch.nn as nn
import torch.nn.functional as F
import math


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv3d(inp, oup, (1, 3, 3), (1, stride, stride), 1, bias=False),
        nn.BatchNorm3d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv3d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm3d(oup),
        nn.ReLU6(inplace=True)
    )


def temporal_ave_pool():
    return nn.AvgPool3d(kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0))


class InvertedResidual_2d(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual_2d, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # pw
            nn.Conv3d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm3d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv3d(inp * expand_ratio, inp * expand_ratio, (1, 3, 3), (1, stride, stride), (0, 1, 1), groups=inp * expand_ratio, bias=False),
            nn.BatchNorm3d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv3d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm3d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class InvertedResidual_3d(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual_3d, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # pw
            nn.Conv3d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm3d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv3d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=False),
            nn.BatchNorm3d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv3d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm3d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class Mix_mobilenet_54(nn.Module):
    def __init__(self, n_class=101, in_channel=3, input_size=224, width_mult=1.):
        super(Mix_mobilenet_54, self).__init__()
        # setting of inverted residual blocks
        self.interverted_residual_setting = [
            # t, c, n, s, conv_d, temporal_pool
            [1, 16, 1, 1, 2, False],
            [6, 24, 2, 2, 2, True],
            [6, 32, 3, 2, 2, False],
            [6, 64, 4, 2, 2, True],
            [6, 96, 3, 1, 2, False],
            [6, 160, 3, 2, 3, False],
            [6, 320, 1, 1, 3, False],
        ]

        # building first layer
        assert input_size % 32 == 0
        input_channel = int(32 * width_mult)
        self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
        self.features = [conv_1x3x3_bn(in_channel, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s, d, p in self.interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    if d == 2:
                        self.features.append(InvertedResidual_2d(input_channel, output_channel, s, t))
                    else:
                        self.features.append(InvertedResidual_3d(input_channel, output_channel, s, t))
                else:
                    if d == 2:
                        self.features.append(InvertedResidual_2d(input_channel, output_channel, 1, t))
                    else:
                        self.features.append(InvertedResidual_3d(input_channel, output_channel, 1, t))
                input_channel = output_channel
            if p:
                self.features.append(temporal_ave_pool())
        # building last several layers
        self.features.append(conv_1x1x1_bn(input_channel, self.last_channel))
        self.features.append(nn.AdaptiveAvgPool3d((1, 1, 1)))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.last_channel, n_class),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.last_channel)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

