"""
Implementation of CLCNet for image demoireing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class my_net(nn.Module):
    def __init__(self):
        super(my_net, self).__init__()
        self.full_encoder = full_encoder()
        self.full_decoder = full_decoder()
        self.sub_encoder = full_encoder()
        self.sub_decoder = sub_decoder()

    def forward(self, x):
        x1 = self.sub_encoder(x)
        x1 = self.full_decoder(x1)
        x2 = self.sub_encoder(x)
        x2 = self.sub_decoder(x2)
        return x1+x2
class eca_layer(nn.Module):
    """Constructs a ECA module.
       Args:
           channel: Number of channels of the input feature map
           k_size: Adaptive selection of kernel size
       """

    def __init__(self, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

class spa_layer(nn.Module):
    def __init__(self, kernel_size=7):
        super(spa_layer, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # x.size() 30,40,50,30
        tmp = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 30,1,50,30
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)  # 30,1,50,30
        return tmp.mul(self.sigmoid(x))  # 30,1,50,30

class full_encoder(nn.Module):
    def __init__(self):
        super(full_encoder, self).__init__()
        self.ca_net = eca_layer()
        # Encoder Layer1
        # 3*256*256
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU()
        )

        # Encoder Layer2
        # 64*128*128
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        )
        self.layer7 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        )
        self.layer8 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU()
        )

        # Encoder Layer3
        # 128*64*64
        self.layer9 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        )
        self.layer10 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        )
        self.layer11 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        )
        self.layer12 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU()
        )
        # 256*32*32

    def forward(self, x):
        # Encoder Layer1
        x = self.layer1(x)
        x = self.layer2(x) + x
        x = self.layer3(x) + x
        x = self.layer4(x)
        x = self.ca_net(x)
        # Encoder Layer2
        x = self.layer5(x)
        x = self.layer6(x) + x
        x = self.layer7(x) + x
        x = self.layer8(x)
        x = self.ca_net(x)
        # Encoder Layer3
        x = self.layer9(x)
        x = self.layer10(x) + x
        x = self.layer11(x) + x
        x = self.layer12(x)
        x = self.ca_net(x)
        return x

class full_decoder(nn.Module):
    def __init__(self):
        super(full_decoder, self).__init__()
        self.ca_net = eca_layer()
        # Decoder Layer1
        # 256*32*32
        self.layer1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        )
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU()
        )

        # Decoder Layer2
        # 128*64*64
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        )
        self.layer7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        )
        self.layer8 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU()
        )

        # Encoder Layer3
        # 64*128*128
        self.layer9 = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        )
        self.layer10 = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        )
        self.layer11 = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        )
        self.layer12 = nn.Sequential(
            nn.ConvTranspose2d(3, 3, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU()
        )
        # 3*256*256

    def forward(self, x):
        # decoder Layer1
        x = self.layer1(x)
        x = self.layer2(x) + x
        x = self.layer3(x) + x
        x = self.layer4(x)
        x = self.ca_net(x)
        # decoder Layer2
        x = self.layer5(x)
        x = self.layer6(x) + x
        x = self.layer7(x) + x
        x = self.layer8(x)
        x = self.ca_net(x)
        # decoder Layer3
        x = self.layer9(x)
        x = self.layer10(x) + x
        x = self.layer11(x) + x
        x = self.layer12(x)
        x = self.ca_net(x)
        return x

class sub_encoder(nn.Module):
    def __init__(self):
        super(sub_encoder, self).__init__()
        self.spa_net = spa_layer()
        # Encoder Layer1
        # 3*256*256
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU()
        )

        # Encoder Layer2
        # 64*128*128
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        )
        self.layer7 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        )
        self.layer8 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU()
        )

        # Encoder Layer3
        # 128*64*64
        self.layer9 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        )
        self.layer10 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        )
        self.layer11 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        )
        self.layer12 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU()
        )
        # 256*32*32

    def forward(self, x):
        # Encoder Layer1
        x = self.layer1(x)
        x = self.layer2(x) + x
        x = self.layer3(x) + x
        x = self.layer4(x)
        H = x.size(2)
        W = x.size(3)
        x_1 = x[:, :, 0:int(H / 2), :]
        x_2 = x[:, :, int(H / 2):H, :]
        x_lv1_1 = x_1[:, :, :, 0:int(W / 2)]
        x_lv1_2 = x_1[:, :, :, int(W / 2):W]
        x_lv1_3 = x_2[:, :, :, 0:int(W / 2)]
        x_lv1_4 = x_2[:, :, :, int(W / 2):W]
        x_lv1_1 = self.spa_net(x_lv1_1)
        x_lv1_2 = self.spa_net(x_lv1_2)
        x_lv1_3 = self.spa_net(x_lv1_3)
        x_lv1_4 = self.spa_net(x_lv1_4)
        x_lv1_top = torch.cat((x_lv1_1, x_lv1_2), 3)
        x_lv1_bot = torch.cat((x_lv1_3, x_lv1_4), 3)
        x_lv1 = torch.cat((x_lv1_top,x_lv1_bot), 2)
        # Encoder Layer2
        x = self.layer5(x_lv1)
        x = self.layer6(x) + x
        x = self.layer7(x) + x
        x = self.layer8(x)
        H = x.size(2)
        W = x.size(3)
        x_1 = x[:, :, 0:int(H / 2), :]
        x_2 = x[:, :, int(H / 2):H, :]
        x_lv3_1 = x_1[:, :, :, 0:int(W / 2)]
        x_lv3_2 = x_1[:, :, :, int(W / 2):W]
        x_lv3_3 = x_2[:, :, :, 0:int(W / 2)]
        x_lv3_4 = x_2[:, :, :, int(W / 2):W]
        x_lv3_1 = self.spa_net(x_lv3_1)
        x_lv3_2 = self.spa_net(x_lv3_2)
        x_lv3_3 = self.spa_net(x_lv3_3)
        x_lv3_4 = self.spa_net(x_lv3_4)
        x_lv3_top = torch.cat((x_lv3_1, x_lv3_2), 3)
        x_lv3_bot = torch.cat((x_lv3_3, x_lv3_4), 3)
        x_lv3 = torch.cat((x_lv3_top, x_lv3_bot), 2)
        # Encoder Layer3
        x = self.layer9(x_lv3)
        x = self.layer10(x) + x
        x = self.layer11(x) + x
        x = self.layer12(x)
        H = x.size(2)
        W = x.size(3)
        x_1 = x[:, :, 0:int(H / 2), :]
        x_2 = x[:, :, int(H / 2):H, :]
        x_lv3_1 = x_1[:, :, :, 0:int(W / 2)]
        x_lv3_2 = x_1[:, :, :, int(W / 2):W]
        x_lv3_3 = x_2[:, :, :, 0:int(W / 2)]
        x_lv3_4 = x_2[:, :, :, int(W / 2):W]
        x_lv3_1 = self.spa_net(x_lv3_1)
        x_lv3_2 = self.spa_net(x_lv3_2)
        x_lv3_3 = self.spa_net(x_lv3_3)
        x_lv3_4 = self.spa_net(x_lv3_4)
        x_lv3_top = torch.cat((x_lv3_1, x_lv3_2), 3)
        x_lv3_bot = torch.cat((x_lv3_3, x_lv3_4), 3)
        x_lv3 = torch.cat((x_lv3_top, x_lv3_bot), 2)
        return x_lv3

class sub_decoder(nn.Module):
    def __init__(self):
        super(sub_decoder, self).__init__()
        self.spa_net = spa_layer()
        # Decoder Layer1
        # 256*32*32
        self.layer1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        )
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU()
        )

        # Decoder Layer2
        # 128*64*64
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        )
        self.layer7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        )
        self.layer8 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU()
        )

        # Encoder Layer3
        # 64*128*128
        self.layer9 = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        )
        self.layer10 = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        )
        self.layer11 = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        )
        self.layer12 = nn.Sequential(
            nn.ConvTranspose2d(3, 3, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU()
        )
        # 3*256*256

    def forward(self, x):
        # decoder Layer1
        x = self.layer1(x)
        x = self.layer2(x) + x
        x = self.layer3(x) + x
        x = self.layer4(x)
        H = x.size(2)
        W = x.size(3)
        x_1 = x[:, :, 0:int(H / 2), :]
        x_2 = x[:, :, int(H / 2):H, :]
        x_lv1_1 = x_1[:, :, :, 0:int(W / 2)]
        x_lv1_2 = x_1[:, :, :, int(W / 2):W]
        x_lv1_3 = x_2[:, :, :, 0:int(W / 2)]
        x_lv1_4 = x_2[:, :, :, int(W / 2):W]
        x_lv1_1 = self.spa_net(x_lv1_1)
        x_lv1_2 = self.spa_net(x_lv1_2)
        x_lv1_3 = self.spa_net(x_lv1_3)
        x_lv1_4 = self.spa_net(x_lv1_4)
        x_lv1_top = torch.cat((x_lv1_1, x_lv1_2), 3)
        x_lv1_bot = torch.cat((x_lv1_3, x_lv1_4), 3)
        x_lv1 = torch.cat((x_lv1_top, x_lv1_bot), 2)
        # decoder Layer2
        x = self.layer5(x_lv1)
        x = self.layer6(x) + x
        x = self.layer7(x) + x
        x = self.layer8(x)
        H = x.size(2)
        W = x.size(3)
        x_1 = x[:, :, 0:int(H / 2), :]
        x_2 = x[:, :, int(H / 2):H, :]
        x_lv3_1 = x_1[:, :, :, 0:int(W / 2)]
        x_lv3_2 = x_1[:, :, :, int(W / 2):W]
        x_lv3_3 = x_2[:, :, :, 0:int(W / 2)]
        x_lv3_4 = x_2[:, :, :, int(W / 2):W]
        x_lv3_1 = self.spa_net(x_lv3_1)
        x_lv3_2 = self.spa_net(x_lv3_2)
        x_lv3_3 = self.spa_net(x_lv3_3)
        x_lv3_4 = self.spa_net(x_lv3_4)
        x_lv3_top = torch.cat((x_lv3_1, x_lv3_2), 3)
        x_lv3_bot = torch.cat((x_lv3_3, x_lv3_4), 3)
        x_lv3 = torch.cat((x_lv3_top, x_lv3_bot), 2)
        # decoder Layer3
        x = self.layer9(x_lv3)
        x = self.layer10(x) + x
        x = self.layer11(x) + x
        x = self.layer12(x)
        H = x.size(2)
        W = x.size(3)
        x_1 = x[:, :, 0:int(H / 2), :]
        x_2 = x[:, :, int(H / 2):H, :]
        x_lv3_1 = x_1[:, :, :, 0:int(W / 2)]
        x_lv3_2 = x_1[:, :, :, int(W / 2):W]
        x_lv3_3 = x_2[:, :, :, 0:int(W / 2)]
        x_lv3_4 = x_2[:, :, :, int(W / 2):W]
        x_lv3_1 = self.spa_net(x_lv3_1)
        x_lv3_2 = self.spa_net(x_lv3_2)
        x_lv3_3 = self.spa_net(x_lv3_3)
        x_lv3_4 = self.spa_net(x_lv3_4)
        x_lv3_top = torch.cat((x_lv3_1, x_lv3_2), 3)
        x_lv3_bot = torch.cat((x_lv3_3, x_lv3_4), 3)
        x_lv3 = torch.cat((x_lv3_top, x_lv3_bot), 2)
        return x_lv3


if __name__ == '__main__':
    a = torch.rand(2, 3, 256, 256)
    encoder_net = full_encoder()
    decoder_net = full_decoder()
    b = torch.rand(2,3,256,256)
    sub_encoder_net = sub_encoder()
    sub_decoder_net = sub_decoder()
    net = my_net()
    # net = eca_layer(channel=3)
    # avg = nn.AdaptiveAvgPool2d(1)(a)
    # print(avg.size())
    # conv = nn.Conv1d(1, 1, kernel_size=3, padding=(3 - 1) // 2, bias=False)
    # conv1 = conv((avg.squeeze(-1).transpose(-1, -2))).transpose(-1, -2).unsqueeze(-1)
    # print(conv1.size())
    # print(nn.Sigmoid()(conv1))
    # a = encoder_net.forward(a)
    # print(a.size())
    # print(decoder_net.forward(a).size())
    # b = sub_encoder_net(b)
    # print(b.size())
    # print(sub_decoder_net(b).size())
    print(net.forward(a).size())
