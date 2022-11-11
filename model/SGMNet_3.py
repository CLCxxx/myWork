"""
Implementation of CLCNet for image demoireing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Snr_Fusion_Net(nn.Module):
    def __init__(self):
        super(Snr_Fusion_Net, self).__init__()

        self.full_encoder = full_encoder()
        self.full_decoder = full_decoder()
        self.sub_encoder = sub_encoder()
        self.sub_decoder = sub_decoder()

        self.conv_first_1 = conv_relu(in_channel=3, out_channel=32, kernel_size=3, padding=1, stride=1, dilation_rate=1)
        self.conv_first_2 = conv_relu(in_channel=32, out_channel=32, kernel_size=3, padding=2, stride=1,
                                      dilation_rate=2)
        self.fusion_module = fusion_module(in_channel=32)
        self.fusion_module1 = fusion_module(in_channel=128)
        self.fusion_module2 = fusion_module(in_channel=64)

    def forward(self, x, mask=None):
        l1_fea_1 = self.conv_first_1(x)
        l1_fea_1 = self.conv_first_2(l1_fea_1)

        x1, x2, x = self.full_encoder(l1_fea_1)
        x1, x2, x = self.full_decoder(x1, x2, x)

        y1, y2, y = self.sub_encoder(l1_fea_1)
        y1, y2, y = self.sub_decoder(y1, y2, y)

        # fea1 = self.fusion_module1(x1, y1, mask)
        # fea2 = self.fusion_module2(x2, y2, mask)
        fea = self.fusion_module(x, y, mask)

        return fea

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.02)
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)


class fusion_module(nn.Module):
    def __init__(self, in_channel):
        super(fusion_module, self).__init__()
        c = in_channel

        self.recon_module = Re_Block(in_channel=c, out_channel=c)

        self.down_conv1 = nn.Conv2d(in_channels=c * 2, out_channels=c * 4, kernel_size=4, stride=2, padding=1)
        self.down_conv2 = nn.Conv2d(in_channels=c, out_channels=c * 4, kernel_size=4, stride=2, padding=1)
        self.HR_conv = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, stride=1, padding=1)
        self.conv_last = conv(in_channel=c, out_channel=3, kernel_size=3, stride=1, padding=1, dilation_rate=1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, x1, x2, mask):
        h_feature = x1.shape[2]
        w_feature = x1.shape[3]
        channel = x1.shape[1]
        mask = F.interpolate(mask, size=[h_feature, w_feature], mode='nearest')
        mask_unfold = F.unfold(mask, kernel_size=4, dilation=1, stride=4, padding=0)
        mask_unfold = mask_unfold.permute(0, 2, 1)
        mask_unfold = torch.mean(mask_unfold, dim=2).unsqueeze(dim=-2)
        mask_unfold[mask_unfold <= 0.5] = 0.0
        mask = mask.repeat(1, channel, 1, 1)
        # fea = x2 * (1 - mask) + x1 * mask # 原计算方式
        fea = x1 * (1 - mask) + x2 * mask
        out_noise = self.recon_module(fea)
        out_noise = torch.cat([out_noise, fea], dim=1)
        out_noise = self.lrelu(self.pixel_shuffle(self.down_conv1(out_noise)))
        out_noise = self.lrelu(self.pixel_shuffle(self.down_conv2(out_noise)))
        out_noise = self.lrelu(self.HR_conv(out_noise))
        out_noise = self.conv_last(out_noise)
        return out_noise


class full_encoder_level(nn.Module):
    def __init__(self, in_channel, out_channel, eca_num):
        super(full_encoder_level, self).__init__()
        self.layer1 = conv_relu(in_channel=in_channel, out_channel=out_channel, kernel_size=3, padding=1, stride=1,
                                dilation_rate=1)
        self.layer2 = conv_relu(in_channel=in_channel + out_channel, out_channel=out_channel, kernel_size=3, padding=2,
                                stride=1, dilation_rate=2)
        self.layer3 = conv(in_channel=in_channel + out_channel * 2, out_channel=out_channel, kernel_size=3,
                           dilation_rate=1, padding=1)
        self.down1 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=2, padding=1)
        self.eca_module = nn.ModuleList()
        for _ in range(eca_num):
            eca_block = eca_layer(in_channel=out_channel, out_channel=out_channel)
            self.eca_module.append(eca_block)

    def forward(self, x):
        t = x
        _t = self.layer1(x)
        t = torch.cat([_t, t], dim=1)
        _t = self.layer2(t)
        t = torch.cat([_t, t], dim=1)
        t = self.layer3(t)
        t = self.down1(t)
        for l_block in self.eca_module:
            t = l_block(t)
        return t


class full_encoder(nn.Module):
    def __init__(self, in_channel=3):
        super(full_encoder, self).__init__()

        self.encoder_layer1 = full_encoder_level(in_channel=32, out_channel=64, eca_num=1)
        self.encoder_layer2 = full_encoder_level(in_channel=64, out_channel=128, eca_num=0)
        self.encoder_layer3 = full_encoder_level(in_channel=128, out_channel=256, eca_num=0)

    def forward(self, x):
        t1 = self.encoder_layer1(x)
        t2 = self.encoder_layer2(t1)
        t = self.encoder_layer3(t2)
        return t1, t2, t


class full_decoder_level(nn.Module):
    def __init__(self, in_channel, out_channel, eca_num):
        super(full_decoder_level, self).__init__()
        self.layer1 = conv_relu(in_channel=in_channel, out_channel=out_channel, kernel_size=3, padding=1, stride=1,
                                dilation_rate=1)
        self.layer2 = conv_relu(in_channel=in_channel + out_channel, out_channel=out_channel, kernel_size=3, padding=2,
                                stride=1, dilation_rate=2)
        self.layer3 = conv(in_channel=out_channel * 2 + in_channel, out_channel=out_channel, kernel_size=3,
                           dilation_rate=1)
        self.eca_module = nn.ModuleList()
        for _ in range(eca_num):
            eca_block = eca_layer(in_channel=out_channel, out_channel=out_channel)
            self.eca_module.append(eca_block)

    def forward(self, x, feat=True):
        t = x
        _t = self.layer1(x)
        t = torch.cat([_t, t], dim=1)
        _t = self.layer2(t)
        t = torch.cat([_t, t], dim=1)
        t = self.layer3(t)
        t = F.interpolate(t, scale_factor=2, mode='bilinear')

        for l_block in self.eca_module:
            t = l_block(t)

        return t


class CA(nn.Module):
    def __init__(self, in_channel):
        super(CA, self).__init__()
        self.mod = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channel, in_channel // 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channel // 16, in_channel, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.mod(x)


class full_decoder(nn.Module):
    def __init__(self, in_channel=256, out_channel=32):
        super(full_decoder, self).__init__()
        # self.conv1 = nn.Conv2d(256, 128, 3, 1, 1)
        # self.conv2 = nn.Conv2d(128, 64, 3, 1, 1)
        self.ca1 = CA(in_channel=256)
        self.ca2 = CA(in_channel=128)
        self.ca3 = CA(in_channel=64)
        self.full_decoder_layer1 = full_decoder_level(in_channel=256, out_channel=128, eca_num=0)
        self.full_decoder_layer2 = full_decoder_level(in_channel=128, out_channel=64, eca_num=0)
        self.full_decoder_layer3 = full_decoder_level(in_channel=64, out_channel=32, eca_num=2)

    def forward(self, x1, x2, x):
        x = self.ca1(x)
        t1 = self.full_decoder_layer1(x)
        x2 = self.ca2(x2)
        t1 += x2
        t2 = self.full_decoder_layer2(t1)
        x1 = self.ca3(x1)
        t2 += x1
        t = self.full_decoder_layer3(t2)
        return t1, t2, t


class sub_encoder_level(nn.Module):
    def __init__(self, in_channel, out_channel, spa_num):
        super(sub_encoder_level, self).__init__()
        self.layer1 = conv_relu(in_channel=in_channel, out_channel=out_channel, kernel_size=3, padding=1, stride=1,
                                dilation_rate=1)
        self.layer2 = conv_relu(in_channel=in_channel + out_channel, out_channel=out_channel, kernel_size=3, padding=2,
                                stride=1, dilation_rate=2)
        self.layer3 = conv(in_channel=in_channel + out_channel * 2, out_channel=out_channel, kernel_size=3,
                           dilation_rate=1)
        self.down1 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=2, padding=1)
        self.spa_module = nn.ModuleList()
        for _ in range(spa_num):
            spa_block = spa_layer(in_channel=out_channel, out_channel=out_channel)
            self.spa_module.append(spa_block)

    def forward(self, x):
        t = x
        _t = self.layer1(x)
        t = torch.cat([_t, t], dim=1)
        _t = self.layer2(t)
        t = torch.cat([_t, t], dim=1)
        t = self.layer3(t)
        t = self.down1(t)
        for l_block in self.spa_module:
            t = l_block(t)

        return t


class sub_encoder(nn.Module):
    def __init__(self):
        super(sub_encoder, self).__init__()

        self.sub_encoder_layer1 = sub_encoder_level(in_channel=32, out_channel=64, spa_num=1)
        self.sub_encoder_layer2 = sub_encoder_level(in_channel=64, out_channel=128, spa_num=0)
        self.sub_encoder_layer3 = sub_encoder_level(in_channel=128, out_channel=256, spa_num=0)

    def forward(self, x):
        t1 = self.sub_encoder_layer1(x)
        t2 = self.sub_encoder_layer2(t1)
        t = self.sub_encoder_layer3(t2)

        return t1, t2, t


class sub_decoder_layer(nn.Module):
    def __init__(self, in_channel, out_channel, spa_num):
        super(sub_decoder_layer, self).__init__()
        self.layer1 = conv_relu(in_channel=in_channel, out_channel=out_channel, kernel_size=3, padding=1, stride=1,
                                dilation_rate=1)
        self.layer2 = conv_relu(in_channel=in_channel + out_channel, out_channel=out_channel, kernel_size=3, padding=2,
                                stride=1, dilation_rate=2)
        self.layer3 = conv(in_channel=in_channel + out_channel * 2, out_channel=out_channel, kernel_size=3,
                           dilation_rate=1)
        self.spa_module = nn.ModuleList()
        for _ in range(spa_num):
            spa_block = spa_layer(in_channel=out_channel, out_channel=out_channel)
            self.spa_module.append(spa_block)

    def forward(self, x):
        t = x
        _t = self.layer1(x)
        t = torch.cat([_t, t], dim=1)
        _t = self.layer2(t)
        t = torch.cat([_t, t], dim=1)
        t = self.layer3(t)
        t = F.interpolate(t, scale_factor=2, mode='bilinear')
        for l_block in self.spa_module:
            t = l_block(t)

        return t


class sub_decoder(nn.Module):
    def __init__(self, in_channel=256):
        super(sub_decoder, self).__init__()

        self.ca1 = CA(in_channel=256)
        self.ca2 = CA(in_channel=128)
        self.ca3 = CA(in_channel=64)

        self.sub_decoder_layer1 = sub_decoder_layer(in_channel=256, out_channel=128, spa_num=0)
        self.sub_decoder_layer2 = sub_decoder_layer(in_channel=128, out_channel=64, spa_num=0)
        self.sub_decoder_layer3 = sub_decoder_layer(in_channel=64, out_channel=32, spa_num=2)

    def forward(self, x1, x2, x):
        x = self.ca1(x)
        t1 = self.sub_decoder_layer1(x)
        x2 = self.ca2(x2)
        t1 += x2
        t2 = self.sub_decoder_layer2(t1)
        x1 = self.ca3(x1)
        t2 += x1
        t = self.sub_decoder_layer3(t2)
        return t1, t2, t


class Re_Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Re_Block, self).__init__()
        c = in_channel
        self.layer1 = conv_relu(in_channel=c, out_channel=c, kernel_size=3, dilation_rate=1, padding=1)
        self.layer2 = conv_relu(in_channel=c * 2, out_channel=c, kernel_size=3, dilation_rate=2, padding=2)
        self.layer3 = conv(c * 3, c, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        t = x
        _t = self.layer1(x)
        t = torch.cat([_t, t], dim=1)
        _t = self.layer2(t)
        t = torch.cat([_t, t], dim=1)
        t = self.layer3(t)
        return t


class eca_layer(nn.Module):
    """Constructs a ECA module.
       Args:
           channel: Number of channels of the input feature map
           k_size: Adaptive selection of kernel size
       """

    def __init__(self, in_channel, out_channel, k_size=3):
        super(eca_layer, self).__init__()

        self.basic_block = Re_Block(in_channel=in_channel, out_channel=out_channel)
        self.basic_block2 = Re_Block(in_channel=in_channel, out_channel=out_channel)
        self.basic_block4 = Re_Block(in_channel=in_channel, out_channel=out_channel)

        self.fusion = Eca_Fusion(3 * in_channel)

    def forward(self, x):
        x_0 = x
        x_2 = F.interpolate(x, scale_factor=0.5, mode='bilinear')
        x_4 = F.interpolate(x, scale_factor=0.25, mode='bilinear')

        y_0 = self.basic_block(x_0)
        y_2 = self.basic_block2(x_2)
        y_4 = self.basic_block4(x_4)

        y_2 = F.interpolate(y_2, scale_factor=2, mode='bilinear')
        y_4 = F.interpolate(y_4, scale_factor=4, mode='bilinear')

        y = self.fusion(y_0, y_2, y_4)

        return x + y


class Eca_Fusion(nn.Module):
    def __init__(self, in_chnls, ratio=4):
        super(Eca_Fusion, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.conv0 = nn.Conv1d(1, 1, kernel_size=3, padding=(3 - 1) // 2, bias=False)
        self.conv2 = nn.Conv1d(1, 1, kernel_size=3, padding=(3 - 1) // 2, bias=False)
        self.conv4 = nn.Conv1d(1, 1, kernel_size=3, padding=(3 - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.compress1 = nn.Conv2d(in_chnls, in_chnls // ratio, 1, 1, 0)
        self.compress2 = nn.Conv2d(in_chnls // ratio, in_chnls // ratio, 1, 1, 0)
        self.excitation = nn.Conv2d(in_chnls // ratio, in_chnls, 1, 1, 0)

    def forward(self, x0, x2, x4):
        out0 = self.squeeze(x0)
        out2 = self.squeeze(x2)
        out4 = self.squeeze(x4)

        out0 = self.conv0(out0.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        out2 = self.conv2(out2.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        out4 = self.conv4(out4.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        out = torch.cat([out0, out2, out4], dim=1)
        out = self.compress1(out)
        out = F.relu(out)
        out = self.compress2(out)
        out = F.relu(out)
        out = self.excitation(out)
        out = F.sigmoid(out)
        w0, w2, w4 = torch.chunk(out, 3, dim=1)
        x = x0 * w0 + x2 * w2 + x4 * w4

        return x


class spa_layer(nn.Module):
    def __init__(self, in_channel, out_channel, k_size=3):
        super(spa_layer, self).__init__()

        self.basic_block = Re_Block(in_channel=in_channel, out_channel=out_channel)
        self.basic_block2 = Re_Block(in_channel=in_channel, out_channel=out_channel)
        self.basic_block4 = Re_Block(in_channel=in_channel, out_channel=out_channel)

        self.fusion = Spa_Fusion()

    def forward(self, x):
        x_0 = x
        x_2 = F.interpolate(x, scale_factor=0.5, mode='bilinear')
        x_4 = F.interpolate(x, scale_factor=0.25, mode='bilinear')

        y_0 = self.basic_block(x_0)
        y_2 = self.basic_block2(x_2)
        y_4 = self.basic_block4(x_4)

        y_2 = F.interpolate(y_2, scale_factor=2, mode='bilinear')
        y_4 = F.interpolate(y_4, scale_factor=4, mode='bilinear')

        y = self.fusion(y_0, y_2, y_4)

        return x + y


class Spa_Fusion(nn.Module):
    def __init__(self, in_chnls=3, ratio=1):
        super(Spa_Fusion, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, 3, padding=1, bias=False, dilation=1)
        self.conv2 = nn.Conv2d(2, 1, 3, padding=1, bias=False, dilation=1)
        self.conv3 = nn.Conv2d(2, 1, 3, padding=1, bias=False, dilation=1)
        self.sigmoid = nn.Sigmoid()

        self.compress1 = nn.Conv2d(in_chnls, in_chnls // ratio, 1, 1, 0)
        self.compress2 = nn.Conv2d(in_chnls // ratio, in_chnls // ratio, 1, 1, 0)
        self.excitation = nn.Conv2d(in_chnls // ratio, in_chnls, 1, 1, 0)

    def forward(self, x0, x2, x4):
        x0_avg = torch.mean(x0, dim=1, keepdim=True)
        x0_max, _ = torch.max(x0, dim=1, keepdim=True)  # 30,1,50,30
        out0 = torch.cat([x0_avg, x0_max], dim=1)
        x2_avg = torch.mean(x2, dim=1, keepdim=True)
        x2_max, _ = torch.max(x2, dim=1, keepdim=True)  # 30,1,50,30
        out2 = torch.cat([x2_avg, x2_max], dim=1)
        x4_avg = torch.mean(x4, dim=1, keepdim=True)
        x4_max, _ = torch.max(x4, dim=1, keepdim=True)  # 30,1,50,30
        out4 = torch.cat([x4_avg, x4_max], dim=1)

        out0 = self.conv1(out0)
        out2 = self.conv2(out2)
        out4 = self.conv3(out4)

        out = torch.cat([out0, out2, out4], dim=1)
        out = self.compress1(out)
        out = F.relu(out)
        out = self.compress2(out)
        out = F.relu(out)
        out = self.excitation(out)
        out = F.sigmoid(out)
        w0, w2, w4 = torch.chunk(out, 3, dim=1)
        x = x0.mul(w0) + x2.mul(w2) + x4.mul(w4)

        return x


class conv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, dilation_rate=1, padding=1, stride=1):
        super(conv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=True, dilation=dilation_rate)

    def forward(self, x_input):
        out = self.conv(x_input)
        return out


class conv_relu(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, dilation_rate=1, padding=1, stride=1):
        super(conv_relu, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride,
                      padding=padding, bias=True, dilation=dilation_rate),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_input):
        out = self.conv(x_input)
        return out


if __name__ == '__main__':
    a = torch.rand(2, 3, 128, 128)
    b = torch.rand(2, 1, 128, 128)
    t1 = torch.rand(2, 32, 128, 128)
    t2 = torch.rand(2, 64, 64, 64)
    t3 = torch.rand(2, 128, 32, 32)
    layer = Snr_Fusion_Net()
    res = layer(a, b)
    print(res[0].size())
    print(res[1].size())
    print(res[2].size())
    # encoder_net = full_encoder()
    # decoder_net = full_decoder()
    # b = torch.rand(2,3,256,256)
    # sub_encoder_net = sub_encoder()
    # sub_decoder_net = sub_decoder()
    # net = my_net()
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
    # print(net.forward(a).size())
