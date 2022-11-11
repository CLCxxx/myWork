import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
from utils.common import *
from torchvision import models as tv
from torch.nn.parameter import Parameter
import os

torch.cuda.current_device()

torch.cuda._initialized = True


class multi_VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, lam=1, lam_p=1):
        super(multi_VGGPerceptualLoss, self).__init__()
        self.loss_fn = VGGPerceptualLoss()
        self.lam = lam
        self.lam_p = lam_p

    def forward(self, out1, out2, out3, gt1, feature_layers=[2]):
        gt2 = F.interpolate(gt1, scale_factor=0.5, mode='bilinear', align_corners=False)
        gt3 = F.interpolate(gt1, scale_factor=0.25, mode='bilinear', align_corners=False)

        loss1 = self.lam_p * self.loss_fn(out1, gt1, feature_layers=feature_layers) + self.lam * F.l1_loss(out1, gt1)
        loss2 = self.lam_p * self.loss_fn(out2, gt2, feature_layers=feature_layers) + self.lam * F.l1_loss(out2, gt2)
        loss3 = self.lam_p * self.loss_fn(out3, gt3, feature_layers=feature_layers) + self.lam * F.l1_loss(out3, gt3)

        return loss1 + loss2 + loss3


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.resize = resize

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss


class net_loss(torch.nn.Module):
    def __init__(self, lam=1, lam_p=1, lam_c=1):
        super(net_loss, self).__init__()
        self.charbonnier_loss = CharbonnierLoss()
        self.vggloss = VGGPerceptualLoss()
        self.color_loss = ColorLoss()
        self.lam = lam
        self.lam_p = lam_p
        self.lam_c = lam_c

    def forward(self, out1, out2, out, gt1, feature_layers=[2]):
        gt2 = F.interpolate(gt1, scale_factor=0.5, mode='bilinear', align_corners=False)
        gt3 = F.interpolate(gt1, scale_factor=0.25, mode='bilinear', align_corners=False)

        loss1 = self.lam_p * self.vggloss(out1, gt3, feature_layers=feature_layers) + self.lam * self.charbonnier_loss(
            out1, gt3)
        loss2 = self.lam_p * self.vggloss(out2, gt2, feature_layers=feature_layers) + self.lam * self.charbonnier_loss(
            out2, gt2)
        loss3 = self.lam_p * self.vggloss(out, gt1, feature_layers=feature_layers) + self.lam * self.charbonnier_loss(
            out, gt1)

        # print("charbonnier_loss:%d vgg_loss: %d color_loss:%d", (charbonnier_loss.item(), vgg_loss.item(), color_loss.item()))
        return loss1 + loss2 + loss3


class net_loss_1(torch.nn.Module):
    def __init__(self, lam=1, lam_p=1, lam_c=1):
        super(net_loss_1, self).__init__()
        self.charbonnier_loss = CharbonnierLoss()
        self.vggloss = VGGPerceptualLoss()
        self.color_loss = ColorLoss()
        self.lam = lam
        self.lam_p = lam_p
        self.lam_c = lam_c

    def forward(self, out, gt1, feature_layers=[2]):
        loss = self.lam_p * self.vggloss(out, gt1, feature_layers=feature_layers) + self.lam * self.charbonnier_loss(
            out, gt1)

        # print("charbonnier_loss:%d vgg_loss: %d color_loss:%d", (charbonnier_loss.item(), vgg_loss.item(), color_loss.item()))
        return loss


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt(diff * diff + self.eps))
        return loss


class CharbonnierLoss2(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-6):
        super(CharbonnierLoss2, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt(diff * diff + self.eps))
        return loss


def get_gaussian_kernel(kernel_size=3, sigma=2, channels=3):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2 * variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, groups=channels,
                                bias=False, padding=kernel_size // 2)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter


class ColorLoss(nn.Module):
    def __init__(self):
        super(ColorLoss, self).__init__()
        self.blur_layer = get_gaussian_kernel()
        self.criterion1 = torch.nn.MSELoss()

    def forward(self, x1, x2):
        x1 = self.blur_layer(x1)
        x2 = self.blur_layer(x2)
        return self.criterion1(x1, x2)
