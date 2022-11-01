import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from utils.loss_util import *
from utils.common import *
from torch.nn.parameter import Parameter
from functools import partial
import time

def model_fn_decorator(loss_fn, device, mode='train'):
    def test_model_fn(args, data, model, save_path, compute_metrics):
        # prepare input and forward
        number = data['number']
        cur_psnr = 0.0
        cur_ssim = 0.0
        cur_lpips = 0.0

        in_img = data['in_img'].to(device)
        label = data['label'].to(device)
        img_nf = data['img_nf'].to(device)
        # print(in_img.size())
        # print(img_nf.size())
        old_img = in_img
        old_img = old_img[:, 0:1, :, :] * 0.299 + old_img[:, 1:2, :, :] * 0.587 + old_img[:, 2:3, :, :] * 0.114
        de_moire_img = img_nf
        de_moire_img = de_moire_img[:, 0:1, :, :] * 0.299 + de_moire_img[:, 1:2, :, :] * 0.587 + de_moire_img[:, 2:3, :,
                                                                                                 :] * 0.114
        noise = torch.abs(old_img - de_moire_img)

        mask = torch.div(de_moire_img, noise + 0.0001)

        batch_size = mask.shape[0]
        height = mask.shape[2]
        width = mask.shape[3]
        mask_max = torch.max(mask.view(batch_size, -1), dim=1)[0]
        mask_max = mask_max.view(batch_size, 1, 1, 1)
        mask_max = mask_max.repeat(1, 1, height, width)
        mask = mask * 1.0 / (mask_max + 0.0001)

        mask = torch.clamp(mask, min=0, max=1.0)
        mask = mask.float()
        b, c, h, w = in_img.size()

        # pad image such that the resolution is a multiple of 32
        w_pad = (math.ceil(w/32)*32 - w) // 2
        h_pad = (math.ceil(h/32)*32 - h) // 2
        w_odd_pad = w_pad
        h_odd_pad = h_pad
        if w % 2 == 1:
            w_odd_pad += 1
        if h % 2 == 1:
            h_odd_pad += 1

        in_img = img_pad(in_img, w_pad=w_pad, h_pad=h_pad, w_odd_pad=w_odd_pad, h_odd_pad=h_odd_pad)

        with torch.no_grad():
            st = time.time()
            out1, out2, out = model(in_img, mask)
            cur_time = time.time()-st
            if h_pad != 0:
               out = out[:, :, h_pad:-h_odd_pad, :]
            if w_pad != 0:
               out = out[:, :, :, w_pad:-w_odd_pad]

        if args.EVALUATION_METRIC:
            cur_lpips, cur_psnr, cur_ssim = compute_metrics.compute(out, label)

        # save images
        if args.SAVE_IMG:
            out_save = out.detach().cpu()
            torchvision.utils.save_image(out_save, save_path + '/' + 'test_%s' % number[0] + '.%s' % args.SAVE_IMG)

        return cur_psnr, cur_ssim, cur_lpips, cur_time

    def model_fn(args, data, model, iters):
        model.train()
        # prepare input and forward
        in_img = data['in_img'].to(device)
        label = data['label'].to(device)
        img_nf = data['img_nf'].to(device)
        # print(in_img.size())
        # print(img_nf.size())
        old_img = in_img
        old_img = old_img[:, 0:1, :, :] * 0.299 + old_img[:, 1:2, :, :] * 0.587 + old_img[:, 2:3, :, :] * 0.114
        de_moire_img = img_nf
        de_moire_img = de_moire_img[:, 0:1, :, :] * 0.299 + de_moire_img[:, 1:2, :, :] * 0.587 + de_moire_img[:, 2:3, :, :] * 0.114
        noise = torch.abs(old_img - de_moire_img)

        mask = torch.div(de_moire_img, noise + 0.0001)

        batch_size = mask.shape[0]
        height = mask.shape[2]
        width = mask.shape[3]
        mask_max = torch.max(mask.view(batch_size, -1), dim=1)[0]
        mask_max = mask_max.view(batch_size, 1, 1, 1)
        mask_max = mask_max.repeat(1, 1, height, width)
        mask = mask * 1.0 / (mask_max + 0.0001)

        mask = torch.clamp(mask, min=0, max=1.0)
        mask = mask.float()
        out1, out2, out = model(in_img, mask=mask)
        loss = loss_fn(out1, out2, out, label)
        # save images
        if iters % args.SAVE_ITER == (args.SAVE_ITER - 1):
            in_save = in_img.detach().cpu()
            out_save = out.detach().cpu()
            gt_save = label.detach().cpu()
            res_save = torch.cat((in_save, out_save, gt_save), 3)
            save_number = (iters + 1) // args.SAVE_ITER
            torchvision.utils.save_image(res_save,
                                         args.VISUALS_DIR + '/visual_x%04d_' % args.SAVE_ITER + '%05d' % save_number + '.jpg')


        return loss

    if mode == 'test':
        fn = test_model_fn
    else:
        fn = model_fn
    return fn