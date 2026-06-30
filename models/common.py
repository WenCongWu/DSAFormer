# YOLOv5 common modules

import math
from copy import copy
from pathlib import Path
import warnings
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

import cv2
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from torch import einsum
from PIL import Image
from torch.cuda import amp
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn.modules.utils import _triple, _pair, _single
# from einops import rearrange, repeat
# from einops.layers.torch import Rearrange

from utils.datasets import letterbox
from utils.general import non_max_suppression, make_divisible, scale_coords, increment_path, xyxy2xywh, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import time_synchronized
from timm.models.layers import DropPath

from torch.nn import init, Sequential
import math
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
import numbers
from einops import rearrange


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


def DWConv(c1, c2, k=1, s=1, act=True):
    # Depthwise convolution
    return Conv(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class Conv_withoutBN(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.act = nn.SiLU() if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.conv(x))


class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*[TransformerLayer(c2, num_heads) for _ in range(num_layers)])
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2)
        p = p.unsqueeze(0)
        p = p.transpose(0, 3)
        p = p.squeeze(3)
        e = self.linear(p)
        x = p + e

        x = self.tr(x)
        x = x.unsqueeze(3)
        x = x.transpose(0, 3)
        x = x.reshape(b, self.c2, w, h)
        return x


class VGGblock(nn.Module):
    def __init__(self, num_convs, c1, c2):
        super(VGGblock, self).__init__()
        self.blk = []
        for num in range(num_convs):
            if num == 0:
                self.blk.append(nn.Sequential(nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=3, padding=1),
                                              nn.ReLU(),
                                              ))
            else:
                self.blk.append(nn.Sequential(nn.Conv2d(in_channels=c2, out_channels=c2, kernel_size=3, padding=1),
                                              nn.ReLU(),
                                              ))
        self.blk.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.vggblock = nn.Sequential(*self.blk)

    def forward(self, x):
        out = self.vggblock(x)

        return out


class ResNetblock(nn.Module):
    expansion = 4

    def __init__(self, c1, c2, stride=1):
        super(ResNetblock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c2)
        self.conv2 = nn.Conv2d(in_channels=c2, out_channels=c2, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.conv3 = nn.Conv2d(in_channels=c2, out_channels=self.expansion * c2, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * c2)

        self.shortcut = nn.Sequential()
        if stride != 1 or c1 != self.expansion * c2:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=c1, out_channels=self.expansion * c2, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * c2),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)

        return out


class ResNetlayer(nn.Module):
    expansion = 4

    def __init__(self, c1, c2, stride=1, is_first=False, num_blocks=1):
        super(ResNetlayer, self).__init__()
        self.blk = []
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(
                nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(c2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        else:
            self.blk.append(ResNetblock(c1, c2, stride))
            for i in range(num_blocks - 1):
                self.blk.append(ResNetblock(self.expansion * c2, c2, 1))
            self.layer = nn.Sequential(*self.blk)

    def forward(self, x):
        out = self.layer(x)

        return out


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class C3TR(C3):
    # C3 module with TransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        # print("c1 * 4, c2, k", c1 * 4, c2, k)
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        # print("Focus inputs shape", x.shape)
        # print()
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
        # return self.conv(self.contract(x))


class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()  # assert (H / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(N, C, H // s, s, W // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(N, C * s * s, H // s, W // s)  # x(1,256,40,40)


class Expand(nn.Module):
    # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(N, s, s, C // s ** 2, H, W)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(N, C // s ** 2, H * s, W * s)  # x(1,16,160,160)


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        # print(x.shape)
        return torch.cat(x, self.d)


class Add(nn.Module):
    # Add a list of tensors and averge
    def __init__(self, weight=0.5):
        super().__init__()
        self.w = weight

    def forward(self, x):
        return x[0] * self.w + x[1] * (1 - self.w)


class Add2(nn.Module):
    #  x + transformer[0] or x + transformer[1]
    def __init__(self, c1, index):
        super().__init__()
        self.index = index

    def forward(self, x):
        if self.index == 0:
            return torch.add(x[0], x[1][0])
        elif self.index == 1:
            return torch.add(x[0], x[1][1])
        # return torch.add(x[0], x[1])


class NiNfusion(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        super(NiNfusion, self).__init__()

        self.concat = Concat(dimension=1)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.act = nn.SiLU()

    def forward(self, x):
        y = self.concat(x)
        y = self.act(self.conv(y))

        return y


class DMAF(nn.Module):
    def __init__(self, c2):
        super(DMAF, self).__init__()

    def forward(self, x):
        x1 = x[0]
        x2 = x[1]

        subtract_vis = x1 - x2
        avgpool_vis = nn.AvgPool2d(kernel_size=(subtract_vis.size(2), subtract_vis.size(3)))
        weight_vis = torch.tanh(avgpool_vis(subtract_vis))

        subtract_ir = x2 - x1
        avgpool_ir = nn.AvgPool2d(kernel_size=(subtract_ir.size(2), subtract_ir.size(3)))
        weight_ir = torch.tanh(avgpool_ir(subtract_ir))

        x1_weight = subtract_vis * weight_ir
        x2_weight = subtract_ir * weight_vis

        return x1_weight, x2_weight


class NMS(nn.Module):
    # Non-Maximum Suppression (NMS) module
    conf = 0.25  # confidence threshold
    iou = 0.45  # IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self):
        super(NMS, self).__init__()

    def forward(self, x):
        return non_max_suppression(x[0], conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)


class autoShape(nn.Module):
    # input-robust model wrapper for passing cv2/np/PIL/torch inputs. Includes preprocessing, inference and NMS
    conf = 0.25  # NMS confidence threshold
    iou = 0.45  # NMS IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self, model):
        super(autoShape, self).__init__()
        self.model = model.eval()

    def autoshape(self):
        print('autoShape already enabled, skipping... ')  # model already converted to model.autoshape()
        return self

    @torch.no_grad()
    def forward(self, imgs, size=640, augment=False, profile=False):
        # Inference from various sources. For height=640, width=1280, RGB images example inputs are:
        #   filename:   imgs = 'data/images/zidane.jpg'
        #   URI:             = 'https://github.com/ultralytics/yolov5/releases/download/v1.0/zidane.jpg'
        #   OpenCV:          = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(640,1280,3)
        #   PIL:             = Image.open('image.jpg')  # HWC x(640,1280,3)
        #   numpy:           = np.zeros((640,1280,3))  # HWC
        #   torch:           = torch.zeros(16,3,320,640)  # BCHW (scaled to size=640, 0-1 values)
        #   multiple:        = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images

        t = [time_synchronized()]
        p = next(self.model.parameters())  # for device and type
        if isinstance(imgs, torch.Tensor):  # torch
            with amp.autocast(enabled=p.device.type != 'cpu'):
                return self.model(imgs.to(p.device).type_as(p), augment, profile)  # inference

        # Pre-process
        n, imgs = (len(imgs), imgs) if isinstance(imgs, list) else (1, [imgs])  # number of images, list of images
        shape0, shape1, files = [], [], []  # image and inference shapes, filenames
        for i, im in enumerate(imgs):
            f = f'image{i}'  # filename
            if isinstance(im, str):  # filename or uri
                im, f = np.asarray(Image.open(requests.get(im, stream=True).raw if im.startswith('http') else im)), im
            elif isinstance(im, Image.Image):  # PIL Image
                im, f = np.asarray(im), getattr(im, 'filename', f) or f
            files.append(Path(f).with_suffix('.jpg').name)
            if im.shape[0] < 5:  # image in CHW
                im = im.transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
            im = im[:, :, :3] if im.ndim == 3 else np.tile(im[:, :, None], 3)  # enforce 3ch input
            s = im.shape[:2]  # HWC
            shape0.append(s)  # image shape
            g = (size / max(s))  # gain
            shape1.append([y * g for y in s])
            imgs[i] = im if im.data.contiguous else np.ascontiguousarray(im)  # update
        shape1 = [make_divisible(x, int(self.stride.max())) for x in np.stack(shape1, 0).max(0)]  # inference shape
        x = [letterbox(im, new_shape=shape1, auto=False)[0] for im in imgs]  # pad
        x = np.stack(x, 0) if n > 1 else x[0][None]  # stack
        x = np.ascontiguousarray(x.transpose((0, 3, 1, 2)))  # BHWC to BCHW
        x = torch.from_numpy(x).to(p.device).type_as(p) / 255.  # uint8 to fp16/32
        t.append(time_synchronized())

        with amp.autocast(enabled=p.device.type != 'cpu'):
            # Inference
            y = self.model(x, augment, profile)[0]  # forward
            t.append(time_synchronized())

            # Post-process
            y = non_max_suppression(y, conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)  # NMS
            for i in range(n):
                scale_coords(shape1, y[i][:, :4], shape0[i])

            t.append(time_synchronized())
            return Detections(imgs, y, files, t, self.names, x.shape)


class Detections:
    # detections class for YOLOv5 inference results
    def __init__(self, imgs, pred, files, times=None, names=None, shape=None):
        super(Detections, self).__init__()
        d = pred[0].device  # device
        gn = [torch.tensor([*[im.shape[i] for i in [1, 0, 1, 0]], 1., 1.], device=d) for im in imgs]  # normalizations
        self.imgs = imgs  # list of images as numpy arrays
        self.pred = pred  # list of tensors pred[0] = (xyxy, conf, cls)
        self.names = names  # class names
        self.files = files  # image filenames
        self.xyxy = pred  # xyxy pixels
        self.xywh = [xyxy2xywh(x) for x in pred]  # xywh pixels
        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]  # xyxy normalized
        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]  # xywh normalized
        self.n = len(self.pred)  # number of images (batch size)
        self.t = tuple((times[i + 1] - times[i]) * 1000 / self.n for i in range(3))  # timestamps (ms)
        self.s = shape  # inference BCHW shape

    def display(self, pprint=False, show=False, save=False, crop=False, render=False, save_dir=Path('')):
        for i, (im, pred) in enumerate(zip(self.imgs, self.pred)):
            str = f'image {i + 1}/{len(self.pred)}: {im.shape[0]}x{im.shape[1]} '
            if pred is not None:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()  # detections per class
                    str += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                if show or save or render or crop:
                    for *box, conf, cls in pred:  # xyxy, confidence, class
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        if crop:
                            save_one_box(box, im, file=save_dir / 'crops' / self.names[int(cls)] / self.files[i])
                        else:  # all others
                            plot_one_box(box, im, label=label, color=colors(cls))

            im = Image.fromarray(im.astype(np.uint8)) if isinstance(im, np.ndarray) else im  # from np
            if pprint:
                print(str.rstrip(', '))
            if show:
                im.show(self.files[i])  # show
            if save:
                f = self.files[i]
                im.save(save_dir / f)  # save
                print(f"{'Saved' * (i == 0)} {f}", end=',' if i < self.n - 1 else f' to {save_dir}\n')
            if render:
                self.imgs[i] = np.asarray(im)

    def print(self):
        self.display(pprint=True)  # print results
        print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {tuple(self.s)}' % self.t)

    def show(self):
        self.display(show=True)  # show results

    def save(self, save_dir='runs/hub/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/hub/exp', mkdir=True)  # increment save_dir
        self.display(save=True, save_dir=save_dir)  # save results

    def crop(self, save_dir='runs/hub/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/hub/exp', mkdir=True)  # increment save_dir
        self.display(crop=True, save_dir=save_dir)  # crop results
        print(f'Saved results to {save_dir}\n')

    def render(self):
        self.display(render=True)  # render results
        return self.imgs

    def pandas(self):
        # return detections as pandas DataFrames, i.e. print(results.pandas().xyxy[0])
        new = copy(self)  # return copy
        ca = 'xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'  # xyxy columns
        cb = 'xcenter', 'ycenter', 'width', 'height', 'confidence', 'class', 'name'  # xywh columns
        for k, c in zip(['xyxy', 'xyxyn', 'xywh', 'xywhn'], [ca, ca, cb, cb]):
            a = [[x[:5] + [int(x[5]), self.names[int(x[5])]] for x in x.tolist()] for x in getattr(self, k)]  # update
            setattr(new, k, [pd.DataFrame(x, columns=c) for x in a])
        return new

    def tolist(self):
        # return a list of Detections objects, i.e. 'for result in results.tolist():'
        x = [Detections([self.imgs[i]], [self.pred[i]], self.names, self.s) for i in range(self.n)]
        for d in x:
            for k in ['imgs', 'pred', 'xyxy', 'xyxyn', 'xywh', 'xywhn']:
                setattr(d, k, getattr(d, k)[0])  # pop out of list
        return x

    def __len__(self):
        return self.n


class Classify(nn.Module):
    # Classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Classify, self).__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)  # to x(b,c1,1,1)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g)  # to x(b,c2,1,1)
        self.flat = nn.Flatten()

    def forward(self, x):
        z = torch.cat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)  # cat if list
        return self.flat(self.conv(z))  # flatten to x(b,c2)


class AdaptivePool2d(nn.Module):
    def __init__(self, output_h, output_w, pool_type='avg'):
        super(AdaptivePool2d, self).__init__()

        self.output_h = output_h
        self.output_w = output_w
        self.pool_type = pool_type

    def forward(self, x):
        bs, c, input_h, input_w = x.shape

        if (input_h > self.output_h) or (input_w > self.output_w):
            self.stride_h = input_h // self.output_h
            self.stride_w = input_w // self.output_w
            self.kernel_size = (
                input_h - (self.output_h - 1) * self.stride_h, input_w - (self.output_w - 1) * self.stride_w)

            if self.pool_type == 'avg':
                y = nn.AvgPool2d(kernel_size=self.kernel_size, stride=(self.stride_h, self.stride_w), padding=0)(x)
            else:
                y = nn.MaxPool2d(kernel_size=self.kernel_size, stride=(self.stride_h, self.stride_w), padding=0)(x)
        else:
            y = x

        return y


class SE_Block(nn.Module):
    def __init__(self, inchannel, ratio=16):
        super(SE_Block, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // ratio, bias=False),  # 从 c -> c/r
            nn.ReLU(),
            nn.Linear(inchannel // ratio, inchannel, bias=False),  # 从 c/r -> c
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.gap(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)

        return x * y.expand_as(x)


# 通道注意力模块
class Channel_Attention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        '''
        :param in_channels: 输入通道数
        :param reduction_ratio: 输出通道数量的缩放系数
        :param pool_types: 池化类型
        '''

        super(Channel_Attention, self).__init__()

        self.pool_types = pool_types
        self.in_channels = in_channels
        self.shared_mlp = nn.Sequential(nn.Flatten(),
                                        nn.Linear(in_features=in_channels, out_features=in_channels // reduction_ratio),
                                        nn.ReLU(),
                                        nn.Linear(in_features=in_channels // reduction_ratio, out_features=in_channels)
                                        )

    def forward(self, x):
        channel_attentions = []

        for pool_types in self.pool_types:
            if pool_types == 'avg':
                pool_init = nn.AvgPool2d(kernel_size=(x.size(2), x.size(3)))
                avg_pool = pool_init(x)
                channel_attentions.append(self.shared_mlp(avg_pool))
            elif pool_types == 'max':
                pool_init = nn.MaxPool2d(kernel_size=(x.size(2), x.size(3)))
                max_pool = pool_init(x)
                channel_attentions.append(self.shared_mlp(max_pool))

        pooling_sums = torch.stack(channel_attentions, dim=0).sum(dim=0)
        output = nn.Sigmoid()(pooling_sums).unsqueeze(2).unsqueeze(3).expand_as(x)

        return x * output


# 空间注意力模块
class Spatial_Attention(nn.Module):
    def __init__(self, kernel_size=7):
        super(Spatial_Attention, self).__init__()

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, stride=1, dilation=1,
                      padding=(kernel_size - 1) // 2, bias=False),
            nn.BatchNorm2d(num_features=1, eps=1e-5, momentum=0.01, affine=True)
        )

    def forward(self, x):
        x_compress = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)),
                               dim=1)  # 在通道维度上分别计算平均值和最大值，并在通道维度上进行拼接
        x_output = self.spatial_attention(x_compress)  # 使用7x7卷积核进行卷积
        scaled = nn.Sigmoid()(x_output)

        return x * scaled  # 将输入F'和通道注意力模块的输出Ms相乘，得到F''


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, pool_types=['avg', 'max'], spatial=True):
        super(CBAM, self).__init__()

        self.spatial = spatial
        self.channel_attention = Channel_Attention(in_channels=in_channels, reduction_ratio=reduction_ratio,
                                                   pool_types=pool_types)

        if self.spatial:
            self.spatial_attention = Spatial_Attention(kernel_size=7)

    def forward(self, x):
        x_out = self.channel_attention(x)
        if self.spatial:
            x_out = self.spatial_attention(x_out)

        return x_out


####################################################top-k sparse transformer
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##  Multi-Scale Feature Refinement Layer (MSFRL)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_dim = dim * ffn_expansion_factor // 2
        self.linear1 = nn.Sequential(nn.Linear(dim, dim * ffn_expansion_factor),
                                     nn.ReLU())
        self.dwconv3x3 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, stride=1, padding=0, groups=hidden_dim, bias=bias),
            nn.ReLU())

        self.dwconv5x5 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, groups=hidden_dim, bias=bias),
            nn.ReLU())

        self.dwconv7x7 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=5, stride=1, padding=2, groups=hidden_dim, bias=bias),
            nn.ReLU())

        self.dwconv3x3_1 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, stride=1, padding=0, groups=hidden_dim, bias=bias),
            nn.ReLU())

        self.dwconv5x5_1 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, groups=hidden_dim, bias=bias),
            nn.ReLU())

        self.dwconv7x7_1 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=5, stride=1, padding=2, groups=hidden_dim, bias=bias),
            nn.ReLU())

        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))
        self.dim = dim
        self.hidden_dim = hidden_dim

        self.dim_conv = self.dim // 4
        self.dim_untouched = self.dim - self.dim_conv
        self.partial_conv = nn.Conv2d(self.dim_conv, self.dim_conv, 3, 1, 1, bias=bias)

        self.conv = nn.Conv2d(self.dim * 9 // 2, self.dim * 3 // 2, kernel_size=1, bias=bias)

    def forward(self, x):
        # bs h w c
        b, c, h, w = x.size()

        # spatial restore
        # x = rearrange(x, ' b (h w) (c) -> b c h w ', h=h, w=w)

        # upper branch
        # print('x shape:', x.shape)
        x1, x2 = torch.split(x, [self.dim_conv, self.dim_untouched], dim=1)
        x1 = self.partial_conv(x1)
        x = torch.cat((x1, x2), 1)
        # print('afetr partial_conv x shape:', x.shape)

        # flaten
        x = x.permute(0, 2, 3, 1)
        x = rearrange(x, ' b h w c -> b (h w) c', h=h, w=w)

        x = self.linear1(x)
        # gate mechanism
        x_1, x_2 = x.chunk(2, dim=-1)

        x_1 = rearrange(x_1, ' b (h w) (c) -> b h w c', h=h, w=w)
        x_1 = x_1.permute(0, 3, 1, 2)
        # print('afetr partial_conv x_1 shape:', x_1.shape)

        x1_1, x2_1, x3_1 = self.dwconv3x3(x_1).chunk(3, dim=1)
        x1_3, x2_3, x3_3 = self.dwconv5x5(x_1).chunk(3, dim=1)
        x1_5, x2_5, x3_5 = self.dwconv7x7(x_1).chunk(3, dim=1)

        x1 = torch.cat([x1_1, x1_3, x1_5], dim=1)
        x2 = torch.cat([x2_1, x2_3, x2_5], dim=1)
        x3 = torch.cat([x3_1, x3_3, x3_5], dim=1)

        x1 = self.dwconv3x3_1(x1)
        x2 = self.dwconv5x5_1(x2)
        x3 = self.dwconv7x7_1(x3)

        x3 = torch.cat([x1, x2, x3], dim=1)
        x_1 = self.conv(x3)
        x_1 = rearrange(x_1, ' b c h w -> b c (h w)', h=h, w=w)
        x_1 = x_1.permute(0, 2, 1)

        x = x_1 * x_2
        x = self.linear2(x)

        x = rearrange(x, ' b (h w) (c) -> b h w c', h=h, w=w)
        x = x.permute(0, 3, 1, 2)
        # print('last x shape:', x.shape)

        return x


class SelfAttention(nn.Module):
    """
     Multi-head masked self-attention layer
    """

    def __init__(self, dim, h, bias):
        '''
        :param dim: Output dimensionality of the model
        :param h: Number of heads
        '''
        super(SelfAttention, self).__init__()
        self.num_heads = h

        self.temperature = nn.Parameter(torch.ones(self.num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3,
                                    bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.attn_ir1 = torch.nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        self.attn_ir2 = torch.nn.Parameter(torch.tensor([0.5]), requires_grad=True)

        self.attn_rgb1 = torch.nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        self.attn_rgb2 = torch.nn.Parameter(torch.tensor([0.5]), requires_grad=True)

    def forward(self, x):
        b, c, h, w = x[0].shape

        rgb_fea = x[0]  # 2024/11/1 added by wwc
        ir_fea = x[1]  # 2024/11/1 added by wwc

        rgb_qkv = self.qkv_dwconv(self.qkv(rgb_fea))
        rgb_q, rgb_k, rgb_v = rgb_qkv.chunk(3, dim=1)

        ir_qkv = self.qkv_dwconv(self.qkv(ir_fea))
        ir_q, ir_k, ir_v = ir_qkv.chunk(3, dim=1)

        rgb_q = rearrange(rgb_q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        rgb_k = rearrange(rgb_k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        rgb_v = rearrange(rgb_v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        ir_q = rearrange(ir_q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        ir_k = rearrange(ir_k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        ir_v = rearrange(ir_v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        rgb_q = torch.nn.functional.normalize(rgb_q, dim=-1)
        rgb_k = torch.nn.functional.normalize(rgb_k, dim=-1)

        ir_q = torch.nn.functional.normalize(ir_q, dim=-1)
        ir_k = torch.nn.functional.normalize(ir_k, dim=-1)

        _, _, _, S = rgb_q.shape

        ir_mask1 = torch.zeros(b, self.num_heads, S, S, device=x[0].device, requires_grad=False)
        ir_mask2 = torch.zeros(b, self.num_heads, S, S, device=x[0].device, requires_grad=False)

        rgb_mask1 = torch.zeros(b, self.num_heads, S, S, device=x[0].device, requires_grad=False)
        rgb_mask2 = torch.zeros(b, self.num_heads, S, S, device=x[0].device, requires_grad=False)

        attn_ir = (rgb_q.transpose(-2, -1) @ ir_k) * self.temperature

        attn_rgb = (ir_q.transpose(-2, -1) @ rgb_k) * self.temperature

        index_ir = torch.topk(attn_ir, k=int(S / 3), dim=-1, largest=True)[1]
        ir_mask1.scatter_(-1, index_ir, 1.)
        attn_ir1 = torch.where(ir_mask1 > 0, attn_ir, torch.full_like(attn_ir, float('-inf')))

        index_ir = torch.topk(attn_ir, k=int(S * 4 / 5), dim=-1, largest=True)[1]
        ir_mask2.scatter_(-1, index_ir, 1.)
        attn_ir2 = torch.where(ir_mask2 > 0, attn_ir, torch.full_like(attn_ir, float('-inf')))

        index_rgb = torch.topk(attn_rgb, k=int(S / 3), dim=-1, largest=True)[1]
        rgb_mask1.scatter_(-1, index_rgb, 1.)
        attn_rgb1 = torch.where(rgb_mask1 > 0, attn_rgb, torch.full_like(attn_rgb, float('-inf')))

        index_rgb = torch.topk(attn_rgb, k=int(S * 4 / 5), dim=-1, largest=True)[1]
        rgb_mask2.scatter_(-1, index_rgb, 1.)
        attn_rgb2 = torch.where(rgb_mask2 > 0, attn_rgb, torch.full_like(attn_rgb, float('-inf')))

        attn_ir1 = attn_ir1.softmax(dim=-1)
        attn_ir2 = attn_ir2.softmax(dim=-1)

        attn_rgb1 = attn_rgb1.softmax(dim=-1)
        attn_rgb2 = attn_rgb2.softmax(dim=-1)

        out_ir1 = (attn_ir1 @ ir_v.transpose(-2, -1))
        out_ir2 = (attn_ir2 @ ir_v.transpose(-2, -1))

        out_rgb1 = (attn_rgb1 @ rgb_v.transpose(-2, -1))
        out_rgb2 = (attn_rgb2 @ rgb_v.transpose(-2, -1))

        out_ir = out_ir1 * self.attn_ir1 + out_ir2 * self.attn_ir2
        out_rgb = out_rgb1 * self.attn_rgb1 + out_rgb2 * self.attn_rgb2

        out_ir = out_ir.transpose(-2, -1)
        out_rgb = out_rgb.transpose(-2, -1)

        out_ir = rearrange(out_ir, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out_rgb = rearrange(out_rgb, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out_ir = self.project_out(out_ir)
        out_rgb = self.project_out(out_rgb)

        return [out_rgb, out_ir]


class Up(nn.Module):

    def __init__(self, dim, bias):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels=dim, out_channels=dim, kernel_size=2, stride=2, bias=bias)

    def forward(self, x1, x):
        x2 = self.up(x1)

        diffY = x.size()[2] - x2.size()[2]
        diffX = x.size()[3] - x2.size()[3]
        x3 = F.pad(x2, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        return x3


# Cross-Modal Spatial Sparse Transformer
class CMSST(nn.Module):

    def __init__(self, dim, num_heads, bias):
        super(CMSST, self).__init__()

        ffn_expansion_factor = 3
        bias = False
        LayerNorm_type = 'WithBias'
        self.norm = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

        self.down = nn.Conv2d(dim, dim, kernel_size=2, stride=2, bias=bias)

        self.up = Up(dim, bias)

        # transformer
        self.Att = SelfAttention(dim, num_heads, bias)

    def forward(self, x):
        rgb_fea = x[0]  # rgb_fea (tensor): dim:(B, C, H, W)
        ir_fea = x[1]  # ir_fea (tensor): dim:(B, C, H, W)
        assert rgb_fea.shape[0] == ir_fea.shape[0]

        # -------------------------------------------------------------------------
        # AvgPooling
        # -------------------------------------------------------------------------
        # AvgPooling for reduce the dimension due to expensive computation
        rgb_fea_down = self.down(rgb_fea)  # (B, C, H/2, W/2)
        ir_fea_down = self.down(ir_fea)  # (B, C, H/2, W/2)

        rgb_fea_norm = self.norm(rgb_fea_down)  # 2024/11/1 added by wwc
        ir_fea_norm = self.norm(ir_fea_down)  # 2024/11/1 added by wwc

        x = self.Att([rgb_fea_norm, ir_fea_norm])  # dim:(B, C, vert_anchors, horz_anchors)

        # -------------------------------------------------------------------------
        # Interpolate (or Upsample)
        # -------------------------------------------------------------------------
        out_rgb = self.up(x[0], rgb_fea)
        out_ir = self.up(x[1], ir_fea)

        s_out_rgb = rgb_fea + out_rgb
        s_out_ir = ir_fea + out_ir

        new_rgb_fea = s_out_rgb + self.ffn(self.norm(s_out_rgb))
        new_ir_fea = s_out_ir + self.ffn(self.norm(s_out_ir))

        return [new_rgb_fea, new_ir_fea]


##  Cross-Modal Channel Sparse Transformer (CMCST)
class CMCST(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(CMCST, self).__init__()
        self.num_heads = num_heads
        LayerNorm_type = 'WithBias'

        ffn_expansion_factor = 3
        bias = False
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

        self.norm = LayerNorm(dim, LayerNorm_type)

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3,
                                    bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.down = nn.Conv2d(dim, dim, kernel_size=2, stride=2, bias=bias)

        self.up = Up(dim, bias)

        self.attn_ir1 = torch.nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        self.attn_ir2 = torch.nn.Parameter(torch.tensor([0.5]), requires_grad=True)

        self.attn_rgb1 = torch.nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        self.attn_rgb2 = torch.nn.Parameter(torch.tensor([0.5]), requires_grad=True)

    def forward(self, x):
        rgb_fea0 = x[0]  # 2024/11/1 added by wwc
        ir_fea0 = x[1]  # 2024/11/1 added by wwc

        rgb_fea_down = self.down(rgb_fea0)  # (B, C, H/2, W/2)
        ir_fea_down = self.down(ir_fea0)  # (B, C, H/2, W/2)

        rgb_fea_norm = self.norm(rgb_fea_down)  # 2024/11/1 added by wwc
        ir_fea_norm = self.norm(ir_fea_down)  # 2024/11/1 added by wwc
        b, c, h, w = rgb_fea_norm.shape

        rgb_qkv = self.qkv_dwconv(self.qkv(rgb_fea_norm))
        rgb_q, rgb_k, rgb_v = rgb_qkv.chunk(3, dim=1)

        ir_qkv = self.qkv_dwconv(self.qkv(ir_fea_norm))
        ir_q, ir_k, ir_v = ir_qkv.chunk(3, dim=1)

        rgb_q = rearrange(rgb_q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        rgb_k = rearrange(rgb_k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        rgb_v = rearrange(rgb_v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        ir_q = rearrange(ir_q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        ir_k = rearrange(ir_k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        ir_v = rearrange(ir_v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        rgb_q = torch.nn.functional.normalize(rgb_q, dim=-1)
        rgb_k = torch.nn.functional.normalize(rgb_k, dim=-1)

        ir_q = torch.nn.functional.normalize(ir_q, dim=-1)
        ir_k = torch.nn.functional.normalize(ir_k, dim=-1)

        _, _, C, _ = rgb_q.shape

        ir_mask1 = torch.zeros(b, self.num_heads, C, C, device=x[0].device, requires_grad=False)
        ir_mask2 = torch.zeros(b, self.num_heads, C, C, device=x[0].device, requires_grad=False)

        rgb_mask1 = torch.zeros(b, self.num_heads, C, C, device=x[0].device, requires_grad=False)
        rgb_mask2 = torch.zeros(b, self.num_heads, C, C, device=x[0].device, requires_grad=False)

        attn_ir = (rgb_q @ ir_k.transpose(-2, -1)) * self.temperature

        attn_rgb = (ir_q @ rgb_k.transpose(-2, -1)) * self.temperature

        index_ir = torch.topk(attn_ir, k=int(C * 2 / 3), dim=-1, largest=True)[1]
        ir_mask1.scatter_(-1, index_ir, 1.)
        attn_ir1 = torch.where(ir_mask1 > 0, attn_ir, torch.full_like(attn_ir, float('-inf')))

        index_ir = torch.topk(attn_ir, k=int(C * 4 / 5), dim=-1, largest=True)[1]
        ir_mask2.scatter_(-1, index_ir, 1.)
        attn_ir2 = torch.where(ir_mask2 > 0, attn_ir, torch.full_like(attn_ir, float('-inf')))

        index_rgb = torch.topk(attn_rgb, k=int(C * 2 / 3), dim=-1, largest=True)[1]
        rgb_mask1.scatter_(-1, index_rgb, 1.)
        attn_rgb1 = torch.where(rgb_mask1 > 0, attn_rgb, torch.full_like(attn_rgb, float('-inf')))

        index_rgb = torch.topk(attn_rgb, k=int(C * 4 / 5), dim=-1, largest=True)[1]
        rgb_mask2.scatter_(-1, index_rgb, 1.)
        attn_rgb2 = torch.where(rgb_mask2 > 0, attn_rgb, torch.full_like(attn_rgb, float('-inf')))

        attn_ir1 = attn_ir1.softmax(dim=-1)
        attn_ir2 = attn_ir2.softmax(dim=-1)

        attn_rgb1 = attn_rgb1.softmax(dim=-1)
        attn_rgb2 = attn_rgb2.softmax(dim=-1)

        out_ir1 = (attn_ir1 @ ir_v)
        out_ir2 = (attn_ir2 @ ir_v)

        out_rgb1 = (attn_rgb1 @ rgb_v)
        out_rgb2 = (attn_rgb2 @ rgb_v)

        out_ir = out_ir1 * self.attn_ir1 + out_ir2 * self.attn_ir2
        out_rgb = out_rgb1 * self.attn_rgb1 + out_rgb2 * self.attn_rgb2

        out_ir = rearrange(out_ir, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out_rgb = rearrange(out_rgb, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out_ir = self.project_out(out_ir)
        out_rgb = self.project_out(out_rgb)

        out_rgb = self.up(out_rgb, rgb_fea0)
        out_ir = self.up(out_ir, ir_fea0)

        c_out_rgb = rgb_fea0 + out_rgb
        c_out_ir = ir_fea0 + out_ir

        new_rgb_fea = c_out_rgb + self.ffn(self.norm(c_out_rgb))
        new_ir_fea = c_out_ir + self.ffn(self.norm(c_out_ir))

        return [new_rgb_fea, new_ir_fea]


class LearnableWeights_rgb(nn.Module):
    def __init__(self):
        super(LearnableWeights_rgb, self).__init__()
        self.w1 = nn.Parameter(torch.tensor([1.0]), requires_grad=True)
        self.w2 = nn.Parameter(torch.tensor([1.0]), requires_grad=True)

    def forward(self, x1, x2):
        out = x1 * self.w1 + x2 * self.w2
        return out


class LearnableWeights_ir(nn.Module):
    def __init__(self):
        super(LearnableWeights_ir, self).__init__()
        self.w1 = nn.Parameter(torch.tensor([1.0]), requires_grad=True)
        self.w2 = nn.Parameter(torch.tensor([1.0]), requires_grad=True)

    def forward(self, x1, x2):
        out = x1 * self.w1 + x2 * self.w2
        return out


class LearnableWeights_fusion(nn.Module):
    def __init__(self):
        super(LearnableWeights_fusion, self).__init__()
        self.w1 = nn.Parameter(torch.tensor([1.0]), requires_grad=True)
        self.w2 = nn.Parameter(torch.tensor([1.0]), requires_grad=True)

    def forward(self, x1, x2):
        out = x1 * self.w1 + x2 * self.w2
        return out


##  Dual Sparse Aggregation Transformer
class IDSFormer(nn.Module):
    def __init__(self, dim):
        super(IDSFormer, self).__init__()
        num_heads = 8
        bias = False
        self.loops = 1

        self.cmsst = CMSST(dim, num_heads, bias)
        self.cmcst = CMCST(dim, num_heads, bias)

        # LearnableCoefficient
        self.rgb_weight = LearnableWeights_rgb()
        self.ir_weight = LearnableWeights_ir()
        self.fusion_weight = LearnableWeights_fusion()

    def forward(self, x):
        rgb_fea = x[0]
        ir_fea = x[1]

        for loop in range(self.loops):
            s_rgb_fea, s_ir_fea = self.cmsst([rgb_fea, ir_fea])
            c_rgb_fea, c_ir_fea = self.cmcst([rgb_fea, ir_fea])
            rgb_fea = self.rgb_weight(s_rgb_fea, c_rgb_fea)
            ir_fea = self.ir_weight(s_ir_fea, c_ir_fea)

        # Fusion module
        new_fea = self.fusion_weight(rgb_fea, ir_fea)

        return new_fea