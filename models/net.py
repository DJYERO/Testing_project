# -*- coding: utf-8 -*-
"""
# @Time    : 2021/12/9 上午11:04
# @Author  : DJ_YERO
# @File    : net.py
# @Content : 内容说明
"""
from torchvision import models
from torchsummary import summary
import torch

decive = "cuda:0" if torch.cuda.is_available() else "cpu"

net = models.mobilenet_v2(pretrained=False).to(decive)
net.eval()

summary(net, (3, 640, 640))
