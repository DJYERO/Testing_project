# -*- coding: utf-8 -*-
"""
# @Time    : 2021/12/9 上午9:57
# @Author  : DJ_YERO
# @File    : main
# @Content : 内容说明
"""
import torch
from torchvision import models

device = "cuda" if torch.cuda.is_available() else "cpu"
model = models.resnet18(pretrained=False)
model.to(device)
model.eval()

data = torch.randn(2, 3, 640, 640).to(device)
output = model(data)

print(output.shape)
