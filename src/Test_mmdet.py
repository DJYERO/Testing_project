# -*- coding: utf-8 -*-
"""
# @Time    : 2021/12/9 上午11:36
# @Author  : DJ_YERO
# @File    : Test_mmdet.py
# @Content : 测试mmdetection
"""
# 配置环境
# Check Pytorch installation
# import torch, torchvision
#
# print(torch.__version__, torch.cuda.is_available())
#
# # Check MMDetection installation
# import mmdet
#
# print(mmdet.__version__)
#
# # Check mmcv installation
# from mmcv.ops import get_compiling_cuda_version, get_compiler_version
#
# print(get_compiling_cuda_version())
# print(get_compiler_version())

from mmdet.apis import inference_detector, init_detector, show_result_pyplot

config = '../../mmdetection-master/configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco.py'
checkpoint = '../checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'
model = init_detector(config, checkpoint, device="cuda:0")

img = '../demo/demo.jpg'
result = inference_detector(model, img)
show_result_pyplot(model, img, result,score_thr=0.3)
