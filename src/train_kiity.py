# -*- coding: utf-8 -*-
"""
# @Time    : 2021/12/9 下午3:09
# @Author  : DJ_YERO
# @File    : train_kiity.py
# @Content : kitty数据集训练
"""
from config.kitty_config import cfg
# 配置信息
# print(cfg.pretty_text)
import os.path as osp
import mmcv
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.apis import inference_detector, show_result_pyplot

from dataset.kittyDataset import KittiTinyDataset

if __name__ == "__main__":
    # Build dataset
    datasets = [build_dataset(cfg.data.train)]

    # Build the detector
    model = build_detector(
        cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    # Add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES

    # Create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    train_detector(model, datasets, cfg, distributed=False, validate=True)

    # 测试训练结果
    img = mmcv.imread('../data/kitti_tiny/training/image_2/000068.jpeg')

    model.cfg = cfg
    result = inference_detector(model, img)
    show_result_pyplot(model, img, result)
