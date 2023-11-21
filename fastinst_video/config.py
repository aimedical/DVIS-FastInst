# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.config import CfgNode as CN


def add_fastinst_video_config(cfg):
    # referenced https://github.com/zhang-tao-whu/DVIS/blob/main/mask2former_video/config.py
    # video data
    # DataLoader
    cfg.INPUT.SAMPLING_FRAME_NUM = 2
    cfg.INPUT.SAMPLING_FRAME_RANGE = 20
    cfg.INPUT.SAMPLING_FRAME_SHUFFLE = False
    cfg.INPUT.AUGMENTATIONS = [] # "brightness", "contrast", "saturation", "rotation"

    # DVISとMASK_FORMERは密結合しているため、仕方なく、MASK_FORMERの設定を流用する
    # referenced https://github.com/zhang-tao-whu/DVIS/blob/main/mask2former/config.py
    # mask_former model config
    cfg.MODEL.MASK_FORMER = CN()

    # loss
    cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION = cfg.MODEL.FASTINST.DEEP_SUPERVISION
    cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT = cfg.MODEL.FASTINST.NO_OBJECT_WEIGHT
    cfg.MODEL.MASK_FORMER.CLASS_WEIGHT = cfg.MODEL.FASTINST.CLASS_WEIGHT
    cfg.MODEL.MASK_FORMER.DICE_WEIGHT = cfg.MODEL.FASTINST.DICE_WEIGHT
    cfg.MODEL.MASK_FORMER.MASK_WEIGHT = cfg.MODEL.FASTINST.MASK_WEIGHT

    # transformer config
    cfg.MODEL.MASK_FORMER.NHEADS = cfg.MODEL.FASTINST.NHEADS
    cfg.MODEL.MASK_FORMER.DROPOUT = cfg.MODEL.FASTINST.DROPOUT
    cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD = cfg.MODEL.FASTINST.DIM_FEEDFORWARD
    cfg.MODEL.MASK_FORMER.DEC_LAYERS = cfg.MODEL.FASTINST.DEC_LAYERS
    cfg.MODEL.MASK_FORMER.PRE_NORM = cfg.MODEL.FASTINST.PRE_NORM

    cfg.MODEL.MASK_FORMER.HIDDEN_DIM = cfg.MODEL.FASTINST.HIDDEN_DIM
    cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = cfg.MODEL.FASTINST.NUM_OBJECT_QUERIES

    # mask_former inference config
    cfg.MODEL.MASK_FORMER.TEST = CN()
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = cfg.MODEL.FASTINST.TEST.SEMANTIC_ON
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = cfg.MODEL.FASTINST.TEST.INSTANCE_ON
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = cfg.MODEL.FASTINST.TEST.PANOPTIC_ON
    cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD = cfg.MODEL.FASTINST.TEST.OBJECT_MASK_THRESHOLD
    cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD = cfg.MODEL.FASTINST.TEST.OVERLAP_THRESHOLD
    cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE = cfg.MODEL.FASTINST.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE

    # Sometimes `backbone.size_divisibility` is set to 0 for some backbone (e.g. ResNet)
    # you can use this config to override
    cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY = cfg.MODEL.FASTINST.SIZE_DIVISIBILITY

    # NOTE: maskformer2 extra configs
    # transformer module
    cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME = cfg.MODEL.FASTINST.TRANSFORMER_DECODER_NAME

    # point loss configs
    # Number of points sampled during training for a mask point head.
    cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS = cfg.MODEL.FASTINST.TRAIN_NUM_POINTS
    # Oversampling parameter for PointRend point sampling during training. Parameter `k` in the
    # original paper.
    cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO = cfg.MODEL.FASTINST.OVERSAMPLE_RATIO
    # Importance sampling parameter for PointRend point sampling during training. Parametr `beta` in
    # the original paper.
    cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO = cfg.MODEL.FASTINST.IMPORTANCE_SAMPLE_RATIO