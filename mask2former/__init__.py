# Copyright (c) Facebook, Inc. and its affiliates.
# TODO: data系をimportするとfastinstの学習時にはエラーが消えるけど、おそらくmask2formerの学習時にエラーが起きる。mask2formerも動くようにする
# TODO: cocoのregieterをif文で、2重登録を防止するようにする
# from . import data  # register all new datasets
from . import modeling

# config
from .config import add_maskformer2_config

# dataset loading
# from .data.dataset_mappers.coco_instance_new_baseline_dataset_mapper import COCOInstanceNewBaselineDatasetMapper
# from .data.dataset_mappers.coco_panoptic_new_baseline_dataset_mapper import COCOPanopticNewBaselineDatasetMapper
# from .data.dataset_mappers.mask_former_instance_dataset_mapper import (
#     MaskFormerInstanceDatasetMapper,
# )
# from .data.dataset_mappers.mask_former_panoptic_dataset_mapper import (
#     MaskFormerPanopticDatasetMapper,
# )
# from .data.dataset_mappers.mask_former_semantic_dataset_mapper import (
#     MaskFormerSemanticDatasetMapper,
# )

# models
from .maskformer_model import MaskFormer
from .test_time_augmentation import SemanticSegmentorWithTTA

# evaluation
from .evaluation.instance_evaluation import InstanceSegEvaluator
