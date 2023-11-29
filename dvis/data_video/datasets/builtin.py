# quoted from https://github.com/aimedical/clearml-dvis-scc/blob/dev/kkumakura/DVIS/dvis/data_video/datasets/builtin.py

# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/MinVIS/blob/main/LICENSE

# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/sukjunhwang/IFC

import os
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata
from detectron2.data.datasets.coco import register_coco_instances

from .ytvis import (
    register_ytvis_instances,
    _get_ytvis_2019_instances_meta,
    _get_ytvis_2021_instances_meta,
    _get_ovis_instances_meta,
    _get_scc_4cls_instances_meta,
    _get_scc_9cls_instances_meta,
)

# ==== Predefined splits for YTVIS 2019 ===========
_PREDEFINED_SPLITS_YTVIS_2019 = {
    "ytvis_2019_train": ("ytvis_2019/train/JPEGImages",
                         "ytvis_2019/train.json"),
    "ytvis_2019_val": ("ytvis_2019/valid/JPEGImages",
                       "ytvis_2019/valid.json"),
    "ytvis_2019_test": ("ytvis_2019/test/JPEGImages",
                        "ytvis_2019/test.json"),
}


# ==== Predefined splits for YTVIS 2021 ===========
_PREDEFINED_SPLITS_YTVIS_2021 = {
    "ytvis_2021_train": ("ytvis_2021/train/JPEGImages",
                         "ytvis_2021/train.json"),
    "ytvis_2021_val": ("ytvis_2021/valid/JPEGImages",
                       "ytvis_2021/valid.json"),
    "ytvis_2021_test": ("ytvis_2021/test/JPEGImages",
                        "ytvis_2021/test.json"),
}

# ==== Predefined splits for OVIS ===========
_PREDEFINED_SPLITS_OVIS = {
    "ovis_train": ("ovis/train",
                         "ovis/annotations/annotations_train.json"),
    "ovis_val": ("ovis/valid",
                       "ovis/annotations/annotations_valid.json"),
    "ovis_test": ("ovis/test",
                        "ovis/annotations/annotations_test.json"),
}

_PREDEFINED_SPLITS_COCO_VIDEO = {
    "coco2ytvis2019_train": ("coco/train2017", "coco/annotations/coco2ytvis2019_train.json"),
    "coco2ytvis2019_val": ("coco/val2017", "coco/annotations/coco2ytvis2019_val.json"),
    "coco2ytvis2021_train": ("coco/train2017", "coco/annotations/coco2ytvis2021_train.json"),
    "coco2ytvis2021_val": ("coco/val2017", "coco/annotations/coco2ytvis2021_val.json"),
    "coco2ovis_train": ("coco/train2017", "coco/annotations/coco2ovis_train.json"),
    "coco2ovis_val": ("coco/val2017", "coco/annotations/coco2ovis_val.json"),
}

# ==== Predefined splits for SCC ===========
_PREDEFINED_SPLITS_SCC_4CLS = {
    "scc_4cls_train": ("train", "annotations/annotations_train.json"),
    "scc_4cls_val": ("valid", "annotations/annotations_valid.json"),
    "scc_4cls_test": ("valid", "annotations/annotations_valid.json"),
}

_PREDEFINED_SPLITS_SCC_9CLS = {
    "scc_9cls_train": ("train", "annotations/annotations_train.json"),
    "scc_9cls_val": ("valid", "annotations/annotations_valid.json"),
    "scc_9cls_test": ("valid", "annotations/annotations_valid.json"),
}

def register_all_ytvis_2019(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_YTVIS_2019.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_ytvis_2019_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


def register_all_ytvis_2021(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_YTVIS_2021.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_ytvis_2021_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


def register_all_ovis(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_OVIS.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_ovis_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )

def register_all_coco_video(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_COCO_VIDEO.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_coco_instances(
            key,
            _get_builtin_metadata("coco"),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )

def register_all_scc_4cls(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_SCC_4CLS.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_scc_4cls_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )

def register_all_scc_9cls(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_SCC_9CLS.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_scc_9cls_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )
        
if __name__.endswith(".builtin"):
    # Assume pre-defined datasets live in `./datasets`.
    _root = os.getenv("DETECTRON2_DATASETS", "datasets")
    # register_all_ytvis_2019(_root)
    # register_all_ytvis_2021(_root)
    # register_all_ovis(_root)
    # register_all_coco_video(_root)
    # from . import vps
    # from . import vss
    #register_all_scc_4cls("/workspace/dataset")
    #register_all_scc_9cls("/workspace/dataset")