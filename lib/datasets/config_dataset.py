from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import numpy as np
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict
from model.utils.parser_func import parse_args

args = parse_args()
__D = edict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg_d = __D

__D.PREFIX = '/media/datasets/cityscape_and_foggycityscape/'

#
# Training options
#with regard to pascal, the directories under the path will be ./VOC2007, ./VOC2012"
# __D.PASCAL = os.path.join(__D.PREFIX, "pascal_voc/VOCdevkit")
# __D.PASCALCLIP = ""
# __D.PASCALWATER = "/VOCdevkit"

#For these datasets, the directories under the path will be Annotations  ImageSets  JPEGImages."

# __D.CLIPART = os.path.join(__D.PREFIX, "clipart")
# __D.WATER = os.path.join(__D.PREFIX, "watercolor")

__D.CITYSCAPE = os.path.join(__D.PREFIX, "cityscapes/VOC2007")
__D.FOGGYCITY = os.path.join(__D.PREFIX, "cityscapes_foggy/VOC2007")
__D.SIM10K_FOG = "/media/datasets/sim10k_fog/VOC2012"
__D.SIM10K_BRIGHT = "/media/datasets/sim10k_bright/VOC2012"
__D.CITYSCAPE_FOG = "/media/datasets/cityscape_fog/VOC2007"
__D.CITYSCAPE_VGG = "/media/datasets/cityscape_from_official/noise_vgg16bn_500/VOC2007"
__D.CITYSCAPE_RES = "/media/datasets/cityscape_from_official/noise_resnet50_0/VOC2007"
__D.CITYSCAPE_DEN = "/media/datasets/cityscape_from_official/noise_densenet161_999/VOC2007"
__D.CITYSCAPE_ELASTIC = "/media/datasets/cityscape_elastic/VOC2007"
#__D.KITTI = os.path.join(__D.PREFIX, "kitti/VOC2007")
__D.DOMAIN1 = "/media/datasets/AAAI2022/0813domainVOC/domain1"
__D.DOMAIN2 = "/media/datasets/AAAI2022/0813domainVOC/domain2"
__D.DOMAIN3 = "/media/datasets/AAAI2022/0813domainVOC/domain3"

def _merge_a_into_b(a, b):
  """Merge config dictionary a into config dictionary b, clobbering the
  options in b whenever they are also specified in a.
  """
  if type(a) is not edict:
    return

  for k, v in a.items():
    # a must specify keys that are in b
    if k not in b:
      raise KeyError('{} is not a valid config key'.format(k))

    # the types must match, too
    old_type = type(b[k])
    if old_type is not type(v):
      if isinstance(b[k], np.ndarray):
        v = np.array(v, dtype=b[k].dtype)
      else:
        raise ValueError(('Type mismatch ({} vs. {}) '
                          'for config key: {}').format(type(b[k]),
                                                       type(v), k))

    # recursively merge dicts
    if type(v) is edict:
      try:
        _merge_a_into_b(a[k], b[k])
      except:
        print(('Error under config key: {}'.format(k)))
        raise
    else:
      b[k] = v


def cfg_from_file(filename):
  """Load a config file and merge it into the default options."""
  import yaml
  with open(filename, 'r') as f:
    yaml_cfg = edict(yaml.load(f))

  _merge_a_into_b(yaml_cfg, __D)


def cfg_from_list(cfg_list):
  """Set config keys via list (e.g., from command line)."""
  from ast import literal_eval
  assert len(cfg_list) % 2 == 0
  for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
    key_list = k.split('.')
    d = __D
    for subkey in key_list[:-1]:
      assert subkey in d
      d = d[subkey]
    subkey = key_list[-1]
    assert subkey in d
    try:
      value = literal_eval(v)
    except:
      # handle the case when v is a string literal
      value = v
    assert type(value) == type(d[subkey]), \
      'type {} does not match original type {}'.format(
        type(value), type(d[subkey]))
    d[subkey] = value
