# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__sets = {}

from datasets.sim10k import sim10k
from datasets.sim10k_fog import sim10k_fog
from datasets.sim10k_bright import sim10k_bright
from datasets.domain1 import domain1
from datasets.domain2 import domain2
from datasets.domain3 import domain3
from datasets.cityscape import cityscape
from datasets.cityscape_fog import cityscape_fog
from datasets.cityscape_elastic import cityscape_elastic
from datasets.cityscape_vgg import cityscape_vgg
from datasets.cityscape_res import cityscape_res
from datasets.cityscape_den import cityscape_den
from datasets.cityscape_car import cityscape_car
from datasets.foggy_cityscape import foggy_cityscape
from datasets.cityscape_kitti import cityscape_kitti
from datasets.kitti import kitti

import numpy as np
for split in ['train', 'trainval','val', 'test']:
  name = 'cityscape_{}'.format(split)
  __sets[name] = (lambda split=split : cityscape(split))
for split in ['train', 'trainval','val', 'test']:
  name = 'cityscape_fog_{}'.format(split)
  __sets[name] = (lambda split=split : cityscape_fog(split))
for split in ['train', 'trainval','val', 'test']:
  name = 'cityscape_vgg_{}'.format(split)
  __sets[name] = (lambda split=split : cityscape_vgg(split))
for split in ['train', 'trainval','val', 'test']:
  name = 'cityscape_res_{}'.format(split)
  __sets[name] = (lambda split=split : cityscape_res(split))
for split in ['train', 'trainval','val', 'test']:
  name = 'cityscape_den_{}'.format(split)
  __sets[name] = (lambda split=split : cityscape_den(split))
for split in ['train', 'trainval','val', 'test']:
  name = 'cityscape_elastic_{}'.format(split)
  __sets[name] = (lambda split=split : cityscape_elastic(split))
for split in ['train', 'trainval','val', 'test']:
  name = 'cityscape_car_{}'.format(split)
  __sets[name] = (lambda split=split : cityscape_car(split))
for split in ['train', 'trainval', 'test']:
  name = 'cityscape_kitti_{}'.format(split)
  __sets[name] = (lambda split=split : cityscape_kitti(split))
for split in ['train', 'trainval', 'val','test']:
  name = 'foggy_cityscape_{}'.format(split)
  __sets[name] = (lambda split=split : foggy_cityscape(split))
for split in ['train', 'trainval','val', 'test']:
  name = 'sim10k_{}'.format(split)
  __sets[name] = (lambda split=split : sim10k(split))
for split in ['train', 'trainval','val', 'test']:
  name = 'sim10k_fog_{}'.format(split)
  __sets[name] = (lambda split=split : sim10k_fog(split))
for split in ['train', 'trainval','val', 'test']:
  name = 'sim10k_bright_{}'.format(split)
  __sets[name] = (lambda split=split : sim10k_bright(split))
for split in ['train', 'trainval','val', 'test']:
  name = 'kitti_{}'.format(split)
  __sets[name] = (lambda split=split : kitti(split))
for split in ['train']:
  name = 'domain1_{}'.format(split)
  __sets[name] = (lambda split=split : domain1(split))
for split in ['train']:
  name = 'domain2_{}'.format(split)
  __sets[name] = (lambda split=split : domain2(split))
for split in ['train']:
  name = 'domain3_{}'.format(split)
  __sets[name] = (lambda split=split : domain3(split))


def get_imdb(name):
  """Get an imdb (image database) by name."""
  if name not in __sets:
    raise KeyError('Unknown dataset: {}'.format(name))
  return __sets[name]()


def list_imdbs():
  """List all registered imdbs."""
  return list(__sets.keys())
