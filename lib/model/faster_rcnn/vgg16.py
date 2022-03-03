# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable

from model.faster_rcnn.faster_rcnn import _fasterRCNN
from model.utils.config import cfg


def conv3x3(in_planes, out_planes, stride=1):
  "3x3 convolution with padding"
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
           padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
  "1x1 convolution without padding"
  return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
           padding=0, bias=False)

class netD_pixel(nn.Module):

    def __init__(self):
        super(netD_pixel, self).__init__()
        self.conv1 = conv1x1(256, 256)
        #self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = conv1x1(256, 128)
        #self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = conv1x1(128, 1)

    def forward(self, x):
        x = F.relu(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.sigmoid(self.conv3(x))
        return x.view(-1,1)#F.sigmoid(x)

class netD(nn.Module):

    def __init__(self):
        super(netD, self).__init__()
        self.conv1 = conv3x3(512, 512, stride=2)
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = conv3x3(512, 128, stride=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = conv3x3(128, 128, stride=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc = nn.Linear(128,2)

    def forward(self, x):
        x = F.dropout(F.relu(self.bn1(self.conv1(x))),training=self.training)
        x = F.dropout(F.relu(self.bn2(self.conv2(x))),training=self.training)
        x = F.dropout(F.relu(self.bn3(self.conv3(x))),training=self.training)
        x = F.avg_pool2d(x,(x.size(2),x.size(3)))
        x = x.view(-1,128)
        x = self.fc(x)
        return x


class Discriminator(nn.Module):

    def __init__(self, dim):
        super(Discriminator, self).__init__()
        self.conv1 = conv1x1(dim, 256)
        self.in1 = nn.InstanceNorm2d(256)
        self.conv2 = conv1x1(256, 128)
        self.in2 = nn.InstanceNorm2d(128)
        self.conv3 = conv1x1(128, 1)

    def forward(self, x):
        x = F.leaky_relu(self.in1(self.conv1(x)))
        x = F.leaky_relu(self.in2(self.conv2(x)))
        x = F.sigmoid(self.conv3(x))
        # x = x.view(x.size(0), -1)
        return x



class vgg16(_fasterRCNN):
  def __init__(self, classes, pretrained=False, class_agnostic=False):
    self.model_path = cfg.VGG_PATH
    print(self.model_path)
    self.dout_base_model = 512
    self.pretrained = pretrained
    self.class_agnostic = class_agnostic

    _fasterRCNN.__init__(self, classes, class_agnostic)

  def _init_modules(self):
    vgg = models.vgg16()
    if self.pretrained:
        print("Loading pretrained weights from %s" %(self.model_path))
        state_dict = torch.load(self.model_path)
        vgg.load_state_dict({k:v for k,v in state_dict.items() if k in vgg.state_dict()})

    vgg.classifier = nn.Sequential(*list(vgg.classifier._modules.values())[:-1])

    # not using the last maxpool layer
    #print(vgg.features)
    self.RCNN_base1 = nn.Sequential(*list(vgg.features._modules.values())[:17]) # 256
    self.RCNN_base2 = nn.Sequential(*list(vgg.features._modules.values())[17:24]) # 512
    self.RCNN_base3 = nn.Sequential(*list(vgg.features._modules.values())[24:-1]) # 512
    self.Domain_classifier1 = Discriminator(256)
    self.Domain_classifier2 = Discriminator(512)
    self.Domain_classifier3 = Discriminator(512)
    feat_d = 4096

    # Fix the layers before conv3:
    for layer in range(10):
          for p in self.RCNN_base1[layer].parameters(): 
              p.requires_grad = False

    # self.RCNN_base = _RCNN_base(vgg.features, self.classes, self.dout_base_model)

    self.RCNN_top = vgg.classifier
    self.RCNN_cls_score = nn.Linear(feat_d, self.n_classes)

    if self.class_agnostic:
        self.RCNN_bbox_pred = nn.Linear(feat_d, 4)
    else:
        self.RCNN_bbox_pred = nn.Linear(feat_d, 4 * self.n_classes)

  def _head_to_tail(self, pool5):

    pool5_flat = pool5.view(pool5.size(0), -1)
    fc7 = self.RCNN_top(pool5_flat)

    return fc7

