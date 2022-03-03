import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from datetime import datetime


from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
from model.nms.nms_wrapper import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.parser_func import parse_args, set_dataset_args


try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


class AverageMeter(object):
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val   = 0
        self.avg   = 0
        self.sum   = 0
        self.count = 0

    def update(self, val, n=1):
        self.val   = val
        self.sum   += val * n
        self.count += n
        self.avg   = self.sum / self.count


class WeightEMA (object):
    """
    Exponential moving average weight optimizer for mean teacher model
    """
    def __init__(self, params, src_params, alpha, max_step=70000):
        self.params = list(params)
        self.src_params = list(src_params)
        self.alpha = alpha
        self.max_step = max_step
        self.cur_step = 1

        for p, src_p in zip(self.params, self.src_params):
            p.data[:] = src_p.data[:]

    def step(self):
        # alpha = min(self.alpha, 1.0 * self.cur_step / self.max_step)
        alpha = self.alpha
        # print(alpha)
        for p, src_p in zip(self.params, self.src_params):
            p.data.mul_(alpha)
            p.data.add_(src_p.data * (1.0 - alpha))
        self.cur_step += 1

def adjust_learning_rate(optimizer, decay=0.1):
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay * param_group['lr']


def log(msg, end='\n'):
    print('{}: {}'.format(datetime.now().strftime("%Y/%m/%d %H:%M:%S"), msg), end=end)


class EMA (object):
    """
    Exponential moving average weight optimizer for mean teacher model
    """
    def __init__(self, params, src_params, alpha, beta, max_step=70000):
        self.params = list(params)
        self.src_params = list(src_params)
        self.alpha = alpha
        self.beta = beta
        self.max_step = max_step
        self.cur_step = 1

        for p, src_p in zip(self.params, self.src_params):
            p.data[:] = src_p.data[:]

    def step(self):
        alpha = 2.0 / (1.0 + np.exp(-self.beta * self.cur_step / self.max_step)) - 1.0
        alpha = min(alpha, self.alpha)
        # print('{}: {}'.format(self.beta, alpha))
        for p, src_p in zip(self.params, self.src_params):
            p.data.mul_(alpha)
            p.data.add_(src_p.data * (1.0 - alpha))
        self.cur_step += 1