import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from torch.autograd import Variable
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta, grad_reverse

def gpCosineClassifier(model_c, sgp, tgp):
    #gpdc = gpDomainClassifier(sgp.shape[1] * sgp.shape[0], sgp.shape[0]).cuda()
    #type(sgp.cuda())
    #pdb.set_trace()
    sgp_scores = model_c(grad_reverse(sgp))
    tgp_scores = model_c(grad_reverse(tgp))
    #pdb.set_trace()    
    sgp_labels = Variable(torch.ones(sgp.shape[0]).long().cuda())
    loss_sgp_domain = nn.CrossEntropyLoss()(sgp_scores, sgp_labels)
    tgp_labels = Variable(torch.zeros(tgp.shape[0]).long().cuda())
    loss_tgp_domain = nn.CrossEntropyLoss()(tgp_scores, tgp_labels)
    #pdb.set_trace()
    loss_gp_domain = (loss_sgp_domain + loss_tgp_domain) / 2
    return loss_gp_domain

def cosine_fully_connected_layer(
    x_in, weight, scale=None, bias=None, normalize_x=True, normalize_w=True
):
    # x_in: a 2D tensor with shape [batch_size x num_features_in]
    # weight: a 2D tensor with shape [num_features_in x num_features_out]
    # scale: (optional) a scalar value
    # bias: (optional) a 1D tensor with shape [num_features_out]

    assert x_in.dim() == 2
    assert weight.dim() == 2
    assert x_in.size(1) == weight.size(0)

    if normalize_x:
        x_in = F.normalize(x_in, p=2, dim=1, eps=1e-12)

    if normalize_w:
        weight = F.normalize(weight, p=2, dim=0, eps=1e-12)

    x_out = torch.mm(x_in, weight)

    if scale is not None:
        x_out = x_out * scale.view(1, -1)

    if bias is not None:
        x_out = x_out + bias.view(1, -1)

    return x_out


class CosineClassifier(nn.Module):
    def __init__(
        self,
        num_channels,
        num_classes,
        scale=20.0,
        learn_scale=False,
        bias=False,
        normalize_x=True,
        normalize_w=True,
    ):
        super().__init__()

        self.num_channels = num_channels
        self.num_classes = num_classes
        self.normalize_x = normalize_x
        self.normalize_w = normalize_w

        weight = torch.FloatTensor(num_classes, num_channels).normal_(
            0.0, np.sqrt(2.0 / num_channels)
        )
        self.weight = nn.Parameter(weight, requires_grad=True)

        if bias:
            bias = torch.FloatTensor(num_classes).fill_(0.0)
            self.bias = nn.Parameter(bias, requires_grad=True)
        else:
            self.bias = None

        scale_cls = torch.FloatTensor(1).fill_(scale)
        self.scale_cls = nn.Parameter(scale_cls, requires_grad=learn_scale)

    def forward(self, x_in):
        assert x_in.dim() == 2
        return cosine_fully_connected_layer(
            x_in,
            self.weight.t(),
            scale=self.scale_cls,
            bias=self.bias,
            normalize_x=self.normalize_x,
            normalize_w=self.normalize_w,
        )

    def extra_repr(self):
        s = "num_channels={}, num_classes={}, scale_cls={} (learnable={})".format(
            self.num_channels,
            self.num_classes,
            self.scale_cls.item(),
            self.scale_cls.requires_grad,
        )
        learnable = self.scale_cls.requires_grad
        s = (
            "num_channels={self.num_channels}, "
            "num_classes={self.num_classes}, "
            "scale_cls={self.scale_cls.item()} (learnable={learnable}), "
            "normalize_x={self.normalize_x}, normalize_w={self.normalize_w}"
        )

        if self.bias is None:
            s += ", bias=False"
        return s

