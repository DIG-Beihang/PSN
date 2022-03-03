# coding:utf-8
# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
import _init_paths
from model.gcn.gcn import gcn_adaptive_loss
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from model.utils.parser_func import parse_args
from model.utils.net_utils import clip_gradient
from utils import AverageMeter
import pdb
from model.pdomain.gp_domain_classifier import gpClassifier
from model.pdomain.cosine_classifier import gpCosineClassifier
from model.utils.config import cfg
from model.rpn.bbox_transform import clip_boxes
from model.nms.nms_wrapper import nms
from model.rpn.bbox_transform import bbox_transform_inv

args = parse_args()

AM_time = AverageMeter()
AM_loss_det = AverageMeter()
AM_loss_class = AverageMeter()
AM_loss_domain = AverageMeter()
AM_loss_gp_domain = AverageMeter()
# initilize the tensor holder here.
im_data = torch.FloatTensor(1)
im_info = torch.FloatTensor(1)
num_boxes = torch.LongTensor(1)
gt_boxes = torch.FloatTensor(1)

# ship to cuda
if args.cuda:
    im_data = im_data.cuda()
    im_info = im_info.cuda()
    num_boxes = num_boxes.cuda()
    gt_boxes = gt_boxes.cuda()

# make variable
im_data = Variable(im_data)
im_info = Variable(im_info)
num_boxes = Variable(num_boxes)
gt_boxes = Variable(gt_boxes)


def source_forward(model, data, num_classes, step):

    im_data.data.resize_(data[0].size()).copy_(data[0])
    im_info.data.resize_(data[1].size()).copy_(data[1])
    gt_boxes.data.resize_(data[2].size()).copy_(data[2])
    num_boxes.data.resize_(data[3].size()).copy_(data[3])

    pooled_feat, gt_pooled_feat, gt_labels, rois, cls_prob, bbox_pred, \
    rpn_loss_cls, rpn_loss_box, RCNN_loss_cls, RCNN_loss_bbox, \
    rois_label, domain_scores = model(im_data, im_info, gt_boxes, num_boxes)

    loss_det = rpn_loss_cls.mean() + rpn_loss_box.mean() + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
    '''
    if step > args.warmup_steps:
        for c in range(1, num_classes):
            mask = (gt_labels == c)
            if torch.sum(mask).item() > 0:
                local_prototypes = torch.mean(gt_pooled_feat[mask], 0)
                alpha = (F.cosine_similarity(sgp[c - 1], local_prototypes, dim=0).item() + 1) / 2.0
                sgp[c - 1] = (1.0 - alpha) * sgp[c - 1] + alpha * local_prototypes
    '''
    domain_labels = Variable(torch.ones_like(domain_scores).float().cuda())
    #loss_domain = F.binary_cross_entropy(domain_scores, domain_labels, (1.0 + weight_map.data))
    loss_domain = F.binary_cross_entropy(domain_scores, domain_labels)

    return loss_det, rois_label, pooled_feat, cls_prob, rois, loss_domain


def target_forward(model, data, num_classes, step):

    im_data.data.resize_(data[0].size()).copy_(data[0])
    im_info.data.resize_(data[1].size()).copy_(data[1])
    gt_boxes.data.resize_(1, 1, 5).zero_()
    num_boxes.data.resize_(1).zero_()

    rpn_feat_s, pooled_feat, cls_prob, rois, domain_scores = model(im_data, im_info, gt_boxes, num_boxes, target=True)
    '''
    import pdb
    if step >= args.warmup_steps:
        cls_prob = cls_prob.squeeze()
        _, pseudo_label = torch.max(cls_prob, 1)
        for c in range(1, num_classes):
            mask = (pseudo_label == c)
            if torch.sum(mask).item() > 0:
                #pdb.set_trace()
                local_prototypes = torch.mean(pooled_feat[mask], 0)
                alpha = (F.cosine_similarity(tgp[c - 1], local_prototypes, dim=0).item() + 1) / 2.0
                tgp[c - 1] = (1.0 - alpha) * tgp[c - 1] + alpha * local_prototypes
    domain_labels = Variable(torch.zeros_like(domain_scores).float().cuda())
    loss_domain = F.binary_cross_entropy(domain_scores, domain_labels, (1.0 + weight_map.data))
    '''
    domain_labels = Variable(torch.zeros_like(domain_scores).float().cuda())
    loss_domain = F.binary_cross_entropy(domain_scores, domain_labels)

    return pooled_feat, cls_prob, rois, loss_domain

def create_classifier(model_c):
    if type(model_c).__name__ == 'CosineClassifier':
        print('type of model_c is cosine')
        classifier = gpCosineClassifier
    elif type(model_c).__name__ == 'gpDomainClassifier':
        print('type of model_c is fc')
        classifier = gpClassifier
    else:
        print('error', type(model_c).__name__)
        raise Exception
    return classifier

def train(model, model_c, classifier, data_s, data_t, sgp, tgp, optimizers, num_classes, step):

    start_time = time.time()

    model.train()
    model_c.train()
    model.zero_grad()
    model_c.zero_grad()
    #classifier = create_classifier(model_c)
    loss_det, rois_label, pooled_feat, cls_prob, rois, loss_domain_s = source_forward(model, data_s, num_classes, step)
    tgt_pooled_feat, tgt_cls_prob, tgt_rois, loss_domain_t = target_forward(model, data_t, num_classes, step)
    
    loss_domain = torch.zeros(1).cuda()
    loss_class = torch.zeros(1).cuda()
    loss_gp_domain = torch.zeros(1).cuda()
    loss_domain = 0.5 * (loss_domain_s + loss_domain_t)
    if step >= args.warmup_steps:
        #loss_class = args.lam * step / args.max_steps * (sgp - tgp).pow(2).sum(1).mean()
        loss_intra, loss_inter, sgp, tgp, tempSP, tempTP = gcn_adaptive_loss(pooled_feat, cls_prob, rois, tgt_pooled_feat, tgt_cls_prob, tgt_rois, batch_size=1,sgp=sgp,tgp=tgp)
        loss_class = 0.5 * (loss_intra + loss_inter)
        #import pdb
        #pdb.set_trace()
        loss_gp_domain = 0.1 * classifier(model_c=model_c, sgp=tempSP[1:], tgp=tempTP[1:])
    loss = loss_det + loss_domain + loss_class + loss_gp_domain
    #loss = loss_det  + loss_class + loss_gp_domain

    optimizer, optimizer_c = optimizers
    optimizer.zero_grad()
    optimizer_c.zero_grad()
    #pdb.set_trace()
    loss.backward()
    if args.net == "vgg16":
        clip_gradient(model, 10.)

    optimizer.step()
    optimizer_c.step()


    if step % args.disp_interval == 1:
        AM_time.reset()
        AM_loss_det.reset()
        AM_loss_class.reset()
        AM_loss_domain.reset()
        AM_loss_gp_domain.reset()

    AM_time.update(time.time() - start_time)
    AM_loss_det.update(loss_det.item())
    AM_loss_class.update(loss_class.item())
    AM_loss_domain.update(loss_domain.item())
    AM_loss_gp_domain.update(loss_gp_domain.item())
    fg_cnt = torch.sum(rois_label.data.ne(0))
    bg_cnt = rois_label.data.numel() - fg_cnt

    cost_time = AM_time.avg
    losses = [AM_loss_det.avg, AM_loss_class.avg, AM_loss_domain.avg, AM_loss_gp_domain.avg]
    #losses = [AM_loss_det.avg, AM_loss_class.avg, AM_loss_gp_domain.avg]

    return sgp.detach(), tgp.detach(), fg_cnt, bg_cnt, cost_time, losses
