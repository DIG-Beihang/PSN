import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from model.utils.config import cfg
from model.rpn.rpn import _RPN
from model.roi_pooling.modules.roi_pool import _RoIPooling
from model.roi_crop.modules.roi_crop import _RoICrop
from model.roi_align.modules.roi_align import RoIAlignAvg
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
import time
import pdb
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta, grad_reverse


class _fasterRCNN(nn.Module):
    """ faster RCNN """
    def __init__(self, classes, class_agnostic):
        super(_fasterRCNN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
        self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)

        self.grid_size = cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE
        self.RCNN_roi_crop = _RoICrop()

    def forward(self, im_data, im_info, gt_boxes, num_boxes, path='', target=False, teacher=False):

        batch_size = im_data.size(0)
        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        # feed image data to base model to obtain base feature map
        base_feat1 = self.RCNN_base1(im_data)
        base_feat2 = self.RCNN_base2(base_feat1)
        base_feat3 = self.RCNN_base3(base_feat2)
        if self.training:
            ds1 = self.Domain_classifier1(grad_reverse(base_feat1))
            ds2 = self.Domain_classifier2(grad_reverse(base_feat2))
            ds3 = self.Domain_classifier3(grad_reverse(base_feat3))

            domain_scores = torch.cat((ds1.view(batch_size, -1), \
                                       ds2.view(batch_size, -1), \
                                       ds3.view(batch_size, -1)), 1)

        # feed base feature map tp RPN to obtain rois
        rois, rpn_loss_cls, rpn_loss_bbox, rpn_feat, attention_map = self.RCNN_rpn(base_feat3, im_info, gt_boxes, num_boxes, target=target)
        '''
        import pdb 
        #pdb.set_trace()
        if self.training:
            at1 = F.upsample(attention_map, size=base_feat1.shape[2:4], mode='bilinear', align_corners=False)
            at2 = F.upsample(attention_map, size=base_feat2.shape[2:4], mode='bilinear', align_corners=False)
            at3 = F.upsample(attention_map, size=base_feat3.shape[2:4], mode='bilinear', align_corners=False)

            attention_scores = torch.cat((at1.view(batch_size, -1), \
                                            at2.view(batch_size, -1), \
                                            at3.view(batch_size, -1)), 1)

        '''
        if self.training and not target:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data
            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

        rois = Variable(rois)
        pooled_feat = self.RCNN_roi_align(base_feat3, rois.view(-1, 5))
        pooled_feat = self._head_to_tail(pooled_feat)


        if self.training and not target:

            ind = (gt_boxes[..., 4] != 0)
            gt_boxes_filtered = torch.tensor(gt_boxes[ind, ...])

            gt_labels = torch.tensor(gt_boxes_filtered[..., 4])
            
            gt_boxes_filtered[..., 4].zero_()
            gt_boxes_filtered = gt_boxes_filtered[..., [4, 0, 1, 2, 3]]

            gt_pooled_feat = self.RCNN_roi_align(base_feat3, gt_boxes_filtered.view(-1, 5))
            gt_pooled_feat = self._head_to_tail(gt_pooled_feat)


        if not target or not self.training:
            bbox_pred = self.RCNN_bbox_pred(pooled_feat)
            if self.training and not self.class_agnostic:
                bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
                bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
                bbox_pred = bbox_pred_select.squeeze(1)

        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score, 1)
        import pdb
        #pdb.set_trace()
        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        if self.training and not target:
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)

        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)

        if not target or not self.training:
            bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)


        if self.training:
            if target:
                return rpn_feat, pooled_feat, cls_prob, rois, domain_scores
            else:
                return pooled_feat, gt_pooled_feat, gt_labels, rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, \
                       RCNN_loss_cls, RCNN_loss_bbox, rois_label, domain_scores
        else:
            #pdb.set_trace()
            return rois, cls_prob, bbox_pred, attention_map


    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()
