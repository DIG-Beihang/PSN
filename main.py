# coding:utf-8
# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import pprint
import pdb
import time
import _init_paths

import torch
from torch.autograd import Variable
import torch.nn as nn
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, clip_gradient, sampler
from model.utils.parser_func import parse_args, set_dataset_args
from datasets.config_dataset import cfg_d
from utils import WeightEMA, adjust_learning_rate, log
from train import train
from model.pdomain.gp_domain_classifier import gpDomainClassifier
from model.pdomain.cosine_classifier import CosineClassifier
from model.pdomain.gp_domain_classifier import gpClassifier
from model.pdomain.cosine_classifier import gpCosineClassifier

if __name__ == '__main__':

    feat_dim = 4096
    args = parse_args()
    args = set_dataset_args(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    cfg.CUDA = args.cuda
    cfg.USE_GPU_NMS = args.cuda
    cfg.TRAIN.USE_FLIPPED = True
    cfg_d.PREFIX = args.d_prefix

    # pprint.pprint(cfg)

    if torch.cuda.is_available() and not args.cuda:
        log("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # source dataset
    imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name)
    train_size = len(roidb)

    # target dataset
    imdb_t, roidb_t, ratio_list_t, ratio_index_t = combined_roidb(args.imdb_name_target, target=True)
    train_size_t = len(roidb_t)

    log('{:d} source roidb entries'.format(len(roidb)))
    log('{:d} target roidb entries'.format(len(roidb_t)))

    output_dir = args.save_dir + "/" + args.net + "/" + args.dataset + args.out
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sampler_batch = sampler(train_size, args.batch_size)
    sampler_batch_t = sampler(train_size_t, args.batch_size)

    dataset_s = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, imdb.num_classes, training=True, path_return=True)
    dataloader_s = torch.utils.data.DataLoader(dataset_s, batch_size=args.batch_size, sampler=sampler_batch,
                                                num_workers=args.num_workers, pin_memory=True)
    dataset_t = roibatchLoader(roidb_t, ratio_list_t, ratio_index_t, args.batch_size, imdb.num_classes, training=True, path_return=True)
    dataloader_t = torch.utils.data.DataLoader(dataset_t, batch_size=args.batch_size, sampler=sampler_batch_t,
                                                num_workers=args.num_workers, pin_memory=True)

    data_iter_s = iter(dataloader_s)
    data_iter_t = iter(dataloader_t)

    # initilize the network here.
    from model.faster_rcnn.vgg16 import vgg16
    model = vgg16(imdb.classes, pretrained=True, class_agnostic=args.class_agnostic)
    model.create_architecture()
    if args.model_c == 'fc':
        print('using fc classifier')
        model_c = gpDomainClassifier(feat_dim * (imdb.num_classes - 1), imdb.num_classes - 1)
        classifier = gpClassifier
    elif args.model_c == 'cosine':
        print('using cosine classifier')
        model_c = CosineClassifier(feat_dim, 2)
        classifier = gpCosineClassifier
    else:
        print('model_c must be "fc" or "cosine"!')
        raise Exception

    params = []
    params_cls = []
    lr = args.lr

    for key, value in model.named_parameters():
        print(key)
        if 'classifier' in key:
            params_cls += [{'params': [value], 'lr': 5e-5, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
        elif value.requires_grad:
            if 'bias' in key:
                params += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                      'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
            else:
                params += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
    for key, value in model_c.named_parameters():
        print(key)
        params_cls += [{'params': [value], 'lr': 5e-5, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
        '''
        if value.requires_grad:
            if 'bias' in key:
                params_cls += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                      'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
            else:
                params_cls += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
        '''
    #params = list(model.parameters())

    optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)
    optimizer_c = torch.optim.Adam(params_cls)
    optimizers = [optimizer, optimizer_c]
    #optimizers = optimizer


    # 初始化原型
    if args.prior:
        log('[*] Use Prior prototypes!')
        sgp = torch.from_numpy(np.load(args.sp)).type(torch.float32)
        tgp = torch.from_numpy(np.load(args.tp)).type(torch.float32)
    else:
        sgp = torch.zeros(imdb.num_classes - 1, feat_dim).type(torch.float32)
        tgp = torch.zeros(imdb.num_classes - 1, feat_dim).type(torch.float32)

    if args.cuda:
        sgp = sgp.cuda()
        tgp = tgp.cuda()
        #pdb.set_trace()
        model.cuda()
        model_c.cuda()
    lr_idx = 0

    if args.resume:
        checkpoint = torch.load(args.load_name)
        args.start_step = checkpoint['step']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        model_c.load_state_dict(checkpoint['model_c'])
        optimizer_c.load_state_dict(checkpoint['optimizer_c'])
        lr = optimizer.param_groups[0]['lr']
        lr_idx = checkpoint['lr_idx']
        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']
        log("[*] loaded checkpoint {} at Step {}".format(args.load_name, args.start_step - 1))

    if args.use_tfboard:
        from tensorboardX import SummaryWriter
        logger = SummaryWriter("logs", flush_secs=10)

#    print('111111111')
    for step in range(args.start_step, args.max_steps + 1):

        if step in args.lr_decay_step:
            lr_decay_gamma = args.lr_decay_gamma[lr_idx]
            adjust_learning_rate(optimizer, lr_decay_gamma)
            lr *= lr_decay_gamma
            lr_idx += 1
#        print('222222222')
        while True:
            try:
                data_s = next(data_iter_s)
            except:
                data_iter_s = iter(dataloader_s)
                data_s = next(data_iter_s)
            if data_s[3].sum() != 0:
                break
            else:
                print(data_s[4])
        try:
            data_t = next(data_iter_t)
        except:
            data_iter_t = iter(dataloader_t)
            data_t = next(data_iter_t)


#        print('3333333')
        sgp, tgp, fg_cnt, bg_cnt, cost_time, losses = train(model, model_c, classifier, data_s, data_t, \
                                                            sgp, tgp, optimizers, imdb.num_classes, step)
        loss_det, loss_class, loss_domain, loss_gp_domain = losses
        #loss_det, loss_class, loss_gp_domain = losses

        if step % args.disp_interval == 0:

            log("[iter {:5d}/{:5d}] lr: {:.2e} fg/bg=({}/{}) time cost: {:.2f} s/it".format(
                step, args.max_steps, lr, fg_cnt, bg_cnt, cost_time))
            #print("\t\t\tloss_det: {:.4f} loss_domain: {:.4f} loss_class: {:.4f} loss_gp_domain: {:.4f}".format(loss_det, loss_domain, loss_class, loss_gp_domain))
            print("\t\t\tloss_det: {:.4f} loss_domain: {:.4f} loss_class: {:.4f} loss_gp_domain: {:.4f}".format(loss_det, loss_domain, loss_class, loss_gp_domain))

            if args.use_tfboard:
                logger.add_scalars("{}/losses".format(args.log_dir), {
                    'loss_det': loss_det,
                    'loss_class': loss_class,
                    'loss_domain': loss_domain,
                    'loss_gp_domain': loss_gp_domain
                }, step)

 #       print('4')
        if (step % 5000 == 0) or (step >= args.lr_decay_step[-1] and step % 2000 == 0):
            save_name = os.path.join(output_dir, 'target_{}_step_{}.pth'.format(args.dataset_t, step))
            torch.save({
                'step': step + 1,
                'lr_idx': lr_idx,
                'sgp':sgp,
                'tgp':tgp,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_c':model_c.state_dict(),
                'optimizer_c': optimizer_c.state_dict(),
                'pooling_mode': cfg.POOLING_MODE,
                'class_agnostic': args.class_agnostic,
            }, save_name)
            log('[*] Save Model at Step {}'.format(step))
#        print('5')
#    print('6')
    if args.use_tfboard:
        logger.close()
