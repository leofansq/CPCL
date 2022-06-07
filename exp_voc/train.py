from __future__ import division
import os
import sys
import time
import argparse

from tqdm import tqdm
import numpy as np 
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from config import config
from dataloader import get_train_loader
from network import Network
from dataloader import VOC
from utils.init_func import init_weight, group_weight
from engine.lr_policy import WarmUpPolyLR
from engine.engine import Engine


if os.getenv('debug') is not None:
    is_debug = os.environ['debug']
else:
    is_debug = False
os.environ["CUDA_VISIBLE_DEVICES"] = config.device
logfile = open(config.train_log_file, 'a')
parser = argparse.ArgumentParser()


############################################
#                  CutMix                  #
############################################
import mask_gen
from custom_collate import SegCollate
mask_generator = mask_gen.BoxMaskGenerator(prop_range=config.cutmix_mask_prop_range,
                                           n_boxes=config.cutmix_boxmask_n_boxes,
                                           random_aspect_ratio=not config.cutmix_boxmask_fixed_aspect_ratio,
                                           prop_by_area=not config.cutmix_boxmask_by_size,
                                           within_bounds=not config.cutmix_boxmask_outside_bounds,
                                           invert=not config.cutmix_boxmask_no_invert)
add_mask_params_to_batch = mask_gen.AddMaskParamsToBatch(mask_generator)
collate_fn = SegCollate()
mask_collate_fn = SegCollate(batch_aug_fn=add_mask_params_to_batch)


############################################
#                   Main                   #
############################################
with Engine(custom_parser=parser) as engine:
    args = parser.parse_args()

    cudnn.benchmark = True

    seed = config.seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Dataloader (Sup & Unsup)
    train_loader, train_sampler = get_train_loader(engine, VOC, train_source=config.train_source, unsupervised=False, collate_fn=collate_fn)
    unsupervised_train_loader_0, unsupervised_train_sampler_0 = get_train_loader(engine, VOC, train_source=config.unsup_source, unsupervised=True, collate_fn=mask_collate_fn)
    unsupervised_train_loader_1, unsupervised_train_sampler_1 = get_train_loader(engine, VOC, train_source=config.unsup_source, unsupervised=True, collate_fn=collate_fn)

    # Criterion
    criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=255)

    # Model
    model = Network(config.num_classes, criterion=criterion, pretrained_model=config.pretrained_model, norm_layer=nn.BatchNorm2d)
    init_weight(model.branch1.business_layer, nn.init.kaiming_normal_, nn.BatchNorm2d, config.bn_eps, config.bn_momentum, mode='fan_in', nonlinearity='relu')
    init_weight(model.branch2.business_layer, nn.init.kaiming_normal_, nn.BatchNorm2d, config.bn_eps, config.bn_momentum, mode='fan_in', nonlinearity='relu')

    # Optimizer
    params_list_1 = []
    params_list_1 = group_weight(params_list_1, model.branch1.backbone, nn.BatchNorm2d, config.lr)
    for module in model.branch1.business_layer:
        params_list_1 = group_weight(params_list_1, module, nn.BatchNorm2d, config.lr)
    optimizer_1 = torch.optim.SGD(params_list_1, lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)

    params_list_2 = []
    params_list_2 = group_weight(params_list_2, model.branch2.backbone, nn.BatchNorm2d, config.lr)
    for module in model.branch2.business_layer:
        params_list_2 = group_weight(params_list_2, module, nn.BatchNorm2d, config.lr)

    optimizer_2 = torch.optim.SGD(params_list_2, lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)


    # LearningRate Policy
    total_iteration = config.nepochs * config.niters_per_epoch
    lr_policy = WarmUpPolyLR(config.lr, config.lr_power, total_iteration, config.niters_per_epoch * config.warm_up_epoch)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    engine.register_state(dataloader=train_loader, model=model, optimizer_1=optimizer_1, optimizer_2=optimizer_2)
    if engine.continue_state_object: engine.restore_checkpoint()

    # Init
    label = [i for i in range(21)]
    ############################################
    #              Begin Training              #
    ############################################
    model.train()
    print ("-----------------------------------------------------------")
    print ('Start Training... ...')

    for epoch in range(engine.state.epoch, config.nepochs):

        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        if is_debug:
            pbar = tqdm(range(10), file=sys.stdout, bar_format=bar_format)
        else:
            pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout, bar_format=bar_format)

        # Initialize Dataloader
        dataloader = iter(train_loader)
        unsupervised_dataloader_0 = iter(unsupervised_train_loader_0)
        unsupervised_dataloader_1 = iter(unsupervised_train_loader_1)

        "Training"
        for idx in pbar:
            optimizer_1.zero_grad()
            optimizer_2.zero_grad()
            engine.update_iteration(epoch, idx)
            start_time = time.time()

            # Load the data
            minibatch = dataloader.next()
            unsup_minibatch_0 = unsupervised_dataloader_0.next()
            unsup_minibatch_1 = unsupervised_dataloader_1.next()

            imgs = minibatch['data']
            gts = minibatch['label']
            unsup_imgs_0 = unsup_minibatch_0['data']
            unsup_imgs_1 = unsup_minibatch_1['data']
            mask_params = unsup_minibatch_0['mask_params']

            imgs = imgs.cuda(non_blocking=True)
            gts = gts.cuda(non_blocking=True)
            unsup_imgs_0 = unsup_imgs_0.cuda(non_blocking=True)
            unsup_imgs_1 = unsup_imgs_1.cuda(non_blocking=True)
            mask_params = mask_params.cuda(non_blocking=True)

            # Generate the mixed images
            batch_mix_masks = mask_params
            unsup_imgs_mixed = unsup_imgs_0 * (1 - batch_mix_masks) + unsup_imgs_1 * batch_mix_masks
            
            "Supervised Part"
            _, sup_pred_1 = model(imgs, step=1)
            _, sup_pred_2 = model(imgs, step=2)
            loss_sup_1 = criterion(sup_pred_1, gts)
            loss_sup_2 = criterion(sup_pred_2, gts)
            loss_sup = loss_sup_1 + loss_sup_2

            "Unsupervised Part"
            # Estimate the pseudo-label
            with torch.no_grad():
                # teacher#1
                _, logits_u0_tea_1 = model(unsup_imgs_0, step=1)
                _, logits_u1_tea_1 = model(unsup_imgs_1, step=1)
                logits_u0_tea_1 = logits_u0_tea_1.detach()
                logits_u1_tea_1 = logits_u1_tea_1.detach()
                # teacher#2
                _, logits_u0_tea_2 = model(unsup_imgs_0, step=2)
                _, logits_u1_tea_2 = model(unsup_imgs_1, step=2)
                logits_u0_tea_2 = logits_u0_tea_2.detach()
                logits_u1_tea_2 = logits_u1_tea_2.detach()
            # Mix teacher predictions using same mask
            logits_cons_tea_1 = logits_u0_tea_1 * (1 - batch_mix_masks) + logits_u1_tea_1 * batch_mix_masks
            _, ps_label_1 = torch.max(logits_cons_tea_1, dim=1)
            ps_label_1 = ps_label_1.long()
            logits_cons_tea_2 = logits_u0_tea_2 * (1 - batch_mix_masks) + logits_u1_tea_2 * batch_mix_masks
            _, ps_label_2 = torch.max(logits_cons_tea_2, dim=1)
            ps_label_2 = ps_label_2.long()

            loss_w_1 = torch.max(F.softmax(logits_cons_tea_1, dim=-1))
            loss_w_2 = torch.max(F.softmax(logits_cons_tea_2, dim=-1))

            confusion = confusion_matrix(np.concatenate([ps_label_1.view(-1).cpu().numpy(), np.array(label)]), np.concatenate([ps_label_2.view(-1).cpu().numpy(), np.array(label)]))
            w = (np.sum(confusion, axis=0)-np.diag(confusion))/np.sum(confusion, axis=0) + (np.sum(confusion, axis=1)-np.diag(confusion))/np.sum(confusion, axis=1)
            w = torch.from_numpy(w/np.sum(w)).cuda(non_blocking=True)
            overlap = np.sum(np.diag(confusion))/np.sum(confusion)

            ps_label_inter = torch.where(ps_label_1==ps_label_2, ps_label_1, torch.ones_like(ps_label_1)*255)
            ps_label_union = torch.where(w[ps_label_1]>w[ps_label_2], ps_label_1, ps_label_2)
            ps_label_union = torch.where(ps_label_1==ps_label_2, ps_label_1, ps_label_union)

            # Get student prediction for mixed image
            # student#1
            _, logits_cons_stu_1 = model(unsup_imgs_mixed, step=1)
            # student#2
            _, logits_cons_stu_2 = model(unsup_imgs_mixed, step=2)

            # Unsupervised Loss
            loss_unsup_inter = torch.mean(criterion(logits_cons_stu_1, ps_label_inter) * (loss_w_1 + loss_w_2)/2)
            loss_unsup_union =  torch.mean(criterion(logits_cons_stu_2, ps_label_union) * torch.where(ps_label_1==ps_label_2, (loss_w_1 + loss_w_2)/2, torch.where(w[ps_label_1]>w[ps_label_2], loss_w_1, loss_w_2)))
            loss_unsup = (loss_unsup_inter + loss_unsup_union) * config.unsup_weight


            "Total loss"
            loss = loss_sup + loss_unsup
            
            current_idx = epoch * config.niters_per_epoch + idx
            lr = lr_policy.get_lr(current_idx)
            optimizer_1.param_groups[0]['lr'] = lr
            optimizer_1.param_groups[1]['lr'] = lr
            for i in range(2, len(optimizer_1.param_groups)): optimizer_1.param_groups[i]['lr'] = lr
            optimizer_2.param_groups[0]['lr'] = lr
            optimizer_2.param_groups[1]['lr'] = lr
            for i in range(2, len(optimizer_2.param_groups)): optimizer_2.param_groups[i]['lr'] = lr

            loss.backward()
            optimizer_1.step()
            optimizer_2.step()

            print_str = 'Epoch{}/{}'.format(epoch, config.nepochs) \
                        + ' Iter{}/{}:'.format(idx + 1, config.niters_per_epoch) \
                        + ' loss=%.2f  ' % loss.item() \
                        + ' loss_unsup=%.2f' % loss_unsup.item() \
                        + ' (inter=%.2f' % loss_unsup_inter.item() \
                        + ' union=%.2f)  ' % loss_unsup_union.item() \
                        + ' loss_sup=%.2f' % loss_sup.item() \
                        + ' (sup_1=%.2f' % loss_sup_1.item() \
                        + ' sup_2=%.2f)  ' % loss_sup_2.item() \
                        + ' overlap=%.2f' % overlap
                        
            pbar.set_description(print_str, refresh=False)
            logfile.write(print_str+'\n')
            logfile.flush()

            end_time = time.time() 

        # Save the model
        if (epoch > config.nepochs // 6) and (epoch % config.snapshot_iter == 0) or (epoch == config.nepochs - 1):
            engine.save_and_link_checkpoint(config.snapshot_dir, config.log_dir, config.log_dir_link)

    logfile.close()