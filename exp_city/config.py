# encoding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import sys
import time
import numpy as np
from easydict import EasyDict as edict

C = edict()
config = C
cfg = C

############################################
#                Path Config               #
############################################
remoteip = os.popen('pwd').read()
if os.getenv('volna') is not None:
    C.volna = os.environ['volna']
else:
    C.volna = '/Path_to_CPCL/'
C.repo_name = 'CPCL'
C.abs_dir = osp.realpath(".")
C.root_dir = C.abs_dir[:C.abs_dir.index(C.repo_name) + len(C.repo_name)]
C.log_dir = osp.abspath('log')

C.log_dir_link = osp.join(C.abs_dir, 'log')

# the path for Checkpoints
if os.getenv('snapshot_dir'):
    C.snapshot_dir = osp.join(os.environ['snapshot_dir'], "snapshot")
else:
    C.snapshot_dir = osp.abspath(osp.join(C.log_dir, "snapshot"))

# the path for Log
exp_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
C.train_log_file = C.log_dir + '/train_' + exp_time + '.log'
C.val_log_file = C.log_dir + '/val_' + exp_time + '.log'
C.link_val_log_file = C.log_dir + '/val_last.log'

# the path for Data & Weight
C.dataset_path = C.volna + "DATA/city"
C.img_root_folder = C.dataset_path
C.gt_root_folder = C.dataset_path
C.pretrained_model = C.volna +  'DATA/pytorch-weight/resnet50_v1c.pth'

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
add_path(osp.join(C.root_dir, 'furnace'))


############################################
#            Experiments Config            #
############################################
C.device = '0'
C.seed = 12345

C.labeled_ratio = 8     # ratio of labeled set
C.nepochs = 137 # nepochs

# Dataset Config
C.train_source = osp.join(C.dataset_path, "config_new/subset_train/train_aug_labeled_1-{}.txt".format(C.labeled_ratio))
C.unsup_source = osp.join(C.dataset_path, "config_new/subset_train/train_aug_unlabeled_1-{}.txt".format(C.labeled_ratio))
C.eval_source = osp.join(C.dataset_path, "config_new/val.txt")
C.test_source = osp.join(C.dataset_path, "config_new/test.txt")
C.demo_source = osp.join(C.dataset_path, "config_new/demo.txt")

# Cutmix Config
C.cutmix_mask_prop_range = (0.25, 0.5)
C.cutmix_boxmask_n_boxes = 3
C.cutmix_boxmask_fixed_aspect_ratio = False
C.cutmix_boxmask_by_size = False
C.cutmix_boxmask_outside_bounds = False
C.cutmix_boxmask_no_invert = False

# Image Config
C.num_classes = 19
C.background = 255
C.image_mean = np.array([0.485, 0.456, 0.406])
C.image_std = np.array([0.229, 0.224, 0.225])
C.image_height = 800
C.image_width = 800
C.num_train_imgs = 2975 // C.labeled_ratio
C.num_eval_imgs = 500
C.num_unsup_imgs = 2975 - C.num_train_imgs

# Training Config
if os.getenv('learning_rate'):
    C.lr = float(os.environ['learning_rate'])
else:
    C.lr = 0.01

if os.getenv('batch_size'):
    C.batch_size = int(os.environ['batch_size'])
else:
    C.batch_size = 16 # batch_size

C.lr_power = 0.9
C.momentum = 0.9
C.weight_decay = 1e-4

C.bn_eps = 1e-5
C.bn_momentum = 0.1
C.unsup_weight = 5

C.max_samples = max(C.num_train_imgs, C.num_unsup_imgs)
C.niters_per_epoch = C.max_samples // C.batch_size
C.num_workers = 4
C.train_scale_array = [0.5, 0.75, 1, 1.5, 1.75, 2.0]
C.warm_up_epoch = 0

# Evaluation Config
C.eval_iter = 30
C.eval_stride_rate = 2 / 3
C.eval_scale_array = [1]  # 0.5, 0.75, 1, 1.5, 1.75
C.eval_flip = False
C.eval_crop_size = 800

# Display Config
if os.getenv('snapshot_iter'):
    C.snapshot_iter = int(os.environ['snapshot_iter'])
else:
    C.snapshot_iter = 2
C.record_info_iter = 20
C.display_iter = 50


