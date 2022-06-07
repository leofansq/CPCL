#!/usr/bin/env python3
# encoding: utf-8
import os
import cv2
import argparse
import numpy as np

import torch
import torch.nn as nn

from config import config
from utils.pyt_utils import parse_devices
from utils.visualize import print_iou, show_img
from engine.evaluator import Evaluator
from engine.logger import get_logger
from seg_opr.metric import hist_info, compute_score
from dataloader import CityScape
from network import Network
from dataloader import ValPre

logger = get_logger()

class SegEvaluator(Evaluator):
    def func_per_iteration(self, data, device):
        img = data['data']
        label = data['label']
        name = data['fn']
        pred = self.sliding_eval(img, config.eval_crop_size, config.eval_stride_rate, device)
        hist_tmp, labeled_tmp, correct_tmp = hist_info(config.num_classes, pred, label)
        results_dict = {'hist': hist_tmp, 'labeled': labeled_tmp, 'correct': correct_tmp}

        if self.save_image:
            fn = name + '.png'
            colors = self.dataset.get_class_colors()
            image = img
            clean = np.zeros(label.shape)
            comp_img = show_img(colors, config.background, image, clean, label, pred)
            cv2.imwrite(os.path.join(self.save_path, fn), comp_img)
            logger.info('Save the image ' + fn)

        return results_dict

    def compute_metric(self, results):
        hist = np.zeros((config.num_classes, config.num_classes))
        correct = 0
        labeled = 0
        count = 0
        for d in results:
            hist += d['hist']
            correct += d['correct']
            labeled += d['labeled']
            count += 1

        iu, mean_IU, _, mean_pixel_acc = compute_score(hist, correct, labeled)
        print ("----------------------------")
        print("class_nums: ", len(dataset.get_class_names()))
        result_line = print_iou(iu, mean_pixel_acc, dataset.get_class_names(), True)

        return result_line


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', default='last', type=str)
    parser.add_argument('-d', '--devices', default='0', type=str)
    parser.add_argument('-v', '--verbose', default=False, action='store_true')
    parser.add_argument('--save_image', '-s', default=False, action='store_true')
    parser.add_argument('--save_path', '-p', default=config.log_dir+'/pred_imgs/')

    args = parser.parse_args()
    all_dev = parse_devices(args.devices)

    network = Network(config.num_classes, norm_layer=nn.BatchNorm2d)
    data_setting = {'img_root': config.img_root_folder,
                    'gt_root': config.gt_root_folder,
                    'train_source': config.train_source,
                    'eval_source': config.eval_source}
    val_pre = ValPre()
    dataset = CityScape(data_setting, 'val', val_pre)

    with torch.no_grad():
        segmentor = SegEvaluator(dataset,
                                 config.num_classes,
                                 config.image_mean,
                                 config.image_std,
                                 network,
                                 config.eval_scale_array,
                                 config.eval_flip,
                                 all_dev, 
                                 args.verbose, 
                                 args.save_path,
                                 args.save_image)
        segmentor.run(config.snapshot_dir, args.epochs, config.val_log_file, config.link_val_log_file)
