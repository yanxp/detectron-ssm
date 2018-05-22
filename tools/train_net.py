#!/usr/bin/env python2

# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Train a network with Detectron."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import logging
import numpy as np
import pprint
import sys

from caffe2.python import workspace

from detectron.datasets.json_dataset import JsonDataset
from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.core.config import merge_cfg_from_list
from detectron.datasets.roidb import flipped_roidb_for_training,extend_with_flipped_entries,get_training_roidb
from detectron.core.test_engine import run_inference
from detectron.utils.logging import setup_logging
import detectron.utils.c2 as c2_utils
import detectron.utils.train
from bitmap import BitMap
from detectron.utils.helper import *

import pickle

c2_utils.import_contrib_ops()
c2_utils.import_detectron_ops()

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a network with Detectron'
    )
    parser.add_argument(
        '--cfg',
        dest='cfg_file',
        help='Config file for training (and optionally testing)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--multi-gpu-testing',
        dest='multi_gpu_testing',
        help='Use cfg.NUM_GPUS GPUs for inference',
        action='store_true'
    )
    parser.add_argument(
        '--skip-test',
        dest='skip_test',
        help='Do not test the final model',
        action='store_true'
    )
    parser.add_argument(
        'opts',
        help='See lib/core/config.py for all options',
        default=None,
        nargs=argparse.REMAINDER
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def main():
    # Initialize C2
    workspace.GlobalInit(
        ['caffe2', '--caffe2_log_level=0', '--caffe2_gpu_memory_tracking=1']
    )
    # Set up logging and load config options
    logger = setup_logging(__name__)
    logging.getLogger('detectron.roi_data.loader').setLevel(logging.INFO)
    args = parse_args()
    logger.info('Called with args:')
    logger.info(args)
    if args.cfg_file is not None:
        merge_cfg_from_file(args.cfg_file)
    if args.opts is not None:
        merge_cfg_from_list(args.opts)

    assert_and_infer_cfg()
    logger.info('Training with config:')
    logger.info(pprint.pformat(cfg))
    # Note that while we set the numpy random seed network training will not be
    # deterministic in general. There are sources of non-determinism that cannot
    # be removed with a reasonble execution-speed tradeoff (such as certain
    # non-deterministic cudnn functions).
    np.random.seed(cfg.RNG_SEED)
    # Execute the training run

    fs = open('imgnames.pkl','rb')
    roidbnames = pickle.load(fs)
    fs.close()

    logger.info('Loading dataset: {}'.format(cfg.TRAIN.DATASETS))
    
    dataset_names = cfg.TRAIN.DATASETS;proposal_files = cfg.TRAIN.PROPOSAL_FILES

    roidb = get_training_roidb(dataset_names,proposal_files)

    logger.info('{:d} roidb entries'.format(len(roidb)))
    
    total_num = len(roidb)
    
    # bitmap idx indicated for training
    bitmapRoidb = BitMap(total_num)

    # initial samples
#    initial_num = int(total_num*0.2)
#    for i in range(initial_num):
#        bitmapRoidb.set(i)
#
#    train_roidb = [roidb[i] for i in range(initial_num)]
   
    initialidx = []
    train_roidb =[]

    for i,x in enumerate(roidb):
        if x['image'].split('/')[-1] in roidbnames:
            initialidx.append(i)
            train_roidb.append(x)
    
    for i in initialidx:
        bitmapRoidb.set(i)

    logger.info('{:d} the number initial roidb entries'.format(len(train_roidb)))
    # append flipped images
    train_roidb = flipped_roidb_for_training(train_roidb)
    
    logger.info('{:d} the number initial roidb entries'.format(len(train_roidb)))
    alamount = 0;ssamount = 0;gamma = 0.95
    # control al proportion
    al_proportion_checkpoint = [int(x*total_num*0.4) for x in np.linspace(0.2,1,10)]
    # control ss proportion
    ss_proportion_checkpoint = [int(x*total_num) for x in np.linspace(0.2,2,10)]

    next_iters = 90000 ; sum_iters = next_iters;
 
    '''load the lasted checkpoints'''
    checkpoints = detectron.utils.train.train_model(sum_iters,train_roidb,cfg.TRAIN.WEIGHTS)
    while True:
        # to do a test on the test dataset
        test_model(checkpoints[(sum_iters-1)], args.multi_gpu_testing, args.opts)
        if sum_iters > cfg.SOLVER.MAX_ITER:
            break
        # next detect unlabeled samples
        unlabeledidx = list(set(range(total_num))-set(bitmapRoidb.nonzero()))
        # labeled samples
        labeledidx = list(set(bitmapRoidb.nonzero()))
        # detect unlabeled samples
        BBoxes,YClass,Scores,al_candidate_idx,ALScore= detect_im(checkpoints[(sum_iters-1)],roidb,gamma,idxs=unlabeledidx,gpu_id=0)
       
        al_avg_idx = np.argsort(np.array(ALScore)) 
        al_candidate_idx = [al_candidate_idx[i] for i in al_avg_idx]

        gamma = max(gamma-0.05,0.7)
        
        # the ss candidate idx  
        ss_candidate_idx = [i for i in unlabeledidx if i not in al_candidate_idx]
    
        # update roidb for next training
        train_roidb = replace_roidb(roidb,BBoxes,YClass,ss_candidate_idx) 

        # control the proportion
        if alamount+len(al_candidate_idx)>=al_proportion_checkpoint[0]:
            al_candidate_idx = al_candidate_idx[:int(al_proportion_checkpoint[0]-alamount)]
            tmp = al_proportion_checkpoint.pop(0)
            al_proportion_checkpoint.append(al_proportion_checkpoint[-1])
        if ssamount+len(ss_candidate_idx)>=ss_proportion_checkpoint[0]:
            ss_candidate_idx = ss_candidate_idx[:int(ss_proportion_checkpoint[0]-ssamount)]
            tmp = ss_proportion_checkpoint.pop(0)
            ss_proportion_checkpoint.append(ss_proportion_checkpoint[-1])

        # record ss and al factor

        alamount += len(al_candidate_idx)
        ssamount += len(ss_candidate_idx)

        logger.info('alfactor:{},ssfactor:{}'.format(alamount/total_num,ssamount/total_num))
         
#       for idx in al_candidate_idx:
#            bitmapRoidb.set(idx)
        next_train_idx = bitmapRoidb.nonzero();
        next_train_idx.extend(ss_candidate_idx)
        
        train_roidb = blur_image(train_roidb,ss_candidate_idx)
        # the next training roidb
        train_roidb = [train_roidb[i] for i in next_train_idx]
        # flipped the roidb
        train_roidb = flipped_roidb_for_training(train_roidb)
        # the next training iters
        next_iters = 30000
        sum_iters += next_iters
        checkpoints = detectron.utils.train.train_model(sum_iters,train_roidb,checkpoints[(sum_iters-next_iters-1)])


        
    # # Test the trained model
    # if not args.skip_test:
    #     test_model(checkpoints['final'], args.multi_gpu_testing, args.opts)


def test_model(model_file, multi_gpu_testing, opts=None):
    """Test a model."""
    # Clear memory before inference
    workspace.ResetWorkspace()
    # Run inference
    run_inference(
        model_file, multi_gpu_testing=multi_gpu_testing,
        check_expected_results=True,
    )


if __name__ == '__main__':
    main()
