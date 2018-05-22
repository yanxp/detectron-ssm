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

"""Test a Detectron network on an imdb (image database)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import cv2
import datetime
import logging
import numpy as np
import os
import yaml
import scipy
import numpy as np

from caffe2.python import workspace

from detectron.core.config import cfg
from detectron.core.config import get_output_dir
from detectron.core.rpn_generator import generate_rpn_on_dataset
from detectron.core.rpn_generator import generate_rpn_on_range
from detectron.core.test import im_detect_all,im_detect_bbox_aug,box_results_with_nms_and_limit
from detectron.datasets import task_evaluation
from detectron.datasets.json_dataset import JsonDataset
from detectron.modeling import model_builder
from detectron.utils.io import save_object
from detectron.utils.timer import Timer
import detectron.utils.c2 as c2_utils
import detectron.utils.env as envu
import detectron.utils.net as net_utils
import detectron.utils.subprocess as subprocess_utils
import detectron.utils.vis as vis_utils
import detectron.core.test_engine as infer_engine
import detectron.datasets.dummy_datasets as dummy_datasets

logger = logging.getLogger(__name__)

def detect_im(weights_file,roidb, gamma, idxs=None,gpu_id = 0):
    '''detect the unlabeled samples'''
    roidb = [roidb[i] for i in idxs]

    model = infer_engine.initialize_model_from_cfg(weights_file, gpu_id=gpu_id)
    thresh = gamma

    allBoxes=[];allScore=[];allY=[];eps=0;al_idx=[];allClass=[]
    ALScore=[]
    timers = defaultdict(Timer)
    dummy_coco_dataset = dummy_datasets.get_coco_dataset()

    for i, entry in enumerate(roidb):
        
        box_proposals = None

        im = cv2.imread(entry['image'])
        with c2_utils.NamedCudaScope(gpu_id):
           cls_boxes_i, cls_segms_i, cls_keyps_i = im_detect_all(
                       model, im, box_proposals, timers )
       
#            scores, boxes, im_scale = im_detect_bbox_aug(model, im, box_proposals)
##            print('scores:{},boxes:{}'.format(scores.shape,boxes.shape))
#
#            scores_i, boxes_i, cls_boxes_i = box_results_with_nms_and_limit(scores, boxes)
#            cls_segms_i = None;cls_keyps_i = None
        
#        output_dir = './'+str(gamma)
#        if True:
#            im_name = os.path.splitext(os.path.basename(entry['image']))[0]
#            vis_utils.vis_one_image(
#                im[:, :, ::-1],
#                '{:d}_{:s}'.format(i, im_name),
#                os.path.join(output_dir, 'vis'),
#                cls_boxes_i,
#                segms=None,
#                keypoints=None,
#                thresh=0.9,
#                box_alpha=0.8,
#                dataset=dummy_coco_dataset,
#                show_class=True
#            )
        
        if isinstance(cls_boxes_i, list):
            boxes, segms, keypoints, classes = convert_from_cls_format(
                cls_boxes_i, cls_segms_i, cls_keyps_i)
        if boxes is None or boxes.shape[0] == 0 or max(boxes[:, 4]) < thresh:
            # al process
            al_idx.append(idxs[i])
            if boxes is not None and boxes.shape[0] != 0:

               ALScore.append(np.mean(boxes[:, 4]))
            else:
               ALScore.append(0)
            continue

#        print('scores_i:{},boxes_i:{},boxes:{},cls_boxes_i:{}'.format(scores_i, boxes_i,boxes, cls_boxes_i))

        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        sorted_inds = np.argsort(-areas)

        BBox = []
        Score = []
        Y = []
        Class = []

        for i in sorted_inds:
            bbox = boxes[i, :4]
            score = boxes[i, -1]
            # add self-supervised process
            if score < thresh:
                continue
            BBox.append(list(bbox))
            Score.append(score) # only one class score ??
            Class.append(classes[i])
    
        allBoxes.append(BBox);allClass.append(Class);allScore.append(Score)
    return allBoxes,allClass,allScore,al_idx,ALScore

def replace_roidb(roidb,BBoxes,YClass,unlabeledidx):
    ''' with fake replace gt '''
    for i,idx in enumerate(unlabeledidx):
        curr_len = len(YClass[i])
        boxes = np.array(BBoxes[i] ,dtype=np.float32)
        gt_classes = np.array(YClass[i],dtype=np.int32)
        gt_overlaps = np.zeros((curr_len, cfg.MODEL.NUM_CLASSES), dtype=np.float32)
        for j in range(curr_len):
            gt_overlaps[j, YClass[j]] = 1.0
        gt_overlaps = scipy.sparse.csr_matrix(gt_overlaps)
        max_classes = np.array(YClass[i],dtype=np.int32)
        max_overlaps = np.ones(curr_len)  
        box_to_gt_ind_map = np.array(range(curr_len),dtype=np.int32)
        is_crowd = np.array([False]*curr_len)
        roidb[idx]['boxes'] = boxes
        roidb[idx]['gt_classes'] = gt_classes
        roidb[idx]['gt_overlaps'] = gt_overlaps
        roidb[idx]['max_classes'] = max_classes
        roidb[idx]['max_overlaps'] = max_overlaps
        roidb[idx]['box_to_gt_ind_map'] = box_to_gt_ind_map
        roidb[idx]['is_crowd'] = is_crowd
    print('-----replace gt with fake gt----')

    return roidb
    
def blur_image(roidbs,ss_candidate_idx):
    '''blur images except BBox regions'''
    def _handle(roi, idx):
        imgpath = roi['image'].split('/')[-1]
        im = cv2.imread(roi['image'])
        im_bbox = []
        for box in roi['boxes']:
            box = list(map(int, box))
            im_bbox.append(im[box[1]:box[3], box[0]:box[2]])
        new_im = cv2.blur(im, (25,25))
        for i, box in enumerate(roi['boxes']):
            box = list(map(int, box))
            cv2.rectangle(new_im,(box[0],box[1]),(box[2],box[3]),(255,0,0),3)
            new_im[box[1]:box[3], box[0]:box[2]] = im_bbox[i]
        path = 'tmpdata/{}'.format(imgpath)
        cv2.imwrite(path, new_im)
        assert os.path.exists(path), "didnt save successfully"
        roi['image'] = path
        return roi
    copy_roidb = []
    for i in range(len(roidbs)):
        if len(roidbs[i]['boxes'])>0 and i in ss_candidate_idx and not roidbs[i]['flipped']:
            copy_roidb.append(roidbs[i].copy())
            copy_roidb[i] = _handle(copy_roidb[i], i)
        else:
            copy_roidb.append(roidbs[i].copy())
    return copy_roidb

def get_roidb_and_dataset(dataset_name, idxs):
    """Get the roidb for the dataset specified in the global cfg. Optionally
    restrict it to a range of indices if ind_range is a pair of integers.
    """
    dataset = JsonDataset(dataset_name)
    
    roidb = dataset.get_roidb()

    if idxs is not None:
        total_num_images = len(roidb)
        start = 0
        end = len(idxs)
        roidb = [roidb[i] for i in idxs]
    else:
        start = 0
        end = len(roidb)
        total_num_images = end

    return roidb, dataset, start, end, total_num_images

def convert_from_cls_format(cls_boxes, cls_segms, cls_keyps):
    """Convert from the class boxes/segms/keyps format generated by the testing
    code.
    """
    box_list = [b for b in cls_boxes if len(b) > 0]
    if len(box_list) > 0:
        boxes = np.concatenate(box_list)
    else:
        boxes = None
    if cls_segms is not None:
        segms = [s for slist in cls_segms for s in slist]
    else:
        segms = None
    if cls_keyps is not None:
        keyps = [k for klist in cls_keyps for k in klist]
    else:
        keyps = None
    classes = []
    for j in range(len(cls_boxes)):
        classes += [j] * len(cls_boxes[j])
    return boxes, segms, keyps, classes
