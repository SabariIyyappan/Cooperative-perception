# -*- coding: utf-8 -*-
"""
Helpers for the RQ1 evaluation harness. Library code is left untouched; these
functions mirror library behaviour but keep GT object identity.
"""

import random

import numpy as np
import torch

from opencood.utils import box_utils, common_utils


def seed_dataset_per_index(dataset, base_seed=303):
    """
    Make every __getitem__(idx) draw the same random numbers (pose noise,
    point shuffling) regardless of DataLoader workers, run, or model mode.
    """
    cls = type(dataset)

    class _Seeded(cls):
        def __getitem__(self, idx):
            np.random.seed(base_seed + int(idx))
            random.seed(base_seed + int(idx))
            return super().__getitem__(idx)

    _Seeded.__name__ = 'Seeded' + cls.__name__
    dataset.__class__ = _Seeded
    return dataset


def frame_key(dataset, idx):
    """dataset index -> (scenario folder name, timestamp), as in retrieve_base_data."""
    scenario_index = 0
    for i, ele in enumerate(dataset.len_record):
        if idx < ele:
            scenario_index = i
            break
    timestamp_index = idx if scenario_index == 0 else \
        idx - dataset.len_record[scenario_index - 1]
    scenario_database = dataset.scenario_database[scenario_index]
    timestamp_key = dataset.return_timestamp_key(scenario_database, timestamp_index)
    scenario_name = dataset.scenario_folders[scenario_index].rstrip('/').split('/')[-1]
    return scenario_name, timestamp_key


def gt_with_ids(batch_data, post_processor):
    """
    Same boxes as BasePostprocessor.generate_gt_bbx, in the same order, but
    also returns the aligned object ids (the library loses them at the final
    range mask).
    """
    gt_list, id_list = [], []
    for _, cav_content in batch_data.items():
        tfm = cav_content['transformation_matrix_clean']
        centers = cav_content['object_bbx_center'][cav_content['object_bbx_mask'] == 1]
        corners = box_utils.boxes_to_corners_3d(centers, post_processor.params['order'])
        gt_list.append(box_utils.project_box3d(corners.float(), tfm))
        id_list += list(cav_content['object_ids'])

    gt = torch.vstack(gt_list)
    selected = [id_list.index(x) for x in set(id_list)]
    gt = gt[selected]
    ids = [id_list[i] for i in selected]

    gt_np, mask = box_utils.mask_boxes_outside_range_numpy(
        gt.cpu().numpy(), post_processor.params['gt_range'], order=None,
        return_mask=True)
    ids = [i for i, m in zip(ids, mask) if m]
    return torch.from_numpy(gt_np).to(device=gt.device), ids


def match_with_ids(det_boxes, det_score, gt_boxes, gt_ids, result_stat, iou_thresh):
    """
    Exactly the greedy rule of eval_utils.caluclate_tp_fp (and it fills
    result_stat identically), but also returns per-prediction matches.

    Returns
    -------
    dict with
      'score'   : (N,) prediction scores, descending
      'tp'      : (N,) 1/0
      'gt_id'   : (N,) matched GT object id, -1 for FP
    """
    fp, tp, matched = [], [], []
    scores = np.zeros(0)
    if det_boxes is not None:
        det_boxes = common_utils.torch_tensor_to_numpy(det_boxes)
        det_score = common_utils.torch_tensor_to_numpy(det_score)
        gt_np = common_utils.torch_tensor_to_numpy(gt_boxes)

        order = np.argsort(-det_score)
        scores = det_score[order]
        det_polygons = list(common_utils.convert_format(det_boxes))
        gt_polygons = list(common_utils.convert_format(gt_np))
        remaining_ids = list(gt_ids)

        for i in range(order.shape[0]):
            ious = common_utils.compute_iou(det_polygons[order[i]], gt_polygons)
            if len(gt_polygons) == 0 or np.max(ious) < iou_thresh:
                fp.append(1)
                tp.append(0)
                matched.append(-1)
                continue
            fp.append(0)
            tp.append(1)
            k = int(np.argmax(ious))
            gt_polygons.pop(k)
            matched.append(remaining_ids.pop(k))
        result_stat[iou_thresh]['score'] += scores.tolist()
    result_stat[iou_thresh]['fp'] += fp
    result_stat[iou_thresh]['tp'] += tp
    result_stat[iou_thresh]['gt'] += gt_boxes.shape[0]
    return {'score': scores.astype(np.float32),
            'tp': np.asarray(tp, dtype=np.int8),
            'gt_id': np.asarray(matched, dtype=np.int64)}


def keep_ego_only(batch_data):
    """
    Reduce an intermediate-fusion test batch (batch size 1) to the ego's own
    features. GT (object_bbx_center / object_ids) is untouched, so it stays the
    union over all agents.
    """
    ego = batch_data['ego']
    pl = ego['processed_lidar']
    keep = pl['voxel_coords'][:, 0] == 0
    ego['processed_lidar'] = {'voxel_features': pl['voxel_features'][keep],
                              'voxel_coords': pl['voxel_coords'][keep],
                              'voxel_num_points': pl['voxel_num_points'][keep]}
    ego['record_len'] = torch.ones_like(ego['record_len'])
    return batch_data
