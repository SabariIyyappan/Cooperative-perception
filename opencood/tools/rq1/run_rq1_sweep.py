# -*- coding: utf-8 -*-
"""
RQ1 pose-noise sweep with per-object matching dumps.

Modes (all with the same CoAlign weights):
  ego_only     intermediate batch reduced to the ego's features (No Fusion);
               GT stays the union over all agents
  fusion_noBA  attention fusion, box alignment removed from hypes
  fusion_BA    CoAlign as configured (box alignment on)

Noise: rot_std (deg) = pos_std (m), means 0, Gaussian. Pose noise is seeded
per sample index so every mode sees identical noisy poses.

Output: <out_dir>/<mode>/noise_<std>/{frames.pkl, eval_<...>.yaml}

Usage:
  python opencood/tools/rq1/run_rq1_sweep.py \
      --model_dir opencood/logs/opv2v_coalign_woba_withnoise_2023_02_10_13_46_05 \
      --mode fusion_BA --noise 0 0.1 0.2 0.4 0.8 --out_dir ../results/rq1
"""

import argparse
import copy
import os
import pickle
import time
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import inference_utils, train_utils
from opencood.tools.rq1 import rq1_utils
from opencood.utils import eval_utils

torch.multiprocessing.set_sharing_strategy('file_system')

IOUS = (0.3, 0.5, 0.7)
MODES = ('ego_only', 'fusion_noBA', 'fusion_BA')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_dir', required=True)
    p.add_argument('--mode', required=True, choices=MODES)
    p.add_argument('--noise', type=float, nargs='+', default=[0, 0.1, 0.2, 0.4, 0.8])
    p.add_argument('--out_dir', required=True)
    p.add_argument('--seed', type=int, default=303)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--max_frames', type=int, default=None, help='smoke tests')
    p.add_argument('--max_cav', type=int, default=None,
                   help='override train_params.max_cav (verification only)')
    p.add_argument('--tag', default='', help='suffix for the mode folder')
    return p.parse_args()


def build_hypes(opt, std):
    hypes = yaml_utils.load_yaml(None, opt)
    hypes['validate_dir'] = hypes['test_dir']
    assert 'test' in hypes['validate_dir']
    if opt.mode == 'fusion_BA':
        assert 'box_align' in hypes, 'config has no box_align block'
        hypes['box_align']['val_result'] = hypes['box_align']['test_result']
    else:
        hypes.pop('box_align', None)
    if opt.max_cav is not None:
        hypes['train_params']['max_cav'] = opt.max_cav
    noise_setting = OrderedDict()
    noise_setting['add_noise'] = True
    noise_setting['args'] = {'pos_std': std, 'rot_std': std,
                             'pos_mean': 0, 'rot_mean': 0}
    hypes['noise_setting'] = noise_setting
    return hypes


def main():
    opt = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    hypes0 = build_hypes(opt, 0.0)
    model = train_utils.create_model(hypes0)
    _, model = train_utils.load_saved_model(opt.model_dir, model)
    model.to(device).eval()

    for std in opt.noise:
        hypes = build_hypes(opt, std)
        dataset = build_dataset(hypes, visualize=False, train=False)
        rq1_utils.seed_dataset_per_index(dataset, opt.seed)
        indices = list(range(len(dataset)))
        if opt.max_frames:
            indices = indices[:opt.max_frames]
        loader = DataLoader(Subset(dataset, indices), batch_size=1,
                            num_workers=opt.num_workers,
                            collate_fn=dataset.collate_batch_test,
                            shuffle=False, pin_memory=False, drop_last=False)

        save_dir = os.path.join(opt.out_dir, opt.mode + opt.tag, f'noise_{std:g}')
        os.makedirs(save_dir, exist_ok=True)
        result_stat = {iou: {'tp': [], 'fp': [], 'gt': 0, 'score': []} for iou in IOUS}
        frames = []
        t0 = time.time()

        for i, batch_data in enumerate(loader):
            if batch_data is None:
                continue
            with torch.no_grad():
                batch_data = train_utils.to_device(batch_data, device)
                if opt.mode == 'ego_only':
                    rq1_utils.keep_ego_only(batch_data)
                infer = inference_utils.inference_intermediate_fusion(
                    batch_data, model, dataset)

            pred, score, gt = (infer['pred_box_tensor'], infer['pred_score'],
                               infer['gt_box_tensor'])
            gt_ids_boxes, gt_ids = rq1_utils.gt_with_ids(batch_data, dataset.post_processor)
            assert gt_ids_boxes.shape == gt.shape and \
                torch.allclose(gt_ids_boxes.float(), gt.float(), atol=1e-4), \
                f'GT/id misalignment at frame {i}'

            matches = {iou: rq1_utils.match_with_ids(pred, score, gt, gt_ids,
                                                     result_stat, iou)
                       for iou in IOUS}

            ego = batch_data['ego']
            idx = int(ego['sample_idx'])
            frames.append({
                'idx': idx,
                'key': rq1_utils.frame_key(dataset, idx),
                'cav_id_list': list(ego['cav_id_list']),
                'gt_ids': np.asarray(gt_ids, dtype=np.int64),
                'gt_corners': gt.cpu().numpy().astype(np.float32),
                'pred_corners': (pred.cpu().numpy().astype(np.float32)
                                 if pred is not None else np.zeros((0, 8, 3), np.float32)),
                'pred_score': (score.cpu().numpy().astype(np.float32)
                               if score is not None else np.zeros(0, np.float32)),
                'matches': matches,
                'lidar_pose': ego['lidar_pose'].cpu().numpy(),
                'lidar_pose_clean': ego['lidar_pose_clean'].cpu().numpy(),
            })
            if i % 100 == 0:
                print(f'[{opt.mode} std={std:g}] {i}/{len(indices)} '
                      f'{time.time() - t0:.0f}s', flush=True)
            torch.cuda.empty_cache()

        infer_info = f'{opt.mode}{opt.tag}_noise_{std:g}'
        ap30, ap50, ap70 = eval_utils.eval_final_results(result_stat, save_dir, infer_info)
        with open(os.path.join(save_dir, 'frames.pkl'), 'wb') as f:
            pickle.dump({'mode': opt.mode, 'std': std, 'seed': opt.seed,
                         'model_dir': os.path.abspath(opt.model_dir),
                         'ap': {0.3: ap30, 0.5: ap50, 0.7: ap70},
                         'frames': frames}, f)
        print(f'[{opt.mode} std={std:g}] done in {time.time() - t0:.0f}s '
              f'AP30/50/70 = {ap30:.4f}/{ap50:.4f}/{ap70:.4f}', flush=True)


if __name__ == '__main__':
    main()
