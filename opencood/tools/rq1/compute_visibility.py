# -*- coding: utf-8 -*-
"""
RQ1: per-object, per-agent visibility labels for OPV2V.

For every frame and every GT object (union of all cavs' `vehicles`), and for
every cav k in the scenario:
  n_pts          number of k's own LiDAR returns inside the object's box
                 (CARLA's ray casting against the full world -> primary label)
  ray_free_frac  fraction of geometric rays from k's sensor to 10 sample points
                 on the box that are not blocked by any other vehicle box
                 (cross-check; only vehicle occluders are known)
  occluders      ids of vehicles that blocked at least one ray

Everything is computed with clean poses, independent of the model and noise.

Output (pickle):
  {'meta': {...},
   'frames': {(scenario, timestamp): {
        'cav_ids': [str, ...],                      # sorted, ego first
        'objects': {obj_id(int): {
             'is_cav': bool,
             'vis': {cav_id(str): (n_pts, ray_free_frac, tuple(occluders))}}}}}}

Usage:
  python opencood/tools/rq1/compute_visibility.py \
      --data_dir dataset/OPV2V/test --out ../results/rq1/visibility_opv2v_test.pkl
"""

import argparse
import os
import pickle
import time
from collections import OrderedDict
from multiprocessing import Pool

import numpy as np

from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.utils.pcd_utils import pcd_to_np
from opencood.utils.transformation_utils import x_to_world

# Approximate half-extent of the CAV body (CARLA Lincoln MKZ), used only as an
# occluder when a CAV is not listed in any peer's `vehicles`.
CAV_HALF_EXTENT = np.array([2.45, 1.06, 0.75])
CAV_LIDAR_HEIGHT = 1.9  # lidar height above ground (lidar_pose z - true_ego_pos z);
                        # only used to place the CAV body box below the sensor


def object_to_world(obj):
    """4x4 transform object-box-frame -> world, box centred at origin."""
    loc, ctr, ang = obj['location'], obj.get('center', [0, 0, 0]), obj['angle']
    pose = [loc[0] + ctr[0], loc[1] + ctr[1], loc[2] + ctr[2],
            ang[0], ang[1], ang[2]]
    return x_to_world(pose)


def points_in_box_count(pts_obj, half_ext, ground_margin):
    """pts_obj: (N,3) points already in the box frame."""
    inside = (np.abs(pts_obj[:, 0]) <= half_ext[0]) & \
             (np.abs(pts_obj[:, 1]) <= half_ext[1]) & \
             (pts_obj[:, 2] <= half_ext[2]) & \
             (pts_obj[:, 2] >= -half_ext[2] + ground_margin)
    return int(inside.sum())


def segment_hits_box(p0, p1, half_ext):
    """Slab test: does segment p0->p1 (in the box frame) intersect the AABB?"""
    d = p1 - p0
    t0, t1 = 0.0, 1.0
    for a in range(3):
        if abs(d[a]) < 1e-9:
            if abs(p0[a]) > half_ext[a]:
                return False
            continue
        ta = (-half_ext[a] - p0[a]) / d[a]
        tb = (half_ext[a] - p0[a]) / d[a]
        if ta > tb:
            ta, tb = tb, ta
        t0, t1 = max(t0, ta), min(t1, tb)
        if t0 > t1:
            return False
    return True


def ray_targets(half_ext, shrink=0.8):
    """Box centre + 4 BEV corners (shrunk), at mid-height and near the roof,
    in the box frame (10 targets). The roof targets matter because the
    sensor sits above most vehicles."""
    ex, ey, ez = half_ext[0] * shrink, half_ext[1] * shrink, half_ext[2] * shrink
    bev = [[0, 0], [ex, ey], [ex, -ey], [-ex, ey], [-ex, -ey]]
    return np.array([[x, y, z] for z in (0.0, ez) for x, y in bev],
                    dtype=np.float64)


def process_frame(cav_files, ground_margin, max_range):
    """
    cav_files: OrderedDict cav_id -> (yaml_path, pcd_path), ego first.
    Returns the per-frame record described in the module docstring.
    """
    params = OrderedDict((cid, load_yaml(y)) for cid, (y, _) in cav_files.items())

    # union of objects (world frame). Same id -> same actor, so first wins.
    objects = OrderedDict()
    for p in params.values():
        for oid, obj in p['vehicles'].items():
            if oid not in objects:
                objects[oid] = obj
    obj_ids = list(objects.keys())
    obj_T = {oid: object_to_world(objects[oid]) for oid in obj_ids}
    obj_ext = {oid: np.asarray(objects[oid]['extent'], dtype=np.float64)
               for oid in obj_ids}

    # occluder set = all objects + CAV bodies not already listed
    occ_T, occ_ext = dict(obj_T), dict(obj_ext)
    for cid, p in params.items():
        if int(cid) not in occ_T:
            T = x_to_world(p['lidar_pose'])
            # drop the box centre below the sensor
            T = T @ np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                              [0, 0, 1, -(CAV_LIDAR_HEIGHT - CAV_HALF_EXTENT[2])],
                              [0, 0, 0, 1]], dtype=np.float64)
            occ_T[int(cid)] = T
            occ_ext[int(cid)] = CAV_HALF_EXTENT
    occ_T_inv = {oid: np.linalg.inv(T) for oid, T in occ_T.items()}

    record = {'cav_ids': list(cav_files.keys()), 'objects': {}}
    cav_ids_int = set(int(c) for c in cav_files.keys())
    for oid in obj_ids:
        record['objects'][oid] = {'is_cav': oid in cav_ids_int, 'vis': {}}

    for cid, (_, pcd_path) in cav_files.items():
        lidar_T = x_to_world(params[cid]['lidar_pose'])           # lidar -> world
        sensor_world = lidar_T[:3, 3]
        pts = pcd_to_np(pcd_path)[:, :3].astype(np.float64)        # lidar frame
        pts_h = np.c_[pts, np.ones(len(pts))]
        pts_world = (lidar_T @ pts_h.T).T[:, :3]

        for oid in obj_ids:
            if oid == int(cid):
                continue  # an agent does not observe its own body
            T_inv = np.linalg.inv(obj_T[oid])
            centre = obj_T[oid][:3, 3]
            if np.linalg.norm(centre[:2] - sensor_world[:2]) > max_range:
                record['objects'][oid]['vis'][cid] = (0, 0.0, ())
                continue

            # points in box (prefilter by radius for speed)
            r = np.linalg.norm(obj_ext[oid]) + 0.5
            near = np.linalg.norm(pts_world[:, :2] - centre[:2], axis=1) < r
            if near.any():
                p_obj = (T_inv @ np.c_[pts_world[near], np.ones(near.sum())].T).T[:, :3]
                n_pts = points_in_box_count(p_obj, obj_ext[oid], ground_margin)
            else:
                n_pts = 0

            # geometric ray cast against all other vehicle boxes
            targets_obj = ray_targets(obj_ext[oid])
            targets_world = (obj_T[oid] @ np.c_[targets_obj,
                                                np.ones(len(targets_obj))].T).T[:, :3]
            seg_len = np.linalg.norm(targets_world[0, :2] - sensor_world[:2])
            free, occluders = 0, set()
            for tw in targets_world:
                blocked = False
                for qid, Tq_inv in occ_T_inv.items():
                    if qid == oid or qid == int(cid):
                        continue
                    # cheap BEV distance-to-segment rejection
                    qc = occ_T[qid][:3, 3]
                    if np.linalg.norm(qc[:2] - sensor_world[:2]) > seg_len + 6:
                        continue
                    p0 = (Tq_inv @ np.r_[sensor_world, 1.0])[:3]
                    p1 = (Tq_inv @ np.r_[tw, 1.0])[:3]
                    if segment_hits_box(p0, p1, occ_ext[qid]):
                        blocked = True
                        occluders.add(qid)
                if not blocked:
                    free += 1
            record['objects'][oid]['vis'][cid] = \
                (n_pts, free / len(targets_world), tuple(sorted(occluders)))
    return record


def scenario_frames(scenario_dir, max_cav):
    """Yield (timestamp, OrderedDict cav_id -> (yaml, pcd)) like OPV2VBaseDataset."""
    cav_list = sorted([x for x in os.listdir(scenario_dir)
                       if os.path.isdir(os.path.join(scenario_dir, x))])
    if int(cav_list[0]) < 0:
        cav_list = cav_list[1:] + [cav_list[0]]
    cav_list = cav_list[:max_cav]
    ego_dir = os.path.join(scenario_dir, cav_list[0])
    timestamps = sorted(f[:-5] for f in os.listdir(ego_dir)
                        if f.endswith('.yaml') and 'additional' not in f)
    for ts in timestamps:
        files = OrderedDict()
        for cid in cav_list:
            d = os.path.join(scenario_dir, cid)
            files[cid] = (os.path.join(d, ts + '.yaml'), os.path.join(d, ts + '.pcd'))
        yield ts, files


def process_scenario(job):
    scenario_dir, max_cav, ground_margin, max_range = job
    name = os.path.basename(scenario_dir)
    out = {}
    t = time.time()
    for ts, files in scenario_frames(scenario_dir, max_cav):
        out[(name, ts)] = process_frame(files, ground_margin, max_range)
    print(f'{name}: {len(out)} frames in {time.time() - t:.0f}s', flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--max_cav', type=int, default=5)
    ap.add_argument('--ground_margin', type=float, default=0.2,
                    help='ignore points within this height of the box bottom')
    ap.add_argument('--max_range', type=float, default=150.0,
                    help='objects beyond this BEV distance get n_pts=0 directly')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--scenarios', nargs='*', default=None,
                    help='subset of scenario folder names (for smoke tests)')
    ap.add_argument('--max_frames', type=int, default=None,
                    help='only for smoke tests: frames per scenario')
    args = ap.parse_args()

    scen = sorted(x for x in os.listdir(args.data_dir)
                  if os.path.isdir(os.path.join(args.data_dir, x)))
    if args.scenarios:
        scen = [s for s in scen if s in args.scenarios]

    if args.max_frames:
        frames = {}
        for s in scen:
            for k, (ts, files) in enumerate(scenario_frames(
                    os.path.join(args.data_dir, s), args.max_cav)):
                if k >= args.max_frames:
                    break
                frames[(s, ts)] = process_frame(files, args.ground_margin, args.max_range)
    else:
        jobs = [(os.path.join(args.data_dir, s), args.max_cav,
                 args.ground_margin, args.max_range) for s in scen]
        frames = {}
        with Pool(args.workers) as pool:
            for part in pool.imap_unordered(process_scenario, jobs):
                frames.update(part)

    meta = {'data_dir': os.path.abspath(args.data_dir),
            'ground_margin': args.ground_margin, 'max_range': args.max_range,
            'max_cav': args.max_cav, 'cav_half_extent': CAV_HALF_EXTENT.tolist(),
            'n_frames': len(frames)}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'wb') as f:
        pickle.dump({'meta': meta, 'frames': frames}, f)
    print(f'saved {len(frames)} frames -> {args.out}')


if __name__ == '__main__':
    main()
