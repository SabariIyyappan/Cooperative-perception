# -*- coding: utf-8 -*-
"""
RQ1 analysis: recall by visibility class vs. pose noise, from the dumps of
run_rq1_sweep.py and the labels of compute_visibility.py. No inference here.

Visibility of GT object o in a frame, over the agents actually used in that
frame (cav_id_list, ego first):
  lidar label: V(o) = {k : n_pts(o, k) >= tau}
  geom  label: V(o) = {k : ray_free_frac(o, k) > 0}   (cross-check)
Classes: none (|V|=0), ego_only (V={ego}), peer_only (|V|=1, not ego),
         co_visible (|V|>=2); single = ego_only + peer_only.

Recall: GT matched (greedy rule of the AP evaluation) by a prediction with
score >= score_thresh at the given IoU.

Outputs in <results_dir>:
  summary_<label>_tau<tau>.csv   mode, noise, iou, cls, n_gt, recall, ci_lo, ci_hi
  deltas_<label>_tau<tau>.csv    gap (co_visible - single) per mode, and
                                 fusion_BA - fusion_noBA per class
  ap.csv                         AP@0.3/0.5/0.7 per mode and noise
  label_agreement_tau<tau>.txt   lidar vs geometric label confusion

Usage:
  python opencood/tools/rq1/analyze_rq1.py --results_dir ../results/rq1
"""

import argparse
import csv
import glob
import os
import pickle
from collections import Counter, defaultdict

import numpy as np

CLASSES = ('none', 'ego_only', 'peer_only', 'co_visible', 'single', 'all_visible', 'all')


def load_runs(results_dir):
    runs = {}
    for path in sorted(glob.glob(os.path.join(results_dir, '*', 'noise_*', 'frames.pkl'))):
        mode = path.split(os.sep)[-3]
        with open(path, 'rb') as f:
            d = pickle.load(f)
        runs[(mode, float(d['std']))] = d
    return runs


def visible_set(obj_vis, agents, label, tau):
    if label == 'lidar':
        return [k for k in agents if k in obj_vis and obj_vis[k][0] >= tau]
    return [k for k in agents if k in obj_vis and obj_vis[k][1] > 0]


def classify(vset, ego):
    if len(vset) == 0:
        return 'none'
    if len(vset) >= 2:
        return 'co_visible'
    return 'ego_only' if vset[0] == ego else 'peer_only'


def per_object_table(run, vis_frames, label, tau, score_thresh, ious, exclude_cav):
    """
    Returns list of rows (scenario, cls, {iou: hit}) for every GT object.
    """
    rows = []
    missing = 0
    for fr in run['frames']:
        rec = vis_frames[tuple(fr['key'])]
        agents = fr['cav_id_list']
        ego = agents[0]
        hits = {}
        for iou in ious:
            m = fr['matches'][iou]
            ok = (m['tp'] == 1) & (m['score'] >= score_thresh)
            hits[iou] = set(m['gt_id'][ok].tolist())
        for gid in fr['gt_ids'].tolist():
            obj = rec['objects'].get(gid)
            if obj is None:
                missing += 1
                continue
            if exclude_cav and obj['is_cav']:
                continue
            cls = classify(visible_set(obj['vis'], agents, label, tau), ego)
            rows.append((fr['key'][0], cls, {iou: gid in hits[iou] for iou in ious}))
    return rows, missing


def class_members(cls):
    if cls == 'single':
        return ('ego_only', 'peer_only')
    if cls == 'all_visible':
        return ('ego_only', 'peer_only', 'co_visible')
    if cls == 'all':
        return ('none', 'ego_only', 'peer_only', 'co_visible')
    return (cls,)


def scenario_counts(rows, iou):
    """{scenario: {base_cls: [n, hits]}}"""
    out = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for scen, cls, hit in rows:
        out[scen][cls][0] += 1
        out[scen][cls][1] += int(hit[iou])
    return out


def recall_from(counts, scenarios, cls):
    n = h = 0
    for s in scenarios:
        for c in class_members(cls):
            n += counts[s][c][0]
            h += counts[s][c][1]
    return (h / n if n else float('nan')), n


def bootstrap(stat_fn, scenarios, n_boot, rng):
    vals = []
    for _ in range(n_boot):
        sample = rng.choice(scenarios, size=len(scenarios), replace=True)
        vals.append(stat_fn(sample))
    vals = np.array(vals, dtype=np.float64)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return float('nan'), float('nan')
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_dir', required=True)
    ap.add_argument('--vis', default=None, help='default: <results_dir>/visibility_opv2v_test.pkl')
    ap.add_argument('--label', choices=('lidar', 'geom'), default='lidar')
    ap.add_argument('--tau', type=int, default=3)
    ap.add_argument('--score_thresh', type=float, default=0.2)
    ap.add_argument('--ious', type=float, nargs='+', default=[0.5, 0.7])
    ap.add_argument('--n_boot', type=int, default=1000)
    ap.add_argument('--exclude_cav', action='store_true',
                    help='drop GT objects that are themselves CAVs')
    args = ap.parse_args()

    vis_path = args.vis or os.path.join(args.results_dir, 'visibility_opv2v_test.pkl')
    with open(vis_path, 'rb') as f:
        vis_frames = pickle.load(f)['frames']
    runs = load_runs(args.results_dir)
    assert runs, f'no frames.pkl under {args.results_dir}'
    rng = np.random.default_rng(0)
    suffix = f'{args.label}_tau{args.tau}' + ('_nocav' if args.exclude_cav else '')

    tables, counts = {}, {}
    for (mode, std), run in sorted(runs.items()):
        rows, missing = per_object_table(run, vis_frames, args.label, args.tau,
                                         args.score_thresh, args.ious, args.exclude_cav)
        if missing:
            print(f'WARNING {mode} {std}: {missing} GT objects without visibility labels')
        tables[(mode, std)] = rows
        for iou in args.ious:
            counts[(mode, std, iou)] = scenario_counts(rows, iou)

    # --- summary: recall per class with scenario bootstrap CI
    summary = []
    for (mode, std, iou), cnt in sorted(counts.items()):
        scen = sorted(cnt.keys())
        for cls in CLASSES:
            r, n = recall_from(cnt, scen, cls)
            lo, hi = bootstrap(lambda s: recall_from(cnt, s, cls)[0], scen, args.n_boot, rng)
            summary.append([mode, std, iou, cls, n, r, lo, hi])
    with open(os.path.join(args.results_dir, f'summary_{suffix}.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['mode', 'noise', 'iou', 'cls', 'n_gt', 'recall', 'ci_lo', 'ci_hi'])
        w.writerows(summary)

    # --- deltas: gap within mode, and BA - noBA per class (paired bootstrap)
    deltas = []
    for (mode, std, iou), cnt in sorted(counts.items()):
        scen = sorted(cnt.keys())
        for single in ('single', 'ego_only', 'peer_only'):
            fn = lambda s, single=single: recall_from(cnt, s, 'co_visible')[0] - recall_from(cnt, s, single)[0]
            lo, hi = bootstrap(fn, scen, args.n_boot, rng)
            deltas.append(['gap', mode, std, iou, f'co_visible-{single}', fn(scen), lo, hi])
        if mode == 'fusion_BA' and ('fusion_noBA', std, iou) in counts:
            cnt0 = counts[('fusion_noBA', std, iou)]
            for cls in CLASSES:
                fn = lambda s, cls=cls: recall_from(cnt, s, cls)[0] - recall_from(cnt0, s, cls)[0]
                lo, hi = bootstrap(fn, scen, args.n_boot, rng)
                deltas.append(['BA-noBA', mode, std, iou, cls, fn(scen), lo, hi])
    with open(os.path.join(args.results_dir, f'deltas_{suffix}.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['kind', 'mode', 'noise', 'iou', 'what', 'value', 'ci_lo', 'ci_hi'])
        w.writerows(deltas)

    # --- AP
    with open(os.path.join(args.results_dir, 'ap.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['mode', 'noise', 'ap30', 'ap50', 'ap70'])
        for (mode, std), run in sorted(runs.items()):
            w.writerow([mode, std, run['ap'][0.3], run['ap'][0.5], run['ap'][0.7]])

    # --- lidar vs geometric agreement (on the GT objects / agents of one run)
    ref_key = sorted(runs.keys())[0]
    pair_conf, cls_conf = Counter(), Counter()
    for fr in runs[ref_key]['frames']:
        rec = vis_frames[tuple(fr['key'])]
        agents = fr['cav_id_list']
        for gid in fr['gt_ids'].tolist():
            obj = rec['objects'].get(gid)
            if obj is None:
                continue
            for k in agents:
                if k in obj['vis']:
                    n_pts, free, _ = obj['vis'][k]
                    pair_conf[(n_pts >= args.tau, free > 0)] += 1
            cls_conf[(classify(visible_set(obj['vis'], agents, 'lidar', args.tau), agents[0]),
                      classify(visible_set(obj['vis'], agents, 'geom', args.tau), agents[0]))] += 1
    lines = [f'reference run: {ref_key}, tau={args.tau}',
             '(object, agent) pairs: (lidar_visible, geom_visible) -> count']
    lines += [f'  {k}: {v}' for k, v in sorted(pair_conf.items())]
    lines.append('object classes: (lidar_class, geom_class) -> count')
    lines += [f'  {k}: {v}' for k, v in sorted(cls_conf.items(), key=lambda x: -x[1])]
    with open(os.path.join(args.results_dir, f'label_agreement_tau{args.tau}.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')

    # --- console overview
    print(f'label={args.label} tau={args.tau} score>={args.score_thresh}')
    for iou in args.ious:
        print(f'\nIoU {iou}: recall [n]  (none / ego_only / peer_only / co_visible / single)')
        for row_mode, row_std in sorted({(m, s) for (m, s, i) in counts if i == iou}):
            cells = []
            for cls in ('none', 'ego_only', 'peer_only', 'co_visible', 'single'):
                r = next(x for x in summary if x[:4] == [row_mode, row_std, iou, cls])
                cells.append(f'{r[5]:.3f} [{r[4]}]')
            ap_ = runs[(row_mode, row_std)]['ap']
            print(f'  {row_mode:12s} {row_std:<4g} AP{int(iou*100)}={ap_[iou]:.3f} | ' + ' / '.join(cells))


if __name__ == '__main__':
    main()
