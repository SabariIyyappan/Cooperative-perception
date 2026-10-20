# -*- coding: utf-8 -*-
"""
RQ1 figures from analyze_rq1.py outputs.

fig_rq1_recall_<suffix>_iou<iou>.{png,pdf}
    top row: recall vs pose noise per visibility class, one panel per mode
    bottom row: overall AP at the same IoU (own axis, never a dual axis)
fig_rq1_ba_delta_<suffix>_iou<iou>.{png,pdf}
    fusion_BA - fusion_noBA recall per class vs pose noise, with 95% CI

Usage:
  python opencood/tools/rq1/plot_rq1.py --results_dir ../results/rq1 --suffix lidar_tau3
"""

import argparse
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# reference palette, categorical slots 1-3 in fixed order (light surface)
SURFACE = '#fcfcfb'
INK, INK2, GRID = '#1f1f1e', '#5f5e58', '#e4e3df'
SERIES = {'co_visible': ('#2a78d6', 'o', 'co-visible'),
          'ego_only': ('#eb6834', 's', 'singly: ego only'),
          'peer_only': ('#1baf7a', '^', 'singly: peer only')}
MODE_TITLES = {'ego_only': 'No fusion (ego only)',
               'fusion_noBA': 'Fusion, no box alignment',
               'fusion_BA': 'CoAlign (box alignment)'}
MODE_ORDER = ('ego_only', 'fusion_noBA', 'fusion_BA')


def style_axis(ax):
    ax.set_facecolor(SURFACE)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_dir', required=True)
    ap.add_argument('--suffix', default='lidar_tau3')
    ap.add_argument('--iou', type=float, default=0.7)
    args = ap.parse_args()
    rd, iou = args.results_dir, args.iou
    plt.rcParams.update({'font.size': 9, 'text.color': INK, 'axes.labelcolor': INK2})

    summary = read_csv(os.path.join(rd, f'summary_{args.suffix}.csv'))
    aps = read_csv(os.path.join(rd, 'ap.csv'))
    deltas = read_csv(os.path.join(rd, f'deltas_{args.suffix}.csv'))
    ap_key = {0.3: 'ap30', 0.5: 'ap50', 0.7: 'ap70'}[iou]

    rec = defaultdict(list)  # (mode, cls) -> [(noise, r, lo, hi)]
    for r in summary:
        if float(r['iou']) == iou:
            rec[(r['mode'], r['cls'])].append(
                (float(r['noise']), float(r['recall']), float(r['ci_lo']), float(r['ci_hi'])))
    modes = [m for m in MODE_ORDER if any(k[0] == m for k in rec)]

    # ---------------- figure 1
    fig, axes = plt.subplots(2, len(modes), figsize=(3.4 * len(modes), 4.6),
                             sharex=True, gridspec_kw={'height_ratios': [3, 1.2]},
                             squeeze=False, facecolor=SURFACE)
    for j, mode in enumerate(modes):
        ax = axes[0, j]
        style_axis(ax)
        for cls, (color, marker, label) in SERIES.items():
            pts = sorted(rec[(mode, cls)])
            if not pts:
                continue
            x, y, lo, hi = zip(*pts)
            ax.fill_between(x, lo, hi, color=color, alpha=0.12, linewidth=0)
            ax.plot(x, y, color=color, linewidth=2, marker=marker, markersize=6,
                    markeredgecolor=SURFACE, markeredgewidth=1.2, label=label)
        ax.set_title(MODE_TITLES.get(mode, mode), fontsize=9, color=INK, loc='left')
        ax.set_ylim(0, 1.02)
        if j == 0:
            ax.set_ylabel(f'Recall @ IoU {iou:g}')

        ax2 = axes[1, j]
        style_axis(ax2)
        pts = sorted((float(r['noise']), float(r[ap_key])) for r in aps if r['mode'] == mode)
        if pts:
            x, y = zip(*pts)
            ax2.plot(x, y, color=INK2, linewidth=2, marker='D', markersize=5,
                     markeredgecolor=SURFACE, markeredgewidth=1.0)
        ax2.set_ylim(0, 1.02)
        ax2.set_xlabel('Pose noise σ (m and °)')
        if j == 0:
            ax2.set_ylabel(f'AP{int(iou * 100)}')
        ax2.set_xticks(sorted({p[0] for p in pts}) if pts else [])

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(rd, f'fig_rq1_recall_{args.suffix}_iou{iou:g}.{ext}'),
                    dpi=200, facecolor=SURFACE)
    plt.close(fig)

    # ---------------- figure 2: BA - noBA
    dl = defaultdict(list)
    for r in deltas:
        if r['kind'] == 'BA-noBA' and float(r['iou']) == iou:
            dl[r['what']].append((float(r['noise']), float(r['value']),
                                  float(r['ci_lo']), float(r['ci_hi'])))
    if dl:
        fig, ax = plt.subplots(figsize=(5.2, 3.4), facecolor=SURFACE)
        style_axis(ax)
        ax.axhline(0, color=INK2, linewidth=1)
        for cls, (color, marker, label) in SERIES.items():
            pts = sorted(dl.get(cls, []))
            if not pts:
                continue
            x, y, lo, hi = zip(*pts)
            ax.fill_between(x, lo, hi, color=color, alpha=0.12, linewidth=0)
            ax.plot(x, y, color=color, linewidth=2, marker=marker, markersize=6,
                    markeredgecolor=SURFACE, markeredgewidth=1.2, label=label)
            ax.set_xticks(x)
        ax.set_xlabel('Pose noise σ (m and °)')
        ax.set_ylabel(f'Δ recall @ IoU {iou:g}\n(CoAlign − no box alignment)')
        ax.legend(frameon=False, fontsize=8, loc='upper left')
        fig.tight_layout()
        for ext in ('png', 'pdf'):
            fig.savefig(os.path.join(rd, f'fig_rq1_ba_delta_{args.suffix}_iou{iou:g}.{ext}'),
                        dpi=200, facecolor=SURFACE)
        plt.close(fig)
    print('figures written to', rd)


if __name__ == '__main__':
    main()
