# RQ1: Does robust fusion suppress objects only one car can see?

Research question: robust collaborative-perception methods work by agreement between cars. Do they
throw away objects that only one agent sees, and does the overall AP score hide it?

**Answer (OPV2V test, CoAlign weights):** No. CoAlign's pose correction does not suppress objects only
a neighbour sees; under pose noise it recovers them best. But those objects are the most fragile as
noise grows, and overall AP understates it.

Dashboard (private until shared): https://claude.ai/artifact/TTp7AdnT8yeVwFS5USHof8

## What was run

Same trained CoAlign network, three ways, on 2,170 test frames, at pose noise σ = 0, 0.1, 0.2, 0.4, 0.8
(σ metres of position error and σ degrees of heading error; identical noisy poses for every model):

| Mode | What the ego uses |
|---|---|
| `ego_only` | its own LiDAR only (no fusion) |
| `fusion_noBA` | neighbours' features merged, poses uncorrected |
| `fusion_BA` | CoAlign: poses corrected by box alignment, then merged |

## How objects are classified

1. **Count laser points.** For every real object and every car, count that car's own LiDAR points inside
   the object's 3D box (true poses, ignoring points within 0.2 m of the box bottom).
2. **"Saw it" = 3 or more points.** The visible set V is the cars (used in that frame) that saw it.
3. **Classes:** `co_visible` (|V| ≥ 2), `ego_only` (V = ego), `peer_only` (|V| = 1, not the ego), none (|V| = 0).

A geometric ray cast against other vehicle boxes is stored as a cross-check (about 90% agreement).
Thresholds of 1 and 10 points, ray-cast labels, and excluding connected cars give the same conclusion.

## Results (recall at IoU 0.7, σ = 0 → 0.8)

| Mode | AP70 | Seen by 2+ cars | Ego only | Peer only |
|---|---|---|---|---|
| Fusion, no alignment | 0.915 → 0.755 | 0.963 → 0.831 | 0.906 → 0.891 | 0.770 → 0.422 |
| CoAlign | 0.913 → 0.841 | 0.961 → 0.906 | 0.907 → 0.896 | 0.761 → 0.583 |
| No fusion | 0.743 | 0.828 | 0.893 | 0.121 |

Our CoAlign AP matches the authors' published values at σ = 0, 0.2 and 0.4 (0.913 / 0.901 / 0.889).

## Files

| File | Purpose |
|---|---|
| `compute_visibility.py` | per object and car: LiDAR point count plus geometric ray cast |
| `rq1_utils.py` | seeded noise per sample, ground truth that keeps object ids, id-aware matching, ego-only slicing |
| `run_rq1_sweep.py` | noise sweep for one mode; dumps per-frame matches |
| `analyze_rq1.py` | recall by class, bootstrap intervals, alignment effect, label agreement |
| `plot_rq1.py` | recall-vs-noise and alignment-effect figures |

## Reproduce

```bash
conda activate coalign && cd CoAlign
python opencood/tools/rq1/compute_visibility.py --data_dir dataset/OPV2V/test \
    --out ../results/rq1/visibility_opv2v_test.pkl
for m in fusion_BA fusion_noBA ego_only; do
  python opencood/tools/rq1/run_rq1_sweep.py \
    --model_dir opencood/logs/opv2v_coalign_woba_withnoise_2023_02_10_13_46_05 \
    --mode $m --out_dir ../results/rq1
done
python opencood/tools/rq1/analyze_rq1.py --results_dir ../results/rq1
python opencood/tools/rq1/plot_rq1.py --results_dir ../results/rq1 --iou 0.7
```

The library code is unchanged, so the baseline reproduction still holds.

## Limits and next steps

- "Fusion without alignment" is CoAlign's network with a step turned off, not an independently trained
  model (e.g. V2X-ViT). Standard No-Fusion and V2X-ViT models still need to be trained.
- CoAlign uses agreement only to correct poses, not to filter detections. A consensus-filtering method
  is untested.
- RQ2 (did not see vs could not see) will use ray casting through each neighbour's point cloud.
