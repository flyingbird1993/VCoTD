"""
VCoTD BEV Trajectory Visualization (no nuscenes dependency).
Plots predicted vs GT future trajectories for a set of val samples.
"""
import json, pickle, re, argparse, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


def parse_traj_str(s):
    nums = re.findall(r'\((-?\d+\.?\d*),(-?\d+\.?\d*)\)', s)
    return np.array([(float(x), float(y)) for x, y in nums]) if nums else None


def draw_sample(ax, pred, gt, token, hist=None):
    ax.set_aspect('equal')
    ax.set_facecolor('#1a1a2e')

    # Grid
    ax.grid(color='#2d2d4e', linewidth=0.5, linestyle='--')
    ax.axhline(0, color='#4d4d6e', linewidth=0.8)
    ax.axhline(0, color='#4d4d6e', linewidth=0.8)

    # Ego vehicle box
    car = plt.Rectangle((-1.0, -2.25), 2.0, 4.5, color='#ffd700', alpha=0.85, zorder=5)
    ax.add_patch(car)

    # Historical trajectory
    if hist is not None and len(hist) > 0:
        ax.plot(hist[:, 0], hist[:, 1], 'o--', color='#aaaaaa',
                markersize=3, linewidth=1.2, alpha=0.6, label='History', zorder=3)

    # GT trajectory
    if gt is not None:
        gt = gt[:6]
        ax.plot(gt[:, 0], gt[:, 1], 'o-', color='#00ff7f',
                markersize=5, linewidth=2.0, label='GT', zorder=4)
        for i, (x, y) in enumerate(gt):
            ax.annotate(f'{i+1}', (x, y), textcoords='offset points',
                        xytext=(4, 4), color='#00ff7f', fontsize=6)

    # Predicted trajectory
    if pred is not None:
        ax.plot(pred[:, 0], pred[:, 1], 's-', color='#ff4444',
                markersize=5, linewidth=2.0, label='VCoTD (ours)', zorder=4)
        for i, (x, y) in enumerate(pred):
            ax.annotate(f'{i+1}', (x, y), textcoords='offset points',
                        xytext=(4, -8), color='#ff4444', fontsize=6)

    ax.set_xlim(-8, 8)
    ax.set_ylim(-5, 25)
    ax.set_xlabel('X (m)', color='white', fontsize=7)
    ax.set_ylabel('Y (m)', color='white', fontsize=7)
    ax.tick_params(colors='white', labelsize=6)
    for spine in ax.spines.values():
        spine.set_edgecolor('#4d4d6e')
    ax.set_title(token[:12] + '...', color='white', fontsize=7, pad=3)
    ax.legend(loc='upper right', fontsize=6, facecolor='#2d2d4e',
              labelcolor='white', edgecolor='#4d4d6e')


def main(args):
    data = pickle.load(open('./create_data/cached_nuscenes_info.pkl', 'rb'))
    split = json.load(open('./create_data/full_split.json', 'r'))
    results = json.load(open(args.result_file, 'r'))

    tokens = [t for t in split[args.split] if t in results][:args.num_samples]

    ncols = 4
    nrows = (len(tokens) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 4))
    fig.patch.set_facecolor('#0f0f1e')
    axes = axes.flatten()

    for i, token in enumerate(tokens):
        d = data[token]
        pred = parse_traj_str(results[token])

        gt_arr = np.array(d.get('gt_ego_fut_trajs', []))
        gt = gt_arr[1:7] if len(gt_arr) >= 7 else None  # skip t=0

        hist_arr = np.array(d.get('gt_ego_his_trajs', []))
        hist = hist_arr if len(hist_arr) > 0 else None

        draw_sample(axes[i], pred, gt, token, hist)

    # Hide unused subplots
    for j in range(len(tokens), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f'VCoTD Trajectory Visualization — {args.split} split\n'
                 f'Red=Predicted  Green=GT  Yellow=Ego Vehicle',
                 color='white', fontsize=11, y=1.01)
    plt.tight_layout()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0f0f1e')
    print(f'Saved to {out}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--result_file', default='results_vcotd.json')
    p.add_argument('--split', default='val')
    p.add_argument('--num_samples', type=int, default=16)
    p.add_argument('--output', default='vis_vcotd/traj_visualization.png')
    main(p.parse_args())
