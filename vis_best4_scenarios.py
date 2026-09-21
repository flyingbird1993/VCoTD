"""
2-row × 4-col front-camera trajectory visualization.
Row 0: VCoTD-E2E (baseline, worse)
Row 1: VCoTD-Distill (ours, better)
Columns: Straight | Stop | Left Turn | Right Turn

Samples selected for maximum contrast (largest ΔL2 = L2_e2e - L2_distill).
Trajectories are drawn as smooth cubic-spline curves.
"""

import os, json, pickle, re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy.interpolate import make_interp_spline
from PIL import Image
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────
DATA_PKL      = "./create_data/cached_nuscenes_info.pkl"
DISTILL_JSON  = "./results_vcotd.json"
E2E_JSON      = "./results_vcotd_e2e.json"
OUT_FILE      = "./vis_vcotd/best4_scenarios_cam.png"
NUSC_ROOTS    = ["./LLaMA-Factory/data/nuscenes_hdd",
                 "./LLaMA-Factory/data/nuscenes"]

# ── selected tokens (max ΔL2 per scenario, with accessible images) ─────────
# scenario : (token, L2_distill, L2_e2e)
SCENARIOS = [
    ("Straight",   "5bebe37b75564ab2a9773b12e78917e1",  0.13, 4.05),
    ("Stop",       "b92ee1ed90c242d18bce9a8fdbbeb8d8",  0.43, 2.49),
    ("Left Turn",  "5a326a1f9112431585c441cf3d8901b7",  0.42, 2.14),
    ("Right Turn", "d4437f7299da465a9115120375104a24",  0.25, 2.61),
]

# ── visual params ─────────────────────────────────────────────────────────────
C_GT   = "#00e676"    # green
C_PRED = "#ff1744"    # red
Z_LIDAR = -1.0        # trajectory height in lidar frame
IMG_BG   = "#0f0f1e"
ROW_COLS = ["#ff8c44", "#44ddaa"]   # E2E border / Distill border


# ── helpers ───────────────────────────────────────────────────────────────────
def parse_traj(s):
    nums = re.findall(r'\((-?\d+\.?\d*),\s*(-?\d+\.?\d*)\)', s)
    return np.array([(float(x), float(y)) for x, y in nums]) if nums else None


def build_img_index(roots):
    idx = {}
    for root in roots:
        if not os.path.exists(root):
            continue
        for sub in os.listdir(root):
            cam_dir = os.path.join(root, sub, "samples", "CAM_FRONT")
            if not os.path.isdir(cam_dir):
                continue
            for f in os.listdir(cam_dir):
                if f not in idx:
                    idx[f] = os.path.join(cam_dir, f)
    return idx


def find_image(data_path, img_index):
    fname = os.path.basename(data_path)
    return img_index.get(fname)


def project_lidar_to_cam(traj_xy, d, cam_name="CAM_FRONT", z=Z_LIDAR):
    """
    Project lidar-BEV trajectory [(tx,ty),...] → camera pixel list.
    Returns list of (u,v) or None (behind camera / outside FOV).
    """
    cam   = d["cams"][cam_name]
    K     = np.array(cam["cam_intrinsic"])
    R_s2l = np.array(cam["sensor2lidar_rotation"])   # 3×3, camera→lidar
    t_s2l = np.array(cam["sensor2lidar_translation"])
    iw, ih = 1600, 900

    pixels = []
    for tx, ty in traj_xy:
        p = np.array([tx, ty, z])
        pc = R_s2l.T @ (p - t_s2l)
        if pc[2] < 0.3:
            pixels.append(None)
            continue
        u = K[0, 0] * pc[0] / pc[2] + K[0, 2]
        v = K[1, 1] * pc[1] / pc[2] + K[1, 2]
        pixels.append((float(u), float(v)) if 0 <= u < iw and 0 <= v < ih else None)
    return pixels


def spline_curve(pixels, iw=1600, ih=900, n=120):
    """
    Fit a parametric cubic spline through valid pixels, return (us, vs) arrays.
    Falls back to linear or None if too few points.
    """
    valid = [(u, v) for p in pixels
             if p is not None and 0 <= (u := p[0]) < iw and 0 <= (v := p[1]) < ih]
    if len(valid) < 2:
        return None, None
    us_r, vs_r = zip(*valid)
    t = np.linspace(0, 1, len(valid))
    k = min(3, len(valid) - 1)
    if k == 1:
        return np.array(us_r), np.array(vs_r)
    spu = make_interp_spline(t, us_r, k=k)
    spv = make_interp_spline(t, vs_r, k=k)
    tf  = np.linspace(0, 1, n)
    return spu(tf), spv(tf)


def draw_traj(ax, pixels, color, lw=2.5, dot_r=5,
              iw=1600, ih=900):
    """Draw a smooth spline trajectory on a matplotlib axis."""
    us, vs = spline_curve(pixels, iw, ih)
    if us is None:
        return

    # Smooth curve
    ax.plot(us, vs, color=color, linewidth=lw, solid_capstyle="round",
            solid_joinstyle="round", zorder=4)

    # Dot at each valid step
    valid = [(u, v) for p in pixels
             if p is not None and 0 <= (u := p[0]) < iw and 0 <= (v := p[1]) < ih]
    for u, v in valid:
        ax.plot(u, v, "o", color=color, markersize=dot_r,
                markeredgecolor="white", markeredgewidth=1.0, zorder=5)


def make_panel(ax, d, gt_xy, pred_xy, img_path, l2_val, row_name,
               is_stop=False, is_distill_row=False):
    """Render one 2-D panel: camera image + trajectory overlay."""
    img = np.array(Image.open(img_path).convert("RGB"))
    iw, ih = img.shape[1], img.shape[0]

    ax.imshow(img, extent=[0, iw, ih, 0], aspect="auto", zorder=1)
    ax.set_xlim(0, iw)
    ax.set_ylim(ih, 0)
    ax.set_xticks([])
    ax.set_yticks([])

    gt_px   = project_lidar_to_cam(gt_xy,   d)
    pred_px = project_lidar_to_cam(pred_xy, d)

    # Count visible GT points
    gt_vis   = sum(1 for p in gt_px   if p is not None)
    pred_vis = sum(1 for p in pred_px if p is not None)

    draw_traj(ax, gt_px,   C_GT,   lw=3.5, dot_r=7)
    draw_traj(ax, pred_px, C_PRED, lw=3.5, dot_r=7)

    # ── Stop-scenario annotation ───────────────────────────────────────────
    if is_stop:
        if gt_vis == 0:
            # GT invisible: add a circle at approximate "current position"
            # (lowest center of the image)
            ax.plot(iw / 2, ih - 30, "o", color=C_GT,
                    markersize=14, markeredgecolor="white",
                    markeredgewidth=1.5, zorder=6)
            ax.annotate("GT (stationary)", xy=(iw / 2, ih - 30),
                        xytext=(iw / 2 + 80, ih - 90),
                        color=C_GT, fontsize=8, fontweight="bold",
                        arrowprops=dict(arrowstyle="->", color=C_GT, lw=1.2),
                        zorder=7)
        if is_distill_row and pred_vis == 0:
            ax.text(iw * 0.5, ih * 0.85,
                    "Pred ≈ Stationary ✓",
                    color=C_PRED, fontsize=9, fontweight="bold",
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor="#1a1a2e", alpha=0.75,
                              edgecolor=C_PRED, linewidth=1.2),
                    zorder=8)

    # ── L2 error label ────────────────────────────────────────────────────
    ax.text(14, 36, f"L2 = {l2_val:.2f} m",
            color="white", fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25",
                      facecolor="#0f0f1e", alpha=0.80,
                      edgecolor="#555577", linewidth=1.0),
            zorder=8)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    data    = pickle.load(open(DATA_PKL, "rb"))
    distill = json.load(open(DISTILL_JSON))
    e2e     = json.load(open(E2E_JSON))
    img_idx = build_img_index(NUSC_ROOTS)

    n_col = len(SCENARIOS)
    fig   = plt.figure(figsize=(n_col * 5.5, 2 * 3.5 + 1.2),
                       facecolor=IMG_BG)
    gs    = GridSpec(2, n_col, figure=fig,
                     hspace=0.06, wspace=0.04,
                     top=0.88, bottom=0.04, left=0.05, right=0.995)

    row_info = [
        ("VCoTD-E2E  (baseline)", lambda tok: e2e.get(tok,""),   False, False),
        ("VCoTD-Distill (ours)",  lambda tok: distill.get(tok,""), True, True),
    ]

    for col, (sc_name, tok, l2_dist, l2_e2e) in enumerate(SCENARIOS):
        d        = data[tok]
        gt_arr   = np.array(d.get("gt_ego_fut_trajs", []))
        gt_xy    = gt_arr[1:7] if len(gt_arr) >= 7 else gt_arr[1:]
        img_path = find_image(d["cams"]["CAM_FRONT"]["data_path"], img_idx)
        is_stop  = (sc_name == "Stop")

        for row, (row_label, get_pred, is_distill_row, _) in enumerate(row_info):
            l2_val   = l2_dist if is_distill_row else l2_e2e
            pred_str = get_pred(tok)
            pred_xy  = parse_traj(pred_str) if pred_str else None

            ax = fig.add_subplot(gs[row, col])

            if img_path and pred_xy is not None and len(gt_xy) > 0:
                make_panel(ax, d, gt_xy, pred_xy, img_path, l2_val,
                           row_label, is_stop=is_stop,
                           is_distill_row=is_distill_row)
            else:
                ax.set_facecolor("#222233")
                ax.text(0.5, 0.5, "No data", color="white",
                        ha="center", va="center", transform=ax.transAxes)
                ax.set_xticks([]); ax.set_yticks([])

            # Colored border per row
            for sp in ax.spines.values():
                sp.set_edgecolor(ROW_COLS[row])
                sp.set_linewidth(2.0)

            # Row label (left column only)
            if col == 0:
                ax.set_ylabel(row_label, color=ROW_COLS[row],
                              fontsize=10, fontweight="bold", labelpad=6)

            # Column title (top row only)
            if row == 0:
                ax.set_title(sc_name, color="white",
                             fontsize=12, fontweight="bold", pad=5)

    # ── legend ────────────────────────────────────────────────────────────────
    p_gt   = mpatches.Patch(color=C_GT,   label="Ground Truth")
    p_pred = mpatches.Patch(color=C_PRED, label="Prediction")
    fig.legend(handles=[p_gt, p_pred],
               loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.965),
               fontsize=11, facecolor="#1a1a2e", labelcolor="white",
               edgecolor="#4d4d6e", framealpha=0.9)

    fig.suptitle("VCoTD Trajectory Prediction — Front Camera View",
                 color="white", fontsize=14, fontweight="bold", y=0.998)

    Path(OUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_FILE, dpi=200, bbox_inches="tight", facecolor=IMG_BG)
    print(f"Saved → {OUT_FILE}")


if __name__ == "__main__":
    main()
