"""
Save each of the 8 panels (2 models × 4 scenarios) as individual images.
Trajectories are drawn as smooth cubic-spline curves directly on the camera image.
Stop column: no text annotations.
Other columns: L2 error label in top-left corner.

Output folder: vis_vcotd/panels/
  e2e_straight.jpg  e2e_stop.jpg  e2e_left_turn.jpg  e2e_right_turn.jpg
  distill_straight.jpg  distill_stop.jpg  distill_left_turn.jpg  distill_right_turn.jpg
"""

import os, json, pickle, re
import numpy as np
from scipy.interpolate import make_interp_spline
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────
DATA_PKL      = "./create_data/cached_nuscenes_info.pkl"
DISTILL_JSON  = "./results_vcotd.json"
E2E_JSON      = "./results_vcotd_e2e.json"
OUT_DIR       = Path("./vis_vcotd/panels")
NUSC_ROOTS    = ["./LLaMA-Factory/data/nuscenes_hdd",
                 "./LLaMA-Factory/data/nuscenes"]

SCENARIOS = [
    ("straight",   "5bebe37b75564ab2a9773b12e78917e1",  0.13, 4.05),
    ("stop",       "b92ee1ed90c242d18bce9a8fdbbeb8d8",  0.43, 2.49),
    ("left_turn",  "5a326a1f9112431585c441cf3d8901b7",  0.42, 2.14),
    ("right_turn", "d4437f7299da465a9115120375104a24",  0.25, 2.61),
]

# RGB tuples for PIL
C_GT_PIL   = (0,   230, 118)   # green
C_PRED_PIL = (255,  23,  68)   # red
C_WHITE    = (255, 255, 255)

Z_LIDAR = -1.0
IW, IH  = 1600, 900


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


def project_lidar_to_cam(traj_xy, d, z=Z_LIDAR):
    cam   = d["cams"]["CAM_FRONT"]
    K     = np.array(cam["cam_intrinsic"])
    R_s2l = np.array(cam["sensor2lidar_rotation"])
    t_s2l = np.array(cam["sensor2lidar_translation"])
    pixels = []
    for tx, ty in traj_xy:
        pc = R_s2l.T @ (np.array([tx, ty, z]) - t_s2l)
        if pc[2] < 0.3:
            pixels.append(None)
            continue
        u = K[0, 0] * pc[0] / pc[2] + K[0, 2]
        v = K[1, 1] * pc[1] / pc[2] + K[1, 2]
        pixels.append((float(u), float(v)) if 0 <= u < IW and 0 <= v < IH else None)
    return pixels


def spline_points(pixels, n=200):
    """Return dense (us, vs) arrays from valid pixel list via cubic spline."""
    valid = [(u, v) for p in pixels
             if p is not None and 0 <= (u := p[0]) < IW and 0 <= (v := p[1]) < IH]
    if len(valid) < 2:
        return None, None
    us_r, vs_r = zip(*valid)
    t = np.linspace(0, 1, len(valid))
    k = min(3, len(valid) - 1)
    if k == 1:
        return np.array(us_r), np.array(vs_r)
    tf = np.linspace(0, 1, n)
    return make_interp_spline(t, us_r, k=k)(tf), make_interp_spline(t, vs_r, k=k)(tf)


def draw_traj_pil(draw, pixels, color,
                  line_w=7, dot_r=14, outline_w=2):
    """Draw smooth spline trajectory on a PIL ImageDraw canvas."""
    us, vs = spline_points(pixels)
    if us is not None:
        pts = [(float(x), float(y)) for x, y in zip(us, vs)]
        for i in range(len(pts) - 1):
            draw.line([pts[i], pts[i + 1]], fill=color, width=line_w)

    # Dots at each original valid step
    valid = [(u, v) for p in pixels
             if p is not None and 0 <= (u := p[0]) < IW and 0 <= (v := p[1]) < IH]
    for u, v in valid:
        draw.ellipse([u - dot_r, v - dot_r, u + dot_r, v + dot_r],
                     fill=color, outline=C_WHITE, width=outline_w)


def add_l2_label(draw, l2_val, font_size=40):
    """Draw L2 error badge in top-left corner."""
    text = f"L2 = {l2_val:.2f} m"
    # Draw dark background rectangle then white text
    x0, y0, x1, y1 = 18, 18, 310, 18 + font_size + 16
    draw.rectangle([x0, y0, x1, y1],
                   fill=(15, 15, 30, 200), outline=(100, 100, 150), width=2)
    draw.text((x0 + 12, y0 + 8), text, fill=C_WHITE)


def render_single(img_path, d, gt_xy, pred_xy,
                  l2_val=None, add_text=True):
    """
    Load camera image, draw GT + pred trajectories, return PIL Image.
    add_text=False → skip L2 label (used for stop column).
    """
    img  = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    gt_px   = project_lidar_to_cam(gt_xy,   d)
    pred_px = project_lidar_to_cam(pred_xy, d)

    draw_traj_pil(draw, gt_px,   C_GT_PIL)
    draw_traj_pil(draw, pred_px, C_PRED_PIL)

    if add_text and l2_val is not None:
        add_l2_label(draw, l2_val)

    return img


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    data    = pickle.load(open(DATA_PKL, "rb"))
    distill = json.load(open(DISTILL_JSON))
    e2e     = json.load(open(E2E_JSON))
    img_idx = build_img_index(NUSC_ROOTS)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = [
        ("e2e",     lambda tok: e2e.get(tok, ""),     False),  # (prefix, get_fn, is_distill)
        ("distill", lambda tok: distill.get(tok, ""), True),
    ]

    for sc_name, tok, l2_dist, l2_e2e in SCENARIOS:
        d        = data[tok]
        gt_arr   = np.array(d.get("gt_ego_fut_trajs", []))
        gt_xy    = gt_arr[1:7] if len(gt_arr) >= 7 else gt_arr[1:]
        img_path = img_idx.get(os.path.basename(d["cams"]["CAM_FRONT"]["data_path"]))
        is_stop  = (sc_name == "stop")

        if img_path is None:
            print(f"[WARN] image not found for {sc_name}/{tok[:8]}")
            continue

        for prefix, get_pred, is_distill in rows:
            pred_str = get_pred(tok)
            pred_xy  = parse_traj(pred_str)
            if pred_xy is None or len(gt_xy) == 0:
                print(f"[WARN] missing prediction for {prefix}/{sc_name}")
                continue

            l2_val   = l2_dist if is_distill else l2_e2e
            add_text = not is_stop          # no text for stop column

            panel = render_single(img_path, d, gt_xy, pred_xy,
                                  l2_val=l2_val, add_text=add_text)

            out_path = OUT_DIR / f"{prefix}_{sc_name}.jpg"
            panel.save(out_path, quality=95)
            print(f"Saved: {out_path}  (L2={l2_val:.2f})")


if __name__ == "__main__":
    main()
