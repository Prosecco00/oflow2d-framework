
import os, json, time
import cv2
import numpy as np
import torch

from oflow2d.common.viz import flow_to_color
from oflow2d.adapters import make_model

# default uguali al notebook
MAX_EDGE = 640
ITERS    = 8
VIZ_FPS  = None
SAVE_FLO = True

def resize_pair(f0, f1, max_edge=MAX_EDGE):
    h, w = f0.shape[:2]
    scale = min(1.0, max_edge / max(h, w))
    if scale < 1.0:
        new_w = int((w * scale) // 8 * 8)
        new_h = int((h * scale) // 8 * 8)
        f0s = cv2.resize(f0, (new_w, new_h), interpolation=cv2.INTER_AREA)
        f1s = cv2.resize(f1, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return f0s, f1s, scale, (h, w)
    return f0, f1, 1.0, (h, w)

def upsample_flow(flow, orig_hw, scale):
    H, W = orig_hw
    if scale != 1.0:
        flow = cv2.resize(flow, (W, H), interpolation=cv2.INTER_LINEAR)
        flow = flow / scale
    return flow

def save_flo(path, flow):
    with open(path, "wb") as f:
        f.write(b"PIEH")
        np.array([flow.shape[1]], np.int32).tofile(f)  # width
        np.array([flow.shape[0]], np.int32).tofile(f)  # height
        flow.astype(np.float32).tofile(f)

def video_fps(path, default=25.0):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or default
    cap.release()
    return float(fps) if fps and fps > 0 else default

def process_video(in_path, out_dir, model,
                  max_edge=MAX_EDGE, iters=ITERS,
                  viz_fps=VIZ_FPS, save_flo_flag=SAVE_FLO):
    """Video → modello → flow + video color-coded.
    Ritorna (fps_modello, path_mp4).
    """
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(in_path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    assert len(frames) >= 2, "Il video deve avere almeno 2 frame."

    times = []
    viz_frames = []
    for i in range(len(frames) - 1):
        f0, f1 = frames[i], frames[i+1]
        f0s, f1s, scale, orig_hw = resize_pair(f0, f1, max_edge=max_edge)

        t0 = time.time()
        flow = model(f0s, f1s, iters=iters)
        times.append(time.time() - t0)

        flow = upsample_flow(flow, orig_hw, scale)

        np.save(os.path.join(out_dir, f"flow_{i:06d}.npy"), flow)
        if save_flo_flag:
            save_flo(os.path.join(out_dir, f"flow_{i:06d}.flo"), flow)

        viz_frames.append(flow_to_color(flow))

        if torch.cuda.is_available() and i % 10 == 0:
            torch.cuda.empty_cache()

    fps_model = 1.0 / np.mean(times)
    fps_out = viz_fps if viz_fps is not None else video_fps(in_path)

    h, w, _ = viz_frames[0].shape
    out_mp4 = os.path.join(out_dir, "flow_viz.mp4")
    vw = cv2.VideoWriter(
        out_mp4,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps_out,
        (w, h),
    )
    for f in viz_frames:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({"pairs": len(times), "fps_model": float(fps_model)}, f, indent=2)

    return fps_model, out_mp4

# --------- wrapper per creare i modelli -------------------------------------

def create_raft_model():
    weights_small = "/content/RAFT/models/raft-small.pth"
    weights_std   = "/content/RAFT/models/raft-sintel.pth"
    use_small = os.path.exists(weights_small)
    model = make_model(
        "raft",
        weights=(weights_small if use_small else weights_std),
        small=bool(use_small),
        mixed_precision=True,
        alternate_corr=False,
    )
    print("RAFT model ready. small:", use_small)
    return model

def create_spynet_model(device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = make_model("spynet", device=device)
    print("SPyNet model ready on", device)
    return model

def create_flownet2_model(device=None, ckpt="things"):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = make_model("flownet2", device=device, ckpt=ckpt)
    print("FlowNet2 model ready on", device, "ckpt=", ckpt)
    return model

def create_pwcnet_model(device=None, ckpt="things"):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = make_model("pwcnet", device=device, ckpt=ckpt)
    print("PWCNet model ready on", device, "ckpt=", ckpt)
    return model
