
import numpy as np, cv2
def flow_to_color(flow, clip_flow=None):
    fx = flow[...,0]; fy = flow[...,1]
    rad = (fx**2 + fy**2) ** 0.5
    ang = np.arctan2(fy, fx)
    if clip_flow is not None: rad = np.clip(rad, 0, clip_flow)
    rad_n = (rad - rad.min()) / (rad.max() - rad.min() + 1e-8)
    ang_n = (ang + np.pi) / (2*np.pi)
    hsv = np.zeros((*flow.shape[:2],3), dtype=np.float32)
    hsv[...,0] = ang_n * 179.0; hsv[...,1] = 1.0; hsv[...,2] = rad_n
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return (bgr[:,:,::-1]*255.0).astype('uint8')
