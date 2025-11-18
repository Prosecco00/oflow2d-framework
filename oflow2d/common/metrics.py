
import numpy as np
def epe(pred, gt, valid_mask=None):
    diff = pred - gt
    dist = np.sqrt((diff**2).sum(axis=-1))
    if valid_mask is not None: dist = dist[valid_mask>0]
    return float(dist.mean()) if dist.size>0 else float('nan')

def fl_all(pred, gt, tau=3.0, valid_mask=None):
    diff = pred - gt
    dist = np.sqrt((diff**2).sum(axis=-1))
    if valid_mask is not None: dist = dist[valid_mask>0]
    if dist.size==0: return float('nan')
    return float((dist>tau).mean()*100.0)

def angular_error(pred, gt, eps=1e-6, valid_mask=None):
    px,py = pred[...,0], pred[...,1]
    gx,gy = gt[...,0], gt[...,1]
    pnorm = np.sqrt(px*px+py*py)+eps
    gnorm = np.sqrt(gx*gx+gy*gy)+eps
    dot = (px*gx+py*gy)/(pnorm*gnorm)
    dot = np.clip(dot,-1,1)
    ang = np.degrees(np.arccos(dot))
    if valid_mask is not None: ang = ang[valid_mask>0]
    return float(ang.mean()) if ang.size>0 else float('nan')
