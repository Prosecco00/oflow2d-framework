
import torch, numpy as np, torch.nn.functional as F

def _to_tensor(img):
    # img: HxWx3 uint8 RGB -> (1,3,H,W) float in [0,1]
    t = torch.from_numpy(img).permute(2,0,1).float().div(255.0)
    return t.unsqueeze(0)

class SPyNetPkgAdapter:
    def __init__(self, device=None, k: int = 5):
        # Import dal pacchetto funzionante
        try:
            from spynet.model import SpyNet as Net
        except Exception as e:
            raise ImportError("Pacchetto 'spynet-pytorch' non importabile.") from e

        self.device = torch.device(device) if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Istanzia con pesi pre-addestrati (le versioni differiscono leggermente, gestiamo i casi)
        net = None
        # Tentativi robusti
        for attempt in (
            lambda: Net(pretrained=True),
            lambda: Net(k=k),            # crea i livelli e poi carichi tu i pesi se servisse
        ):
            try:
                net = attempt()
                break
            except Exception:
                pass
        if net is None:
            net = Net(k=k)

        self.net = net.to(self.device).eval()

    @torch.no_grad()
    def __call__(self, im0, im1, iters=None):
        t0 = _to_tensor(im0).to(self.device)
        t1 = _to_tensor(im1).to(self.device)

        # Alcune implementazioni richiedono multipli di 32
        H, W = t0.shape[-2], t0.shape[-1]
        Hp = (H + 31)//32*32; Wp = (W + 31)//32*32
        pad = (0, Wp-W, 0, Hp-H)
        t0p = F.pad(t0, pad, mode='replicate')
        t1p = F.pad(t1, pad, mode='replicate')

        # Forward: diverse firme → prova in ordine sicuro
        out = None
        try:
            out = self.net([t0p, t1p], limit_k=-1)
        except Exception:
            try:
                out = self.net([t0p, t1p])
            except Exception:
                try:
                    out = self.net((t0p, t1p))
                except Exception:
                    out = self.net(t0p, t1p)

        # Alcune versioni ritornano lista/tupla
        if isinstance(out, (list, tuple)):
            out = out[0]
        # Taglia padding e converti a (H,W,2) float32
        flow = out[:, :, :H, :W][0].permute(1,2,0).detach().cpu().numpy().astype("float32")
        return flow
