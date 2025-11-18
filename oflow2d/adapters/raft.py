
import os, sys, torch, numpy as np

class _DotArgs(dict):
    __getattr__ = dict.get
    def __setattr__(self, k, v): self[k] = v
    def __contains__(self, k): return dict.__contains__(self, k)

class RAFTAdapter:
    def __init__(self, weights:str, device:str=None,
                 small:bool=True, mixed_precision:bool=True, alternate_corr:bool=False):
        raft_root = os.environ.get("RAFT_ROOT", "/content/RAFT")
        core_path = os.path.join(raft_root, "core")
        for p in (core_path, raft_root):
            if p not in sys.path: sys.path.insert(0, p)

        try:
            from core.raft import RAFT as RAFTCore
            from core.utils.utils import InputPadder
        except Exception as e:
            raise ImportError("RAFT non trovato. Aggiungi /content/RAFT e /content/RAFT/core a sys.path.") from e

        self.device = torch.device(device) if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.args = _DotArgs(small=small, mixed_precision=mixed_precision, alternate_corr=alternate_corr)
        self.net = RAFTCore(self.args)

        state = torch.load(weights, map_location='cpu')
        if isinstance(state, dict) and 'state_dict' in state and isinstance(state['state_dict'], dict):
            state = state['state_dict']
        if isinstance(state, dict):
            state = { (k.split('module.',1)[-1]): v for k,v in state.items() }

        missing, unexpected = self.net.load_state_dict(state, strict=False)
        if missing:    print("[RAFTAdapter] missing:", len(missing))
        if unexpected: print("[RAFTAdapter] unexpected:", len(unexpected))

        self.net.to(self.device).eval()

    @torch.no_grad()
    def __call__(self, im0, im1, iters:int=8):
        from core.utils.utils import InputPadder
        t0 = torch.from_numpy(im0.transpose(2,0,1)).float().unsqueeze(0)/255.0
        t1 = torch.from_numpy(im1.transpose(2,0,1)).float().unsqueeze(0)/255.0
        t0 = t0.to(self.device); t1 = t1.to(self.device)
        padder = InputPadder(t0.shape)
        t0p, t1p = padder.pad(t0, t1)
        _, flow_up = self.net(t0p, t1p, iters=iters, test_mode=True)
        flow = padder.unpad(flow_up)[0].permute(1,2,0).detach().cpu().numpy().astype('float32')
        return flow
