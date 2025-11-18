
import torch, numpy as np

# usa il context manager del notebook per caricare PTLFlow dal site isolato
try:
    from __main__ import use_flow_site
except Exception:
    import sys
    FLOW_SITE = "/content/_flow_site"
    def use_flow_site():
        class _Ctx:
            def __enter__(self):
                sys.path.insert(0, FLOW_SITE)
            def __exit__(self, *a):
                if FLOW_SITE in sys.path:
                    sys.path.remove(FLOW_SITE)
        return _Ctx()

class PWCNetAdapter:
    def __init__(self, device=None, ckpt="things"):
        self.device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        with use_flow_site():
            import ptlflow
            last_err = None
            # prova alcuni nomi comuni
            for name in ("pwcnet", "pwc"):
                try:
                    self.model = ptlflow.get_model(name, ckpt_path=ckpt).to(self.device).eval()
                    self._model_name = name
                    break
                except Exception as e:
                    last_err = e
            if not hasattr(self, "model"):
                raise RuntimeError(
                    f"PWCNet non trovato in PTLFlow: {type(last_err).__name__}: {last_err}"
                )

            from ptlflow.utils.io_adapter import IOAdapter
            self.IOAdapter = IOAdapter

        self._io = None

    @torch.inference_mode()
    def __call__(self, img0_uint8, img1_uint8, iters=None):
        """img0_uint8, img1_uint8: HxWx3 uint8 (RGB o BGR, ma coerenti).
        Ritorna: flow HxWx2 float32.
        """
        assert img0_uint8.dtype == np.uint8 and img1_uint8.dtype == np.uint8, \
            "Servono frame uint8"

        h, w = img0_uint8.shape[:2]

        if self._io is None:
            with use_flow_site():
                self._io = self.IOAdapter(self.model, (h, w))

        with use_flow_site():
            x = self._io.prepare_inputs([img0_uint8, img1_uint8])  # {'images': (1,2,3,H,W)}
            for k in x:
                x[k] = x[k].to(self.device, non_blocking=True)
            preds = self.model(x)

        flows = preds.get("flows", preds.get("flow", None))
        if flows is None:
            raise RuntimeError("PTLFlow/PWCNet: output 'flows'/'flow' non trovato.")

        ft = flows[-1] if isinstance(flows, (list, tuple)) else flows
        if ft.dim() == 5:   # (B,1,2,H,W) -> (B,2,H,W)
            ft = ft[:, 0]

        flow = ft[0].permute(1, 2, 0).detach().cpu().float().numpy()  # HxWx2
        return flow
