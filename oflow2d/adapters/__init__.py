
def make_model(name: str, **kwargs):
    name = (name or "").lower()

    # RAFT -------------------------------------------------------------
    if name in ("raft", "raft_small", "raft_sintel"):
        from .raft import RAFTAdapter
        return RAFTAdapter(**kwargs)

    # SPyNet -----------------------------------------------------------
    if name in ("spynet", "spynet_pkg", "spynet-niklaus", "spy"):
        try:
            from .spynet_adapter import SPyNetAdapter
            return SPyNetAdapter(**kwargs)
        except Exception:
            from .spynet_pkg import SPyNetPkgAdapter
            return SPyNetPkgAdapter(**kwargs)

    # FlowNet2 ---------------------------------------------------------
    if name in ("flownet2", "flownet", "fn2"):
        from .flownet2_adapter import FlowNet2Adapter
        return FlowNet2Adapter(**kwargs)

    # PWC-Net ----------------------------------------------------------
    if name in ("pwcnet", "pwc", "pwc-net"):
        from .pwcnet_adapter import PWCNetAdapter
        return PWCNetAdapter(**kwargs)

    raise ValueError(f"Modello '{name}' non supportato: {name}")
