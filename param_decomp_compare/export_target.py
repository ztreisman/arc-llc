"""Regenerate param-decomp's TMS 5->2 target (deterministic in seed) and save W, b.
Run with param-decomp's venv:
    JAX_PLATFORMS=cpu ~/Projects/param-decomp/.venv/bin/python param_decomp_compare/export_target.py
Writes param_decomp_compare/target_5-2.npz (input to check_target.py)."""
import numpy as np
from param_decomp.targets import tms

cfg = tms.TMSConfig(n_features=5, n_hidden=2, n_hidden_layers=0)
t = tms.pretrain_tms_target(cfg, 0.05, "at_least_zero_active", steps=5000, batch_size=2048,
                            lr=1e-2, seed=0)
np.savez("param_decomp_compare/target_5-2.npz", W=np.asarray(t.W1), b=np.asarray(t.b2))
