"""Restore a param-decomp TMS run's trained decomposition and read the
single-feature-probe CI matrices (the same probe `identity_ci_error` uses).
Run with param-decomp's venv:
    JAX_PLATFORMS=cpu ~/Projects/param-decomp/.venv/bin/python extract_ci.py RUN_DIR OUT.npz
Saves: ci_<site> [n_features, C] (lower-leaky), V_<site>, U_<site>, target W1/b2, step.
"""
import sys
from pathlib import Path

import jax
import numpy as np
import yaml

from param_decomp.core import placement
from param_decomp.core.checkpoint import make_read_only_checkpoint_manager, restore_decomposition
from param_decomp.core.model import PlacedModel
from param_decomp.core.run_state import init_decomposition
from param_decomp.core.sharding import single_device_mesh
from param_decomp.experiments.tms.config import TMSExperimentConfig
from param_decomp.experiments.tms.run import build_tms_built_run, pretrained_tms_model
from param_decomp.targets import tms

run_dir, out = Path(sys.argv[1]), sys.argv[2]
cfg = TMSExperimentConfig(**yaml.safe_load((run_dir / "launch_config.yaml").read_text()))
built = build_tms_built_run(cfg, run_dir.name, run_dir.parent.parent)
mesh = single_device_mesh()
jax.set_mesh(mesh).__enter__()
model = pretrained_tms_model(built.target, mesh, False)
placed = PlacedModel(model=model, placement=placement.from_config("ddp", mesh, model.sites))

import equinox as eqx
import orbax.checkpoint as ocp
reference = init_decomposition(placed, built.ci_fn, jax.random.PRNGKey(0))  # concrete, placed
abstract = jax.tree.map(lambda x: ocp.utils.to_shape_dtype_struct(x) if eqx.is_array(x) else x,
                        reference)
mgr = make_read_only_checkpoint_manager(run_dir / "ckpts")
step = mgr.latest_step()
dec = restore_decomposition(mgr, step, abstract)

ci_lower = tms.single_feature_ci(model, dec.ci_fn, built.target.n_features)
if isinstance(ci_lower, tuple):          # (lower, upper) in this version
    ci_lower = ci_lower[0]
save = {"step": step, "W1": np.asarray(model.target.W1), "b2": np.asarray(model.target.b2)}
for site, ci in ci_lower.items():
    save[f"ci_{site}"] = np.asarray(ci)
for group, (Vs, Us) in dec.components.stacks.items():
    save[f"V_{group}"], save[f"U_{group}"] = np.asarray(Vs), np.asarray(Us)
np.savez(out, **save)
print("step", step, "sites", list(ci_lower), "groups", list(dec.components.stacks))
