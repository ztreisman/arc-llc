"""Run param-decomp's VPD (JAX trainer, unmodified) on a two-layer linear
reduced-rank-regression target y = W2 W1 x, then score the decomposition.

Run with param-decomp's venv (see PREREGISTRATION.md for the protocol):
    JAX_PLATFORMS=cpu ~/Projects/param-decomp/.venv/bin/python run_rrr_spd.py \
        H R0 COEFF_FACTOR RUN_ID OUT.npz

The TMS target module is reused by patching, not editing:
  - `tms.site_dims`: linear1 is M -> H, linear2 is H -> N (TMS ties both to n_features);
  - `tms.jax`: a proxy whose `nn.relu` is the identity, so the TMS forward / masked
    forward become linear (the bias b2 is zero).
The target weights are the exact minimum-norm solution of W2 W1 = M* (rank r0).
"""
import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import yaml
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from param_decomp.core import placement
from param_decomp.core.checkpoint import make_read_only_checkpoint_manager, restore_decomposition
from param_decomp.core.components import SiteDims
from param_decomp.core.log import setup_logger
from param_decomp.core.model import PlacedModel, Positionless
from param_decomp.core.run import MetricsSink, install_sigterm_flag, run_decomposition_training
from param_decomp.core.run_state import init_decomposition
from param_decomp.core.sharding import single_device_mesh
from param_decomp.experiments.config import pin_launch_config
from param_decomp.experiments.tms.config import TMSExperimentConfig
from param_decomp.experiments.tms.run import build_tms_built_run
from param_decomp.targets import tms

M, N = 10, 5
H, R0 = int(sys.argv[1]), int(sys.argv[2])
COEFF_FACTOR = float(sys.argv[3])
RUN_ID, OUT = sys.argv[4], sys.argv[5]
DATA_ROOT = Path.home() / "rrr-spd" / "data"
BASE_CONFIG = Path.home() / "tms-5-2-run" / "tms_5-2_local.yaml"
C_PER_SITE = 20
N_PROBE = 4096


# ---------------------------------------------------------------- patches
def _rrr_site_dims(cfg, name):
    if name == tms.LINEAR1:
        return SiteDims(d_in=M, d_out=H)
    if name == tms.LINEAR2:
        return SiteDims(d_in=H, d_out=N)
    raise AssertionError(f"unexpected site {name!r}")


class _LinearNN:
    relu = staticmethod(lambda x: x)

    def __getattr__(self, k):
        return getattr(jax.nn, k)


class _JaxProxy:
    nn = _LinearNN()

    def __getattr__(self, k):
        return getattr(jax, k)


tms.site_dims = _rrr_site_dims
tms.jax = _JaxProxy()


# ---------------------------------------------------------------- config
raw = yaml.safe_load(BASE_CONFIG.read_text())
raw["run_name"] = f"rrr-spd-H{H}-r{R0}-x{COEFF_FACTOR:.4g}"
raw["target"]["n_features"] = M      # only feeds the (patched) site dims / CI arch
raw["target"]["n_hidden"] = H
for s in raw["decomposition"]["sites"]["sites"]:
    s["C"] = C_PER_SITE
[im] = [m for m in raw["pd"]["loss_metrics"] if m["type"] == "ImportanceMinimalityLoss"]
assert (im["coeff"], im["frequency"]["coeff"]) == (8e-4, 4e-4)
im["coeff"] *= COEFF_FACTOR
im["frequency"]["coeff"] *= COEFF_FACTOR
raw["eval"] = None                   # TMS's eval block assumes the single-feature probe
import os
if os.environ.get("SMOKE_STEPS"):    # engineering smoke tests only; never used for results
    raw["pd"]["steps"] = int(os.environ["SMOKE_STEPS"])
    raw["cadence"]["checkpointing"]["save_every"] = int(os.environ["SMOKE_STEPS"])
cfg = TMSExperimentConfig(**raw)
built = build_tms_built_run(cfg, RUN_ID, DATA_ROOT)
built.run.run_dir.mkdir(parents=True, exist_ok=True)
setup_logger(built.run.run_dir / "logs.log")
pin_launch_config(built.run.run_dir, yaml.safe_dump(raw, sort_keys=False))
install_sigterm_flag()

mesh = single_device_mesh()
jax.set_mesh(mesh).__enter__()

# ---------------------------------------------------------------- target
W1 = np.zeros((H, M), np.float32)
W2 = np.zeros((N, H), np.float32)
for i in range(R0):
    W1[i, i] = 1.0
    W2[i, i] = 1.0
target = tms.TMSTarget(W1=jnp.asarray(W1), hidden=(), W2=jnp.asarray(W2),
                       b2=jnp.zeros((N,), jnp.float32))
tms_cfg = tms.TMSConfig(n_features=M, n_hidden=H)
model = tms.replicate_target(
    tms.tms_decomposed_model(tms_cfg, target, tms.site_specs(tms_cfg, built.target.sites)), mesh)
placed = PlacedModel(model=model, placement=placement.from_config("ddp", mesh, model.sites))

data_key = jax.random.fold_in(jax.random.PRNGKey(built.pd.seed), 17)


@jax.jit
def _sample(key):
    x = jax.random.normal(key, (built.target.global_batch, M))
    return jax.sharding.reshard(x, NamedSharding(mesh, P(placement.batch_axes(mesh))))


run_decomposition_training(
    pd=built.pd, cadence=built.cadence, run=built.run, model=placed, ci_fn=built.ci_fn,
    positions=Positionless(), remat_recon_forwards=False, remat_ci_fn=False,
    compiler_options={}, sample_batch=lambda step: _sample(jax.random.fold_in(data_key, step)),
    evaluation=None, sink=MetricsSink.for_run(built.run, True), profiling=None,
)

# ---------------------------------------------------------------- score
reference = init_decomposition(placed, built.ci_fn, jax.random.PRNGKey(0))
abstract = jax.tree.map(lambda a: ocp.utils.to_shape_dtype_struct(a) if eqx.is_array(a) else a,
                        reference)
mgr = make_read_only_checkpoint_manager(built.run.run_dir / "ckpts")
step = mgr.latest_step()
dec = restore_decomposition(mgr, step, abstract)

probe = jax.random.normal(jax.random.PRNGKey(12345), (N_PROBE, M))
ci = dec.ci_fn(model.clean_forward(probe, dec.ci_fn.capture_keys, placement=None).captures,
               remat=False, placement=None).lower
ci = {s: np.asarray(tms.require_full_emission(v)) for s, v in ci.items()}       # [N_PROBE, C]
alive = {s: np.where((v > 0.5).mean(0) >= 0.5)[0] for s, v in ci.items()}

deltas = tms.weight_deltas_fp32(target, dec.components)
rel_err = {}
for s, Wt in (("linear1", W1), ("linear2", W2)):
    nrm = np.linalg.norm(Wt)
    rel_err[s] = float(np.linalg.norm(np.asarray(deltas[s])[0]) / nrm) if nrm > 0 else float("nan")

V1, U1 = (np.asarray(a)[0] for a in dec.components.stacks["linear1"])   # V [M, C], U [C, H]
V2, U2 = (np.asarray(a)[0] for a in dec.components.stacks["linear2"])   # V [H, C], U [C, N]
x = np.asarray(probe)
y = x @ W1.T @ W2.T
a1, a2 = alive["linear1"], alive["linear2"]
y_hat = x @ (V1[:, a1] @ U1[a1]) @ (V2[:, a2] @ U2[a2])
recon_rel = float(((y_hat - y) ** 2).mean() / max((y ** 2).mean(), 1e-12)) if R0 > 0 else float(
    (y_hat ** 2).mean())

np.savez(OUT, H=H, r0=R0, coeff_factor=COEFF_FACTOR, step=step,
         alive1=a1, alive2=a2, ci1=ci["linear1"].mean(0), ci2=ci["linear2"].mean(0),
         ci_frac1=(ci["linear1"] > 0.5).mean(0), ci_frac2=(ci["linear2"] > 0.5).mean(0),
         rel_err1=rel_err["linear1"], rel_err2=rel_err["linear2"], recon_rel=recon_rel,
         V1=V1, U1=U1, V2=V2, U2=U2)
print(f"H={H} r0={R0} x{COEFF_FACTOR:.4g} step={step}: alive {len(a1)}/{len(a2)}, "
      f"rel_err {rel_err['linear1']:.3g}/{rel_err['linear2']:.3g}, recon_rel {recon_rel:.3g}")
