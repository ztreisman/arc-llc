# Arc-Space LLC Estimator: Toy Implementation

original spec, preserved as-is; see RESULTS.md for corrections including to the ground-truth formula

## Goal

Implement and compare two estimators of the RLCT (real log canonical threshold) / local learning
coefficient (LLC) on a small, analytically tractable singular statistical model:

1. **Hessian null-space estimator** (new, geometric): find zero-curvature directions of K at w*,
   measure their dimension, infer λ.
2. **SGLD estimator** (devinterp standard): run stochastic Langevin dynamics at β = 1/log n,
   estimate λ from free energy slope.

The model where we know the ground truth analytically is rank-1 matrix factorization with true
rank 0 (true distribution = zero matrix). This gives:
- d = r(n+m) parameters, RLCT λ = r(n+m-r)/2
- For r=1, n=m=1: d=2, λ=1/2, λ/(d/2) = 1/2
- For r=1, n=m=2: d=4, λ=3/2, λ/(d/2) = 3/4

Ground truth λ values let us validate both estimators before applying either to anything larger.

---

## Mathematical Background

### The model

Parameters: w = (A, B) where A ∈ ℝ^{n×r}, B ∈ ℝ^{r×m}.
Model output: AB (a rank-r matrix).
True distribution: p* corresponds to the zero matrix.

KL divergence (up to constants):
  K(w) = ||AB||²_F = sum_{i,j} (AB)_{ij}²

W₀ = {w : K(w) = 0} = {(A,B) : AB = 0}
   = {A=0} ∪ {B=0}  (for r=1)

This is a singular variety: two subspaces meeting at the origin.

### Ground truth RLCT (known analytically)

For rank-r factorization with true rank 0:
  λ = r(n + m - r) / 2

Derivation via zeta function (r=1, n=m=1):
  ζ(z) = ∫∫ (a²b²)^z da db = 1/(2z+1)²
  Pole at z = -1/2 with multiplicity 2.
  λ = 1/2, multiplicity m = 2.

For r=1, n=m=2, K = ||a||² ||b||² (with a,b ∈ ℝ²):
  ζ(z) ∝ 1/(2z+2)²
  Pole at z = -1.
  λ = 1, d/2 = 2, λ/(d/2) = 1/2.

### Arc-space / Hessian connection

At a true minimum w*:
- The gradient ∇K(w*) = 0
- The Hessian H = ∇²K(w*) has null space = T_{w*}W₀ (tangent space to the solution set)
- dim(null H) is a lower bound on dim W₀

From Mustață's formula, the RLCT is:
  λ = lim_{m→∞} [n(m+1) - dim Contact_m] / (m+1)

where Contact_m = {arcs of order m tangent to W₀}.

The Hessian null space gives the first-order (m=1) contact: directions δ with H·δ = 0 are
directions where K(w* + tδ) = O(t³) or better.

For our model, we can compute higher-order contact explicitly because K is a polynomial.
K(w* + tδ) = K(tδ) = t^4 * K(δ) for the rank-1 case (K is degree 4 and w*=0).
So ALL directions are in W₀ to order 3 at w*=0 — the first nontrivial contact is at order 4.

This means we need a refinement: instead of checking "does K vanish along this direction",
we check the RATE at which it vanishes as a function of direction magnitude.

**Revised estimator**: estimate λ from the volume scaling
  Vol{w : K(w) ≤ ε} ~ ε^λ  as ε → 0
by sampling directions uniformly and fitting the exponent.

---

## Implementation Plan

### Step 1: Define the model and K

```python
import torch
import numpy as np
from scipy.optimize import curve_fit

def make_K(r=1, n=2, m=2):
    """Returns K(w) = ||AB||^2_F as a callable."""
    def K(w):
        # w is a flat vector of length r*(n+m)
        A = w[:r*n].reshape(n, r)
        B = w[r*n:].reshape(r, m)
        AB = A @ B
        return (AB**2).sum()
    return K

def K_grad_hess(w, r=1, n=2, m=2):
    """Gradient and Hessian of K at w, using torch autograd."""
    w_t = torch.tensor(w, dtype=torch.float64, requires_grad=True)
    A = w_t[:r*n].reshape(n, r)
    B = w_t[r*n:].reshape(r, m)
    AB = A @ B
    K_val = (AB**2).sum()
    grad = torch.autograd.grad(K_val, w_t, create_graph=True)[0]
    d = len(w)
    H = torch.zeros(d, d, dtype=torch.float64)
    for i in range(d):
        g2 = torch.autograd.grad(grad[i], w_t, retain_graph=True)[0]
        H[i] = g2
    return K_val.item(), grad.detach().numpy(), H.detach().numpy()
```

### Step 2: Hessian null-space estimator

```python
def hessian_null_space_estimator(w_star, K_fn, eps=1e-6):
    """
    Estimate dim(W₀) via null space of Hessian at w*.
    Returns eigenvalues and estimated null space dimension.
    """
    _, _, H = K_grad_hess(w_star)
    eigenvalues = np.linalg.eigvalsh(H)
    null_dim = np.sum(np.abs(eigenvalues) < eps)
    d = len(w_star)
    lambda_estimate = (d - null_dim) / 2  # rough: codim(W₀)/2
    return {
        'eigenvalues': eigenvalues,
        'null_dim': null_dim,
        'd': d,
        'lambda_lower_bound': lambda_estimate,
        'ratio': lambda_estimate / (d/2)
    }
```

Note: for K = ||AB||² at w*=0, the Hessian is identically zero (K is degree 4, Hessian is
degree 2, which is 0 at 0). So we need to evaluate at w* slightly off zero, or use the volume
method instead. This is an important implementation detail.

### Step 3: Volume scaling estimator (more robust)

```python
def volume_scaling_estimator(K_fn, d, n_samples=50000, eps_values=None):
    """
    Estimate λ from Vol{w : K(w) ≤ ε} ~ C * ε^λ.
    
    Sample w uniformly from a ball, compute K(w), 
    fit log Vol vs log ε.
    """
    if eps_values is None:
        eps_values = np.logspace(-4, 0, 30)
    
    # Sample uniformly from unit ball in R^d
    w_samples = np.random.randn(n_samples, d)
    norms = np.linalg.norm(w_samples, axis=1, keepdims=True)
    radii = np.random.uniform(0, 1, (n_samples, 1))**(1/d)
    w_samples = w_samples / norms * radii
    
    K_vals = np.array([K_fn(w) for w in w_samples])
    
    volumes = np.array([np.mean(K_vals <= eps) for eps in eps_values])
    
    # Fit log(Vol) = λ * log(ε) + const in the small-ε regime
    mask = volumes > 0
    log_eps = np.log(eps_values[mask])
    log_vol = np.log(volumes[mask])
    
    # Linear fit in log-log space
    coeffs = np.polyfit(log_eps, log_vol, 1)
    lambda_estimate = coeffs[0]
    
    return {
        'eps_values': eps_values,
        'volumes': volumes,
        'lambda_estimate': lambda_estimate,
        'ratio': lambda_estimate / (d/2),
        'log_eps': log_eps,
        'log_vol': log_vol
    }
```

### Step 4: SGLD estimator (devinterp-style, from scratch for the toy)

```python
def sgld_llc_estimator(K_fn, d, n_steps=10000, lr=0.01, 
                        beta=None, n_subsample_sizes=None,
                        localization_strength=1.0, w_star=None):
    """
    Estimate λ via SGLD at inverse temperature β = 1/log(n).
    
    F(n) = -1/β * log Z(β) ≈ λ * log(n) + const
    
    Estimate F(n) for several n values and fit slope.
    """
    if w_star is None:
        w_star = np.zeros(d)
    if n_subsample_sizes is None:
        n_subsample_sizes = [100, 200, 500, 1000, 2000, 5000]
    
    free_energies = []
    
    for n in n_subsample_sizes:
        beta_n = 1.0 / np.log(n)
        
        # Run SGLD with localizing prior
        w = w_star.copy() + 0.01 * np.random.randn(d)
        K_trace = []
        
        for step in range(n_steps):
            # Compute gradient of K + localizing prior
            w_t = torch.tensor(w, dtype=torch.float64, requires_grad=True)
            K_val = compute_K_torch(w_t)  # need torch version
            loc_prior = localization_strength * ((w_t - torch.tensor(w_star))**2).sum()
            loss = beta_n * K_val + loc_prior
            loss.backward()
            
            grad = w_t.grad.numpy()
            noise = np.sqrt(2 * lr) * np.random.randn(d)
            w = w - lr * grad + noise
            
            if step > n_steps // 2:  # burn-in
                K_trace.append(K_val.item())
        
        # Free energy estimate: <K> at temperature β
        F_n = np.mean(K_trace)
        free_energies.append(F_n)
    
    # Fit F(n) = λ * log(n) + const
    log_n = np.log(n_subsample_sizes)
    coeffs = np.polyfit(log_n, free_energies, 1)
    lambda_estimate = coeffs[0]
    
    return {
        'n_values': n_subsample_sizes,
        'free_energies': free_energies,
        'lambda_estimate': lambda_estimate,
        'ratio': lambda_estimate / (d/2)
    }
```

### Step 5: Arc-direction detector (your original idea, refined)

```python
def arc_direction_detector(K_fn, w_star, d, order=4, n_directions=1000):
    """
    For each sampled direction δ, compute the Taylor expansion of 
    K(w* + t*δ) in t, and record the leading order of vanishing.
    
    Directions where K vanishes to high order are "arc directions" —
    they point into W₀ or are tangent to it.
    
    For K = ||AB||² at w*=0, K(tδ) = t^4 * K(δ), so all directions
    have vanishing order exactly 4. The interesting quantity is K(δ)
    itself — directions with K(δ) = 0 lie IN W₀.
    
    Returns the distribution of K values on the unit sphere, and
    the volume estimate.
    """
    directions = np.random.randn(n_directions, d)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    
    K_on_sphere = np.array([K_fn(delta) for delta in directions])
    
    # Fraction of directions with K < threshold = proxy for dim W₀ / d
    thresholds = np.logspace(-4, 0, 20)
    fractions = [np.mean(K_on_sphere < thresh) for thresh in thresholds]
    
    return {
        'K_on_sphere': K_on_sphere,
        'thresholds': thresholds,
        'fractions': fractions,
        'directions': directions
    }
```

---

## Experiments to Run

### Experiment 1: Ground truth validation (r=1, n=m=1, d=2)
- True λ = 1/2, d/2 = 1, ratio = 1/2
- Run all three estimators
- Plot volume scaling log-log (should see slope 1/2)
- Compare SGLD free energy slope to 1/2

### Experiment 2: Slightly larger (r=1, n=m=2, d=4)
- True λ = 1, d/2 = 2, ratio = 1/2
- Same pipeline

### Experiment 3: Regular case comparison (λ = d/2)
- Take K(w) = ||w||² (a regular model, Gaussian true distribution)
- All estimators should return λ ≈ d/2, ratio ≈ 1

### Experiment 4: Track ratio across "training"
- Simulate training on the rank-1 model by gradient descent from random init
- At each checkpoint, estimate λ/(d/2) via volume scaling
- Expect: ratio starts near 1 (random weights, approximately regular), 
  decreases toward 1/2 as model converges to singular solution

### Experiment 5: Arc direction distribution
- At w* = 0 (true solution), K on the unit sphere measures how much of 
  parameter space is "flat"
- Compare distribution of K(δ) for random δ to theoretical prediction
- The fraction of directions with K(δ) < ε should scale as ε^(dim W₀ / 4) 
  (since K ~ t^4 on sphere)

---

## Key Implementation Notes

1. **w* = 0 is special**: K is degree 4 and K(0) = 0 exactly. The Hessian 
   at 0 is identically zero. Evaluate the Hessian slightly off zero (e.g. w* 
   after gradient descent with small regularization) to see nontrivial null space.

2. **Volume estimator is most robust**: doesn't require finding a true minimum,
   works directly from the K function. Implement this first.

3. **SGLD needs a localizing prior**: without it, the chain escapes the basin.
   Use a Gaussian centered at w*, strength controls how local the estimate is.

4. **torch.autograd is your friend**: use it for gradient and Hessian-vector 
   products rather than finite differences. 
   `torch.autograd.functional.hessian` gives the full Hessian cleanly.

5. **Numerical precision**: K = (AB)² can be very small near w*=0. Use 
   float64 throughout.

---

## Files to Create

- `model.py` — K function, torch version, gradient/Hessian utilities
- `estimators.py` — volume_scaling, hessian_null_space, sgld, arc_direction
- `experiments.py` — Experiments 1–5 with ground truth comparison
- `plots.py` — log-log volume plots, ratio trajectory, arc direction histograms
- `main.py` — runs all experiments and saves results

---

## Expected Output

For each experiment, a table like:

| Method              | λ estimate | d/2 | λ/(d/2) | True λ/(d/2) |
|---------------------|-----------|-----|---------|--------------|
| Volume scaling      | 0.51      | 1.0 | 0.51    | 0.50         |
| Hessian null space  | 0.50      | 1.0 | 0.50    | 0.50         |
| SGLD                | 0.49      | 1.0 | 0.49    | 0.50         |
| Arc direction dist. | 0.52      | 1.0 | 0.52    | 0.50         |

Plus plots of:
1. log Vol vs log ε (slope = λ)
2. Free energy vs log n (slope = λ)
3. Distribution of K(δ) on unit sphere
4. λ/(d/2) trajectory across training checkpoints

---

## Connection to Larger Goal

Once validated on this toy:
1. Replace K with a GNN loss (your {3,7} embedding network)
2. Use per-sample gradients from backprop as arc-direction candidates
3. Track λ/(d/2) across training checkpoints
4. Look for phase transitions (discontinuous changes in ratio) at the ring-5 barrier

The volume scaling estimator is cheapest to run on a real GNN — you don't 
need SGLD, just random weight perturbations and loss evaluations, which are 
just forward passes.
