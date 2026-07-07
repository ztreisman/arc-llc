### Experiment 1 (r=1,n=1,m=1,d=2)

d = 2, true lambda = 0.5000, true ratio lambda/(d/2) = 0.5000

| Method | lambda estimate | d/2 | lambda/(d/2) | True lambda/(d/2) |
|---|---|---|---|---|
| Volume scaling | 0.4583 | 1.00 | 0.4583 | 0.5000 |
| Hessian null space (branch='A': A free, on the {B=0} plane) | 0.5000 | 1.00 | 0.5000 | 0.5000 |
| Hessian null space (branch='B': B free, on the {A=0} plane) | 0.5000 | 1.00 | 0.5000 | 0.5000 |
| SGLD (devinterp-style) | 0.5250 | 1.00 | 0.5250 | 0.5000 |

(Volume scaling diagnostics: naive power-law fit (no log-multiplicity correction) = 0.4381, fitted log-multiplicity exponent k = 0.332, fit R^2 = 0.99979)

### Experiment 2 (r=1,n=2,m=2,d=4)

d = 4, true lambda = 1.0000, true ratio lambda/(d/2) = 0.5000

| Method | lambda estimate | d/2 | lambda/(d/2) | True lambda/(d/2) |
|---|---|---|---|---|
| Volume scaling | 0.9830 | 2.00 | 0.4915 | 0.5000 |
| Hessian null space (branch='A': A free, on the {B=0} plane) | 1.0000 | 2.00 | 0.5000 | 0.5000 |
| Hessian null space (branch='B': B free, on the {A=0} plane) | 1.0000 | 2.00 | 0.5000 | 0.5000 |
| SGLD (devinterp-style) | 1.0863 | 2.00 | 0.5431 | 0.5000 |

(Volume scaling diagnostics: naive power-law fit (no log-multiplicity correction) = 0.8981, fitted log-multiplicity exponent k = 0.741, fit R^2 = 0.99994)

### Experiment 3 (regular model, d=4)

d = 4, true lambda = 2.0000, true ratio lambda/(d/2) = 1.0000

| Method | lambda estimate | d/2 | lambda/(d/2) | True lambda/(d/2) |
|---|---|---|---|---|
| Volume scaling | 1.9263 | 2.00 | 0.9632 | 1.0000 |
| Hessian null space (at origin) | 2.0000 | 2.00 | 1.0000 | 1.0000 |
| SGLD (devinterp-style) | 2.4512 | 2.00 | 1.2256 | 1.0000 |

(Volume scaling diagnostics: naive power-law fit (no log-multiplicity correction) = 1.9819, fitted log-multiplicity exponent k = -0.162, fit R^2 = 0.99988)

### Experiment 6 (asymmetric n!=m, r=1,n=1,m=4,d=5)

d = 5, true lambda = 0.5000, true ratio lambda/(d/2) = 0.2000

| Method | lambda estimate | d/2 | lambda/(d/2) | True lambda/(d/2) |
|---|---|---|---|---|
| Hessian, branch='A' (A free, on {B=0} plane, codim m=4) | 2.0000 | 2.50 | 0.8000 | 0.2000 |
| Hessian, branch='B' (B free, on {A=0} plane, codim n=1) | 0.5000 | 2.50 | 0.2000 | 0.2000 |
| Hessian, multi-restart min-codim (20 restarts) | 0.5000 | 2.50 | 0.2000 | 0.2000 |