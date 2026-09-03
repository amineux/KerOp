# KerOp
A reference implementation of

> Mike Nguyen and Nicole Mücke. **Random Features for Operator-Valued Kernels: Bridging
> Kernel Methods and Neural Operators.** *Proceedings of the 29th International Conference
> on Artificial Intelligence and Statistics (AISTATS)*, PMLR 300:1495–1503, 2026.
> [arXiv:2603.00971](https://arxiv.org/abs/2603.00971)

**This package implements theory developed by those authors. No claim of originality is made
for any of the theorems.** What is offered here is a transparent, tested implementation of
the estimator and the parameter prescriptions, together with a set of reproducible numerical
checks whose outcomes — including the places where a measurement does not match a naive
reading of the theory — are written to `results/` and summarized honestly below.

The implementation is written from scratch in NumPy and SciPy. It does not wrap, vendor, or
depend on any existing random-feature codebase.

---

## What the paper says

The setting is regression from an input space $\mathcal{U}$ into a separable Hilbert space
$\mathcal{V}$ — for operator learning, $\mathcal{U}$ and $\mathcal{V}$ are function spaces
and the target $G_\rho$ is the solution operator of a PDE. Learning uses a *vector-valued*
(equivalently operator-valued) reproducing kernel
$K:\mathcal{U}\times\mathcal{U}\to\mathcal{L}(\mathcal{V})$.

**Assumption 2.1 (integral representation).** The kernel is a mixture of rank-one
operator-valued features,

$$K(u,\tilde u) = \sum_{i=1}^{p}\int_\Omega \varphi_i(u,\omega)\otimes\varphi_i(\tilde u,\omega)\,d\pi(\omega),
\qquad \sum_{i=1}^p\|\varphi_i(u,\omega)\|_\mathcal{V}^2\le\kappa^2 .$$

Drawing $\omega_1,\dots,\omega_M\sim\pi$ gives the random feature kernel $K_M$. The sum over
$i$ is what lets the framework cover operator-valued neural tangent kernels, whose feature
expansion carries one block for the activation and one per coordinate of the derivative
term; this is the origin of the factor $p$ throughout.

**The estimator (2.11).** With $\widehat\Sigma_M$ and $\widehat{\mathcal{S}}^*_M$ the
empirical covariance and adjoint sampling operators of the random feature space,

$$F^M_\lambda = \phi_\lambda(\widehat\Sigma_M)\,\widehat{\mathcal{S}}^*_M\mathbf{v},$$

where $\{\phi_\lambda\}$ is a family of *regularization functions* (Definition 2.2): bounded
in the three senses (2.7)–(2.9), with *qualification* $\nu$ the largest exponent for which
the residual $r_\lambda(t)=1-t\phi_\lambda(t)$ satisfies
$\sup_t|r_\lambda(t)|t^q\le c_q\lambda^q$ for all $q\le\nu$. This one formula covers
Tikhonov regularization, gradient descent with early stopping, and accelerated schemes.

**Assumptions 3.2 and 3.3.** The source condition $G_\rho=\mathcal{L}^rH$ with
$\|H\|_{L^2}\le R$ fixes the smoothness of the target relative to the kernel integral
operator $\mathcal{L}$, with $r=\tfrac12$ the well-specified case $G_\rho\in\mathcal{H}$.
The effective dimension $\mathcal{N}(\lambda)=\operatorname{tr}(\mathcal{L}(\mathcal{L}+\lambda)^{-1})\le c_b\lambda^{-b}$
fixes the capacity, and $2r+b>1$ is required (the "easy learning" regime).

**Theorem 3.4.** If the filter has qualification $\nu\ge r\vee1$ and
$\lambda_n = Cn^{-1/(2r+b)}\log^3(2/\delta)$, then with probability $\ge1-\delta$

$$\bigl\|G_\rho - \mathcal{S}_{M_n}F^{M_n}_{\lambda_n}\bigr\|_{L^2(\rho_\mathcal{U})}
  \;\le\; \bar C\,n^{-\frac{r}{2r+b}}\log^{3r+1}(1/\delta),$$

provided $n\ge n_0=\exp\bigl(\frac{2r+b}{2r+b-1}\bigr)$ and

$$M_n \;\ge\; p\,\tilde C\log n\cdot
\begin{cases}
n^{\frac{1}{2r+b}}, & r\in(0,\tfrac12),\\
n^{\frac{1+b(2r-1)}{2r+b}}, & r\in[\tfrac12,1],\\
n^{\frac{2r}{2r+b}}, & r\in(1,\infty).
\end{cases}$$

So random features attain the same minimax rate as the exact kernel method. In the
well-specified capacity-independent case $r=\tfrac12$, $b=1$ this is $M_n=O(\sqrt n\log n)$
features and $t_n=1/\lambda_n=O(\sqrt n)$ gradient steps; accelerated schemes reach the same
$\lambda_n$ in $O(\sqrt{t_n})$ iterations. Corollary 3.5 transfers this to shallow neural
operators in the NTK regime, where the width must satisfy
$M_n\gtrsim\tilde d^{\,2}B_{T_n}^6(T_n^{2r}\vee T_n)\log^2 n$.

A fuller derivation of how each object maps onto code is in
[`docs/mathematical-summary.md`](docs/mathematical-summary.md).

---

## Install

```bash
python -m pip install -e ".[dev]"
```

Requires Python ≥ 3.10, NumPy and SciPy. Everything here runs on a laptop in minutes; there
is no GPU code and no deep learning framework.

## Quick start

Learn the Darcy solution operator with operator-valued NTK random features and Brakhage's
$\nu$-method:

```bash
kerop demo darcy --n-train 800 --n-points 33 --filter nu_method --nu 2
```

```text
KerOp demo: the darcy solution operator
--------------------------------------------------------------
  grid points n_x (= d_v)        33
  feature dimension d_tilde      8
  summands p = 1 + d_tilde       9
  training pairs n               800
  random features M              255
  coefficient dimension pM       2295
  filter                         NuMethod(lam=1.00169e-05, iterations=316, step=1, nu=2)

  excess risk ||G_rho - S_M F||  0.000109
  relative to ||G_rho||          0.137%
```

Other entry points:

```bash
kerop demo poisson --filter landweber      # gradient descent with early stopping
kerop filters                              # measured constants and qualification of each family
kerop theory --r 0.5 --b 1.0 --n 10000     # what Theorem 3.4 prescribes
```

In Python:

```python
import numpy as np
from kerop.data.pde import PoissonDataset
from kerop.estimators import VectorValuedRFRegressor
from kerop.features import OperatorNTKFeatures
from kerop.filters import NuMethod
from kerop.metrics import excess_risk

dataset = PoissonDataset(n_points=33)
rng = np.random.default_rng(0)
train, test = dataset.sample(800, rng), dataset.sample(400, rng)

features = OperatorNTKFeatures(
    dataset.feature_dim, dataset.n_points, n_features=256, rng=rng,
    output_scale=dataset.output_scale(),
)
estimator = VectorValuedRFRegressor(
    features, filter_obj=NuMethod.from_iterations(200, nu=2.0)
).fit(dataset.lift(train.fields), train.outputs)

print(excess_risk(estimator.predict(dataset.lift(test.fields)), test.targets))
```

---

## Reproducing the numbers

```bash
python scripts/run_all.py            # everything, a few minutes
python scripts/run_all.py --quick    # smoke test, well under a minute
```

Each script writes a JSON record (settings, provenance, every measurement) and flat CSVs to
`results/`, and `scripts/collect_summary.py` distills them into `results/summary.json` and
`results/summary.md`, stating each falsifiable claim and whether the measurement supports
it. Individual experiments:

| Script | Question |
| --- | --- |
| `run_filter_diagnostics.py` | Do the filters satisfy Definition 2.2, and what is their qualification? |
| `run_rate_experiment.py` | Does the excess risk decay at the exponent $r/(2r+b)$? |
| `run_feature_threshold.py` | Does $M\sim\sqrt n\,p$ suffice? (the Figure 1 / Appendix A.3 setup) |
| `run_walltime_benchmark.py` | Is it faster than exact operator-valued kernel regression? |

See [`docs/reproducing.md`](docs/reproducing.md) for the experimental protocol, including
what is held fixed and why.

---

## Results

<!-- RESULTS-START -->
Run `python scripts/run_all.py` to populate this section; the committed `results/` directory
holds the output of the run described in `results/summary.json`.
<!-- RESULTS-END -->

---

## What is implemented

| Module | Contents |
| --- | --- |
| `kerop.filters` | Tikhonov, iterated Tikhonov, Landweber (gradient descent), spectral cut-off, heavy-ball, Brakhage's $\nu$-method. Exact cancellation-free residuals, and a numerical probe of the Definition 2.2 constants and the qualification. |
| `kerop.features` | Vector-valued random feature maps satisfying Assumption 2.1: importance-sampled Mercer features, the scalar and operator-valued NTK features of a shallow network and neural operator, and separable random Fourier features. |
| `kerop.kernels` | The exact operator-valued kernels those maps approximate, with the NTK limits in closed form via the arc-cosine kernels. |
| `kerop.estimators` | The random feature estimator (2.11), with both the $O(nM^2)$ normal-equation route and the $O(nMt)$ matrix-free route, and the exact operator-valued baseline running the same filters on the block Gram matrix. |
| `kerop.theory` | Theorem 3.4 and Corollary 3.5 as executable prescriptions: $\lambda_n$, $M_n$, iteration counts, $n_0$, qualification requirements. |
| `kerop.data` | A synthetic instance with prescribed $(r,b)$, and the Poisson and Darcy solution operators. |
| `kerop.experiments` | The four experiment drivers. |

The two estimators share the filter implementations verbatim, so a comparison between them
isolates the effect of the random feature approximation and nothing else.

---

## Honest notes and gaps

This is a reference implementation of published theory, not a validation of it, and several
things deserve to be stated plainly.

**What the rate experiment can and cannot show.** Theorem 3.4 is an asymptotic upper bound
with unspecified constants. Over a bounded range of $n$ those constants leave residual
curvature in the log-log plot, so a finite-sample experiment can confirm the *order* of the
rate, not the exponent to several digits. The agreement reported is therefore relative, with
the per-decade local slopes recorded alongside so the trend is visible.

**$b=1$ is avoided.** At exactly $b=1$ the eigenvalues decay as $i^{-1}$, the integral
operator is not trace class, and $\mathcal{N}(\lambda)\asymp\lambda^{-1}\log(1/\lambda)$
rather than $\lambda^{-1}$. There is then no clean power law for the measured exponent to
agree with. The synthetic configurations use $b\le0.7$; measured against the nominal value,
agreement is excellent for $b\le0.5$ and degrades as $b\to1$, which is a property of the
construction rather than of the estimator.

**The synthetic instance has finite rank.** The Mercer expansion is truncated at $J$ modes,
so $\mathcal{N}(\lambda)$ flattens once $\lambda$ falls below the smallest eigenvalue. Every
experiment checks that its $\lambda_n$ range lies inside the resulting power-law window, and
`SpectralOperatorModel.usable_lambda_window` makes that window explicit.

**The regularization constant is calibrated.** Theorem 3.4 fixes the *exponent* of
$\lambda_n$ and leaves the constant $C$ free. An arbitrary $C$ leaves the estimator
uniformly over- or under-regularized, which biases the finite-sample slope, so $C$ is chosen
once by minimizing the risk at a single reference sample size and then held fixed across
every $n$, filter and repeat. This is a test of the theorem's claim that the optimal
$\lambda$ scales as $n^{-1/(2r+b)}$, not an exploitation of it. The same applies to the
unspecified constant $\tilde C$ in $M_n$, recorded per configuration.

**At matched sample size, exact kernel regression is more accurate.** Theorem 3.4 preserves
the *rate*, not the constant: the random feature approximation contributes an error of order
$M^{-1/2}$ that regularization does not remove. The wall-clock benchmark reports both the
matched-sample-size comparison, which the exact method wins on accuracy, and the
matched-risk comparison, which is the one the scalability claim is about.

**SUSY is replaced.** Appendix A.3 uses a Gaussian design at $d=1$ and a subset of the SUSY
dataset at $d=14$. SUSY is not retrievable in this environment, so the $d=14$ design is
Gaussian as well. The claim under test concerns the scaling in $d$ through $p=d+2$, which is
preserved. Sample sizes at $d=14$ are also capped below the paper's $n=5000$, because the
plateau reference needs $M=4\sqrt n\,p$ and hence a design matrix with $pM$ columns.

**Not implemented.** The following are in the paper's scope but not here.

- *Actual neural operator training.* Corollary 3.5 concerns a shallow neural operator trained
  by gradient descent, whose excess risk decomposes into a finite-width term and the random
  feature term. Only the second is implemented; the first,
  $\|G_{\theta_t}-F^M_t\|=O(\log n/M_n)$, is quoted from Nguyen & Mücke (2024) and not
  measured. `kerop.theory.neural_operator_width` encodes the prescription but nothing trains
  a network. This is the largest gap.
- *Second-stage sampling.* Appendix A.2 requires $n_\mathcal{X}$ collocation points drawn
  i.i.d. from $\mu$ with $n_\mathcal{X}\gtrsim B_T^2T^{2r}\log^2T$. Here the collocation grid
  is fixed and uniform, and its effect on the rate is not studied.
- *The confidence structure.* The $\log^3(2/\delta)$ and $\log^{3r+1}(1/\delta)$ factors are
  implemented in `kerop.theory` but the experiments report means over repeats rather than
  high-probability bounds, so the $\delta$-dependence is untested.
- *Misspecified regime below $2r+b\le1$.* Excluded by assumption, and rejected with an
  explicit error by `kerop.theory.check_assumptions`.
- *Two-dimensional PDEs.* The Poisson and Darcy operators are one-dimensional, chosen so the
  solution operator is available to machine precision and the exact $nd_v\times nd_v$ Gram
  matrix stays within laptop memory.

**Heavy-ball is a partial case.** With momentum $\beta$ held fixed, the residual constants
flatten onto the Landweber values as $\lambda\to0$, so the qualification is unbounded and
the method is asymptotically gradient descent with step $\alpha/(1-\beta)$ rather than an
accelerated scheme; the $T^{-2}$ acceleration requires $\beta\to1$. The genuinely
accelerated method here is the $\nu$-method, whose qualification is measured to be exactly
$\nu$ and which reaches a given $\lambda$ in $O(\sqrt t)$ iterations. Because the
$\lambda\leftrightarrow T$ mapping for heavy-ball is only asymptotically meaningful, its
measured bias exponents are contaminated at large $\lambda$, where the iteration count is
small; those entries are flagged rather than silently included.

---

## Development

```bash
python -m pytest tests/ -q     # ~170 tests, about 30 seconds
ruff check src tests scripts
ruff format --check src tests scripts
mypy
```

CI runs the tests on Python 3.10–3.12, the linter, the type checker, and the experiment
suite in quick mode, uploading its output as an artifact.

## Citation

If you use this code, please cite the paper it implements:

```bibtex
@inproceedings{nguyen2026random,
  title     = {Random Features for Operator-Valued Kernels:
               Bridging Kernel Methods and Neural Operators},
  author    = {Nguyen, Mike and M{\"u}cke, Nicole},
  booktitle = {Proceedings of the 29th International Conference on Artificial
               Intelligence and Statistics},
  series    = {Proceedings of Machine Learning Research},
  volume    = {300},
  pages     = {1495--1503},
  year      = {2026},
  publisher = {PMLR},
  eprint    = {2603.00971},
  archivePrefix = {arXiv},
  primaryClass  = {stat.ML}
}
```

Background references for the machinery used here: Caponnetto & De Vito (2007) and Blanchard
& Mücke (2017) for spectral regularization rates; Gerfo et al. (2008) and Engl, Hanke &
Neubauer (1996, ch. 6) for the filter families and their qualifications; Rudi & Rosasco
(2016) and Lanthaler & Nelsen (2023) for random features in the scalar and vector-valued
kernel ridge regression cases; Pagliana & Rosasco (2019) for accelerated methods; Nguyen &
Mücke (2024) for the neural operator rates that Corollary 3.5 builds on; Cho & Saul (2009)
for the arc-cosine kernels used in the closed-form NTK.

## License

MIT, see [`LICENSE`](LICENSE).
