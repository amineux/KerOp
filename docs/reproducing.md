# Reproducing the numbers

```bash
python -m pip install -e ".[dev,plots]"
python scripts/run_all.py            # everything
python scripts/run_all.py --quick    # smoke test
```

Every script writes a JSON record and flat CSVs to `results/`. The JSON holds the settings,
the environment provenance and every individual measurement; the CSVs hold the tabular rows
a plotting script would want. `scripts/collect_summary.py` reads them back and writes
`results/summary.json` and `results/summary.md`, stating each falsifiable claim and whether
the measurement supports it.

The JSON provenance block records the git commit, the Python and NumPy versions, the
platform, the CPU count and the BLAS build. Wall-clock numbers are only interpretable
alongside the machine that produced them.

---

## 1. Filter diagnostics — `run_filter_diagnostics.py`

**Question.** Do the implemented families satisfy Definition 2.2, what is their
qualification, and does qualification matter?

**Protocol.** For each family, the suprema in (2.7)–(2.9) are evaluated on a grid over
$t\in(0,1]$ refined near zero and over a grid of $\lambda$, giving measured $D$, $E$, $c_0$.
The qualification is probed by evaluating

$$c_q(\lambda)=\sup_{0<t\le1}|r_\lambda(t)|\,t^q\lambda^{-q}$$

for $q\in\{0.5,1,\dots,4\}$ over $\lambda\in[10^{-7},10^{-3}]$ and regressing
$\log c_q$ on $\log\lambda$. Condition (2.10) asks for $c_q$ to be bounded uniformly in
$\lambda$, so an exponent is admissible when the slope is flat; a slope of $\nu-q$ signals
saturation at qualification $\nu$. The probe is capped at $q=4$, so families with unbounded
qualification report `>= 4`.

The second half measures the *consequence*: the exact bias
$\|r_\lambda(\mathcal{L})G_\rho\|$ of the synthetic instance, whose decay exponent should be
$\min(r,\nu)$. This is computed from the known spectrum with no sampling, which is what makes
it a clean isolation of the filter's contribution.

**Caveat.** For heavy-ball the map $\lambda=(1-\beta)/(\alpha T)$ is only asymptotically
meaningful; at the top of the $\lambda$ window the realized iteration count is single-digit
and the momentum transient has not decayed, so its measured bias exponent overshoots. Those
entries are flagged in the output rather than silently included in the verdict.

## 2. The rate of Theorem 3.4 — `run_rate_experiment.py`

**Question.** Does the excess risk decay as $n^{-r/(2r+b)}$?

**Protocol.** For each configuration:

1. Build `SpectralOperatorModel` with the nominal $(r,b)$.
2. Calibrate the single constant $C$ in $\lambda_n=Cn^{-1/(2r+b)}$ — see below.
3. **Measure** $r$ and $b$ on exactly the resulting range of $\lambda$, from the exact
   spectrum (`effective_dimension_fit`) and the exact bias (`source_exponent_fit`). No
   sampling is involved, so these are properties of the instance, and the rate is compared
   against the exponent they imply as well as against the nominal one.
4. For each filter, sample size and repeat, draw fresh data and fresh features, fit
   $F^{M_n}_{\lambda_n}$ with $M_n$ from Theorem 3.4, and evaluate the excess risk against
   $G_\rho$ on a fixed noiseless test set of 4000 points.
5. Average the log risk over repeats, fit $\log(\text{risk})$ against $\log n$, and compare
   the slope with $-r/(2r+b)$.

**What is held fixed and why.** Theorem 3.4 fixes the *exponent* of $\lambda_n$ and leaves
the constant $C$ free, requiring only that it not depend on $n$. That freedom matters in a
finite-sample experiment: an arbitrary $C$ leaves the estimator uniformly over- or
under-regularized, which biases the measured slope even though it does not affect the
asymptotic exponent. $C$ is therefore chosen once, by minimizing the excess risk at the
largest sample size, and then held fixed across every $n$, filter and repeat. This is
legitimate precisely because the theorem asserts the optimal $\lambda$ scales as
$n^{-1/(2r+b)}$, so the best constant is $n$-independent; calibrating at one sample size and
extrapolating with the prescribed exponent tests that claim rather than exploiting it. The
candidates are further restricted to keep every $\lambda_n$ inside the instance's power-law
window, and the full calibration trace is recorded in the JSON.

The same applies to $\tilde C$ in $M_n$, which the theorem also leaves unspecified. It is
recorded per configuration (`feature_constant`) and is set below one for the two
configurations whose feature exponent is steep enough that $pM$ would otherwise dominate the
runtime.

**How agreement is judged.** Theorem 3.4 is an asymptotic upper bound with unspecified
constants, so over a bounded range of $n$ the residual curvature in the log-log plot is
real, and a confidence interval computed from the fit residuals is too tight to be a
meaningful test — it measures how well the data follow *a* power law, not how close the
exponent is to the asymptotic one. Agreement is therefore assessed as a relative error,
$|{\rm slope}+{\rm exponent}|/{\rm exponent}\le0.15$, with the per-decade local slopes and
the tail slope over the upper half of the range reported alongside so the trend is visible.
What this experiment can establish is the *order* of the rate, not the exponent to several
digits.

**Configurations.**

| Name | $r$ | $b$ | Regime | Filters |
| --- | --- | --- | --- | --- |
| `well-specified` | 0.5 | 0.5 | $G_\rho\in\mathcal{H}$ | Tikhonov, Landweber, $\nu$-method, heavy-ball |
| `misspecified` | 0.3 | 0.7 | $G_\rho\notin\mathcal{H}$, $2r+b=1.3$ | Tikhonov, Landweber, $\nu$-method |
| `smooth` | 1.0 | 0.5 | extra smoothness | Tikhonov, Landweber, $\nu$-method |
| `beyond-tikhonov-qualification` | 1.5 | 0.5 | $r>1$ | Tikhonov (**negative control**), Landweber, $\nu$-method, iterated Tikhonov |

The last configuration is the point of including Tikhonov: with qualification $1<r=1.5$ it
violates the hypothesis $\nu\ge r\vee1$, so Theorem 3.4 says nothing about it. It is
reported with `qualification_ok = false` and excluded from the verdict.

$b=1$ is avoided throughout. At exactly $b=1$ the eigenvalues decay as $i^{-1}$, the
operator is not trace class, and $\mathcal{N}(\lambda)\asymp\lambda^{-1}\log(1/\lambda)$
rather than $\lambda^{-1}$, so there is no clean power law for the measured exponent to
agree with.

## 3. The feature threshold — `run_feature_threshold.py`

**Question.** Does $M$ of order $\sqrt n\,p$ suffice, with $p=d+2$?

**Protocol.** This recreates the setup behind Figure 1 in Appendix A.3: kernel gradient
descent on the real-valued NTK, with the test error mapped over the number of random
features $M$ and the number of iterations $T$.

The target is drawn from the NTK RKHS itself, with unit RKHS norm, so the problem is
well-specified ($r=1/2$, the regime the $\sqrt n$ threshold refers to). It is then rescaled
to unit $L^2$ norm, which is not cosmetic: a unit-RKHS-norm coefficient vector spread over
$pM$ directions has $\|f^*\|^2_{L^2}=c^\top\Sigma c\approx\operatorname{tr}(\Sigma)/(pM)$,
which for $pM\sim10^4$ is of order $10^{-2}$, so a nominally modest noise level would drown
the signal and the measured error would be flat in $M$ for entirely the wrong reason.

$M$ is expressed in multiples of $\sqrt n\,p$; the largest multiplier on the grid serves as
the plateau reference. The paper's claim is one of *sufficiency* — "once $M$ exceeds a
threshold of order $O(\sqrt n\,p)$ [...] further increasing $M$ does not lead to any
improvement" — so the test is whether the error at multiplier one is within 5% of the
plateau. Sweeping $T$ costs no more than its largest value, since every iterate along one
gradient descent trajectory is the estimator with $\lambda=1/(\alpha T)$.

The *location* of the threshold is also reported, as the smallest $M$ within 5% of the
plateau, interpolated in log-log so the value is not quantized to the grid, together with
fits of how it scales in $n$ and the ratio between input dimensions at matched $n$. That is a
finer question than the paper's claim and the answer is grid- and range-limited, so it is
offered as a measurement rather than a test.

**Deviations from the paper.** Appendix A.3 uses $n=5000$ with a Gaussian design at $d=1$ and
a subset of SUSY at $d=14$, averaged over 50 runs. SUSY is not retrievable in this
environment, so the $d=14$ design is Gaussian as well; the claim under test concerns the
scaling in $d$ through $p=d+2$, which is preserved. Sample sizes at $d=14$ are capped at
2500 because the plateau reference needs $M=4\sqrt n\,p$, i.e. a design matrix with $pM$
columns, and $n=5000$ would need one of several gigabytes. Repeats are 4 rather than 50.

`--plot` writes a heat map, the visual analogue of the paper's figure, with the column at
$M=\sqrt n\,p$ marked.

## 4. Wall-clock against exact kernel regression — `run_walltime_benchmark.py`

**Question.** Is the random feature estimator faster than exact operator-valued kernel
regression?

Both estimators use the *same* kernel — the exact one in closed form for the baseline, its
random feature approximation for the other — and the same filter implementations, so the
comparison isolates the effect of the approximation.

**Two comparisons, because they answer different questions.**

*Matched sample size.* At fixed $n$, which method reaches the lower excess risk? The exact
method does, and by a margin that closes only slowly in $M$: the approximation contributes an
error of order $M^{-1/2}$ that regularization does not remove. Theorem 3.4 preserves the
rate, not the constant, so this is the expected outcome and is reported plainly.

*Matched excess risk.* Which method reaches a given error level in less wall-clock time, each
free to choose its own sample size and hyper-parameters? This is the question that matters in
practice and the one the paper's scalability claim addresses, since the random feature method
can afford far more data at equal cost. Both methods are swept over the same grid of sample
sizes and their own hyper-parameters, a cost/accuracy frontier is built for each, and the
speed-up is read at several target risk levels drawn from the range both can reach. Matching
on risk first and comparing time second means the comparison cannot be won by under-fitting.

**Fairness.** The reported per-fit time for the exact method includes assembling the Gram
matrix, which is what a single fit costs. The frontier uses single-fit times for both
methods, which is the conservative choice: a practitioner tuning $\lambda$ would reuse the
factorization, and that possibility is not charged against the baseline.

**Tasks.** `--task spectral` is the synthetic instance, where $p=1$ so the coefficient space
has dimension $M$ rather than $pM$, and $(r,b)$ are known. `--task darcy` and
`--task poisson` use the operator-valued NTK on a PDE solution operator, where
$p=1+\tilde d$ and the $\tilde d^{\,2}$ dependence discussed in Section 3.2 bites: with
$\tilde d=8$ and hence $p=9$, the coefficient dimension $pM$ approaches $nd_v$ at the sample
sizes reachable here, so the two tasks give quite different pictures. Both are reported.

---

## Runtime

On four cores the full suite takes a few minutes; the rate experiment dominates. `--quick`
shrinks every experiment to a smoke test that checks the scripts run end to end and write
parseable output, which is what CI runs. It does *not* check that the measured exponents
agree with the theory — that needs the full run.