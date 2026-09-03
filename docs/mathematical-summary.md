# Mathematical summary

How each object in Nguyen & Mücke, [arXiv:2603.00971](https://arxiv.org/abs/2603.00971),
maps onto the code. Equation and assumption numbers refer to that paper.

## Setting

An input space $\mathcal{U}$ (a Banach space), an output space $\mathcal{V}$ (a separable
Hilbert space), and an unknown distribution $\rho$ on $\mathcal{Z}=\mathcal{U}\times\mathcal{V}$.
With the squared loss, the minimizer of the expected risk over all measurable maps is the
regression operator $G_\rho(u)=\int_\mathcal{V}v\,\rho(dv\mid u)$, and the quantity to be
controlled is the excess risk

$$\|G_\rho-\widehat G\|^2_{L^2(\rho_\mathcal{U})}
  = \int_\mathcal{U}\|G_\rho(u)-\widehat G(u)\|^2_\mathcal{V}\,d\rho_\mathcal{U}(u).$$

For operator learning, $\mathcal{U}$ and $\mathcal{V}$ are function spaces on a domain
$\mathcal{X}$ and $G_\rho$ is the solution operator of a boundary value problem.

**In code.** Outputs live in $\mathcal{V}=\mathbb{R}^{d_v}$ under the *standard* Euclidean
inner product. A discretized function space carries instead the empirical inner product
$\langle f,g\rangle_{n_\mathcal{X}}=n_\mathcal{X}^{-1}\sum_kf(x_k)g(x_k)$ used in the
paper's Appendix A.2, so datasets rescale function values by
$1/\sqrt{n_\mathcal{X}}$ before returning them
(`kerop.data.isometric_scale`). Keeping the rescaling in the data rather than carrying a
weight matrix through the linear algebra means every risk this package reports is already an
$L^2(\mathcal{X},\rho_x)$ quantity. `kerop.metrics.excess_risk` returns the norm, not its
square, to match the statement of Theorem 3.4.

## Vector-valued kernels and their random features

A kernel $K:\mathcal{U}\times\mathcal{U}\to\mathcal{L}(\mathcal{V})$ of positive type induces
a unique $\mathcal{V}$-valued RKHS $\mathcal{H}$ with $F(u)=K_u^*F$ and $K(u,\tilde u)=K_u^*K_{\tilde u}$.
Assumption 2.1 asks that $K$ be a mixture of rank-one operator-valued features,

$$K(u,\tilde u)=\sum_{i=1}^p\int_\Omega\varphi_i(u,\omega)\otimes\varphi_i(\tilde u,\omega)\,d\pi(\omega),
\qquad\sum_{i=1}^p\|\varphi_i(u,\omega)\|^2_\mathcal{V}\le\kappa^2\ \ \pi\text{-a.s.},$$

and the random feature kernel replaces the integral by an average over
$\omega_1,\dots,\omega_M\sim\pi$.

**In code.** Everything goes through one object, the *feature tensor*

$$\Psi_M(u)c=\frac{1}{\sqrt M}\sum_{i,m}c_{i,m}\varphi_i(u,\omega_m),
\qquad\Psi_M(u)\in\mathcal{L}(\mathbb{R}^{pM},\mathcal{V}),$$

which satisfies $\Psi_M(u)\Psi_M(\tilde u)^*=K_M(u,\tilde u)$ and identifies
$\mathcal{H}_M\cong\mathbb{R}^{pM}$. `RandomFeatureMap.feature_tensor` returns it with shape
`(n, d_v, p*M)`; `design_matrix` flattens it to `(n*d_v, p*M)`, at which point

$$\widehat\Sigma_M=\frac{Z^\top Z}{n},\qquad
\widehat{\mathcal{S}}^*_M\mathbf v=\frac{Z^\top\operatorname{vec}(\mathbf v)}{n}.$$

The vector-valued structure survives in the rows: each sample contributes $d_v$ of them, all
sharing one coefficient vector. That is the difference from $d_v$ independent scalar
regressions, and it is only substantive when the features $\varphi_i(u,\omega)$ are
non-trivially $\mathcal{V}$-valued — which is exactly what the NTK features below are.

### The four feature maps

**`MercerFeatures`** — importance sampling of a known Mercer expansion. If
$K=\sum_i\sigma_i\Phi_i\otimes\Phi_i$ with $\{\Phi_i\}$ orthonormal in
$L^2(\rho_\mathcal{U};\mathcal{V})$, then taking $\Omega=\{1,\dots,S\}$,
$\pi(i)=\sigma_i/Z$ with $Z=\sum_i\sigma_i$, and $\varphi(u,i)=\sqrt Z\,\Phi_i(u)$ satisfies
Assumption 2.1 exactly with $p=1$ and $\kappa^2=Z\sup_{u,i}\|\Phi_i(u)\|^2$. Because
$\{\Phi_i\}$ diagonalizes $\mathcal{L}$, this is the one construction where $r$ and $b$ are
known rather than estimated.

**`ScalarNTKFeatures`** — the NTK of a two-layer network. With $J(u)=(u,1)\in\mathbb{R}^{\tilde d}$,
$\tilde d=d+1$, and $b^{(0)}_m\sim\pi_0$, the feature blocks are

$$\psi_m(u)=\sigma(\langle b^{(0)}_m,J(u)\rangle),\qquad
\psi'_{m,j}(u)=\sigma'(\langle b^{(0)}_m,J(u)\rangle)J(u)^{(j)},$$

giving $p=1+\tilde d=d+2$ — precisely the $p=d+2$ quoted in Appendix A.3.

**`OperatorNTKFeatures`** — the operator-valued NTK of a shallow neural operator. The feature
map is $\Phi^M_u(v)=\nabla_\theta\langle G_{\theta_0}(u),v\rangle_{L^2(\rho_x)}$ and the same
$\psi,\psi'$ formulas apply, but now with
$J(u)(x)=(A(u)(x),\,u(x),\,c(x))^\top$ depending on $x$. Each $\psi_m(u)$ is therefore an
*element of* $\mathcal{V}$ rather than a scalar: these are genuinely vector-valued random
features, with $p=1+\tilde d$ and $\tilde d=d_k+d_y+d_b$.

**`SeparableRFF`** — random Fourier features for $K(u,\tilde u)=k(u,\tilde u)T$, the classical
vector-valued baseline (Brault et al., 2016; Minh, 2016).

### The exact kernels

`kerop.kernels` supplies the $M\to\infty$ limit of each map, so the random feature estimator
can be compared against the kernel method it approximates. For ReLU and
$b^{(0)}\sim\mathcal{N}(0,s^2I)$ the NTK limit is closed-form via the arc-cosine kernels
(Cho & Saul, 2009): with $a=J(u)(x)$, $a'=J(\tilde u)(x')$ and $\theta=\angle(a,a')$,

$$K(u,\tilde u)(x,x')=\underbrace{\frac{s^2\|a\|\|a'\|}{2\pi}\bigl(\sin\theta+(\pi-\theta)\cos\theta\bigr)}_{\mathbb{E}[\psi\psi]}
+\underbrace{\langle a,a'\rangle\frac{\pi-\theta}{2\pi}}_{\mathbb{E}[\sum_j\psi'_j\psi'_j]}.$$

ReLU is positively homogeneous, so the activation block carries $s^2$; the derivative block,
built from the scale-invariant Heaviside function, does not. Flattening the sample and
collocation axes turns the block Gram matrix into an arc-cosine kernel on the resulting point
cloud, with the block layout falling out automatically.

## Spectral filtering

Definition 2.2 asks for $\{\phi_\lambda\}_{\lambda\in(0,1]}$ on $[0,1]$ with

$$\sup_t|t\phi_\lambda(t)|\le D,\qquad\sup_t|\phi_\lambda(t)|\le E/\lambda,\qquad
\sup_t|r_\lambda(t)|\le c_0,\qquad r_\lambda(t):=1-t\phi_\lambda(t),$$

and defines the *qualification* as the largest $\nu$ with
$\sup_t|r_\lambda(t)|t^q\le c_q\lambda^q$ for every $q\in[0,\nu]$.

| Family | $\phi_\lambda$ or its recursion | $\lambda$ | Qualification |
| --- | --- | --- | --- |
| Tikhonov | $(t+\lambda)^{-1}$ | explicit | $1$ |
| Iterated Tikhonov, order $m$ | $r_\lambda=(\lambda/(t+\lambda))^m$ | explicit | $m$ |
| Spectral cut-off | $t^{-1}\mathbf 1\{t\ge\lambda\}$ | explicit | $\infty$ |
| Landweber (gradient descent) | $x_{k+1}=x_k-\alpha(Ax_k-b)$ | $1/(\alpha T)$ | $\infty$ |
| Heavy-ball, fixed $\beta$ | $x_{k+1}=x_k-\alpha(Ax_k-b)+\beta(x_k-x_{k-1})$ | $(1-\beta)/(\alpha T)$ | $\infty$ (see below) |
| Brakhage's $\nu$-method | $x_k=x_{k-1}+\mu_k(x_{k-1}-x_{k-2})+\omega_k\alpha(b-Ax_{k-1})$ | $1/(\alpha T^2)$ | $\nu$ |

Two implementation points matter.

**The residual is the primitive object, not the filter.** Condition (2.10) probes the regime
where $r_\lambda(t)$ is many orders of magnitude below one, and computing it as
$1-t\phi_\lambda(t)$ subtracts two nearly equal numbers. Dividing the resulting rounding
error by $\lambda^q$ with $\lambda=10^{-7}$, $q=3$ manufactures a spurious constant of order
$10^5$ — identically for every family, which is how the problem announces itself. Each family
therefore overrides `residual_function` with an exact closed form, checked against the
definition in the tests at a moderate $\lambda$ where the generic route is still accurate.

**Qualification is about divergence, not size.** (2.10) asks for $c_q(\lambda)$ to be
*bounded* uniformly in $\lambda$, so the discriminating signal is whether it diverges as
$\lambda\to0$, not whether it is large. `measure_qualification` regresses
$\log c_q(\lambda)$ on $\log\lambda$ and accepts $q$ when the slope is flat; beyond the
qualification the slope is $\nu-q$, which lets $\nu$ itself be read off. This distinction is
not academic: heavy-ball run for few iterations has large but bounded constants, and a
threshold on $c_q$ alone reports it as saturated at $\nu\approx0.75$ when in fact its
constants flatten onto the Landweber values as $\lambda\to0$.

For heavy-ball with fixed $\beta$, the small-eigenvalue behaviour is
$r_T(t)\approx(1-\alpha t/(1-\beta))^T$, so constant momentum rescales the step rather than
producing the $T^{-2}$ acceleration; that requires $\beta\to1$. The genuinely accelerated
method is the $\nu$-method, whose residual is a Jacobi polynomial with
$\sup_t|r_T(t)|t^q\lesssim T^{-2q}$ for $q\le\nu$.

## The estimator and its exact counterpart

The random feature estimator is (2.11), $F^M_\lambda=\phi_\lambda(\widehat\Sigma_M)\widehat{\mathcal{S}}^*_M\mathbf v$.
For the exact kernel, the operator identity
$\phi_\lambda(\mathcal{S}^*\mathcal{S})\mathcal{S}^*=\mathcal{S}^*\phi_\lambda(\mathcal{S}\mathcal{S}^*)$
moves the filter onto the block Gram matrix:

$$\widehat G(u)=\frac1n\sum_{j=1}^nK(u,u_j)c_j,\qquad
c=\phi_\lambda(\mathbf{G}/n)\operatorname{vec}(\mathbf v),\qquad
\mathbf{G}\in\mathbb{R}^{nd_v\times nd_v}.$$

For Tikhonov this is the familiar $(\mathbf{G}+n\lambda I)^{-1}$. The two estimators differ
only in *which* operator the filter is applied to, and they share the filter code verbatim.

| Estimator | Time | Memory |
| --- | --- | --- |
| Random features, explicit filter | $O(nd_v(pM)^2+(pM)^3)$ | $O(nd_v\,pM)$ |
| Random features, iterative filter | $O(nd_v\,pM\,t)$ | $O(nd_v\,pM)$ |
| Exact operator-valued kernel | $O((nd_v)^3)$ | $O((nd_v)^2)$ |

Definition 2.2 constrains the filters on $(0,1]$, so the operator is divided by an upper
bound on its spectral norm before a filter is applied. A Lanczos estimate is used rather
than the trace, which overestimates by up to a factor of the dimension and would silently
inflate the iteration counts of the iterative filters by the same factor.

## The synthetic instance with known $(r,b)$

Testing the rate needs an instance where $r$ and $b$ are known, which needs control over the
spectrum of $\mathcal{L}$. `SpectralOperatorModel` specifies that spectrum directly. On
$\mathcal{U}=[0,1]^d$ with $\rho_\mathcal{U}$ uniform and $\{e_j\}$ the tensorized cosine
basis (orthonormal, and uniformly bounded by $2^{d/2}$, which is what makes (2.6) hold with
an explicit constant), set

$$K(u,\tilde u)=\sum_{j=1}^J\mu_j\,e_j(u)e_j(\tilde u)\,T_j,\qquad
T_j=R_j\operatorname{diag}(\nu)R_j^\top,\qquad\mu_j=j^{-1/b},$$

with $R_j$ Haar-random orthogonal. Writing $g_{j,k}$ for the $k$-th column of $R_j$, a
direct computation gives $\mathcal{L}(e_jg_{j,k})=\mu_j\nu_k\,e_jg_{j,k}$ — the identity the
whole construction rests on, and one the test suite verifies by quadrature. Hence:

- the spectrum of $\mathcal{L}$ is the product set $\{\mu_j\nu_k\}$, whose counting function
  is $\#\{(j,k):\mu_j\nu_k>\epsilon\}=\epsilon^{-b}\sum_k\nu_k^b$, so
  $\mathcal{N}(\lambda)\asymp\lambda^{-b}$ with the constant depending on $\nu$ but the
  exponent not — **Assumption 3.3 with exponent $b$**;
- defining the target through $G_\rho=\mathcal{L}^rH$ for an explicit $H=\sum_ih_i\Phi_i$
  gives **Assumption 3.2 with exponent $r$**.

Because the $R_j$ differ across modes, $T_j\ne T_{j'}$ and the kernel is *not* separable: the
output covariance rotates with the input mode, so the problem does not reduce to $d_v$
independent scalar regressions. Flat output weights would make $T_j=d_v^{-1}I$ and destroy
this, which is why $\nu$ is chosen non-constant.

One subtlety decides whether the instance is usable. With $\sigma_i\asymp i^{-1/b}$ and
$h_i\propto i^{-(1/2+\epsilon)}$, the exact bias of the spectral cut-off is

$$\|r_\lambda(\mathcal{L})G_\rho\|^2\approx\sum_{i>i_\lambda}\sigma_i^{2r}h_i^2
  \asymp\lambda^{2r+2\epsilon b},\qquad i_\lambda\asymp\lambda^{-b},$$

so any extra decay $\epsilon>0$ raises the realized source exponent to $r+\epsilon b$.
Choosing $\epsilon=0$ — coefficients exactly at the boundary of square-summability — gives a
clean $\lambda^r$ bias, and $H$ is still a legitimate $L^2$ element because the expansion is
truncated, with $\|H\|^2\asymp\log(Jd_v)$. Damping $h_i$ by a logarithm instead, the textbook
way to sit just inside $\ell^2$ in infinite dimensions, introduces a $1/\log(1/\lambda)$
factor that inflates a power-law fit of the bias by roughly $+0.2$ over three decades of
$\lambda$. Both exponents are measured rather than assumed, by
`effective_dimension_fit` and `source_exponent_fit`, from the exact spectrum and the exact
bias with no sampling involved.

## The PDE solution operators

Both are posed on $[0,1]$ with homogeneous Dirichlet conditions and solved to machine
precision, so the excess risk is measured against the true operator rather than noisy labels.

**Poisson**, $-u''=f$: diagonal in the Dirichlet sine basis with eigenvalues $(k\pi)^{-2}$,
so the operator is linear and exact in closed form.

**Darcy**, $-(au')'=f$ learning $a\mapsto u$: integrates twice,
$u'(x)=(C-F(x))/a(x)$ with $F(x)=\int_0^xf$ and $C=\int_0^1F/a\big/\int_0^11/a$ fixed by
$u(1)=0$. Nonlinear in $a$.

The lifting operator $A$ is a bank of $d_k$ nonlocal Gaussian smoothing operators at spread
length scales. This is what makes the induced kernel genuinely operator-valued: the value of
a feature at $x$ depends on the whole input function, not just on $u(x)$.

## Theorem 3.4 as code

`kerop.theory` implements the prescriptions directly: `regularization_parameter`,
`features_required`, `iterations_required`, `min_sample_size`, `neural_operator_width`, and
`check_assumptions`, which rejects $2r+b\le1$ and a filter whose qualification is below
$r\vee1$.

One property of the theorem as stated is worth flagging. The feature requirement is
continuous at $r=1/2$, where $b(2r-1)$ vanishes, but **not at $r=1$**: the second branch
gives $(1+b)/(2+b)$ there while the third gives $2/(2+b)$, a jump of $(1-b)/(2+b)$ that
closes only in the capacity-independent case $b=1$. The two branches come from different
arguments in the proof — the $r>1$ case relies on the operator inequalities developed in
Appendix B.5 — so the bound just above $r=1$ is not tight against the bound at $r=1$. The
implementation follows the theorem rather than smoothing this over, and a test documents it.