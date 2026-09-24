# Theory And Algorithms

This section documents the mathematical definitions implemented by the backend and core package.

## Notation

Let $X \in \mathbb{R}^{n \times d}$ denote the design matrix and $y \in \mathbb{R}^n$ the observed scalar response vector. For a query point $x \in \mathbb{R}^d$, the surrogate prediction is $\widehat{f}(x)$.

Unless disabled, models internally standardize each feature using training-set statistics:

$$
z_j = \frac{x_j - \mu_j}{s_j}, \qquad j = 1, \dots, d,
$$

with $s_j = 1$ substituted for zero-variance features.

## Polynomial Least Squares

The least-squares model expands the normalized inputs into a polynomial basis $A \in \mathbb{R}^{n \times p}$ and solves

$$
\min_{w \in \mathbb{R}^p}
\frac{1}{2}\lVert y - Aw \rVert_2^2
+ \frac{\lambda_2}{2}\lVert w_{1:p-1} \rVert_2^2
+ \lambda_1 \lVert w_{1:p-1} \rVert_1,
$$

where the intercept is excluded from regularization. When $\lambda_1 = 0$, the implementation uses least squares; when $\lambda_1 > 0$, it uses coordinate descent ({numref}`alg-ls-coordinate-descent`).

```{algorithm}
:name: alg-ls-coordinate-descent

\begin{algorithm}
\caption{Elastic-net coordinate descent for polynomial least squares ($\lambda_1 > 0$)}
\begin{algorithmic}[1]
\Require $A = [\mathbf{1} \;\; A_{\setminus 1}] \in \mathbb{R}^{n \times p}$, $y \in \mathbb{R}^n$, $\lambda_1 > 0$, $\lambda_2 \ge 0$, $\varepsilon > 0$, $T_{\max} \in \mathbb{N}$
\Ensure $w \in \mathbb{R}^p$
\State $\bar{a} \gets \tfrac{1}{n} A_{\setminus 1}^\top \mathbf{1}, \quad Z \gets A_{\setminus 1} - \mathbf{1}\bar{a}^\top, \quad \tilde{y} \gets y - \bar{y}\mathbf{1}$
\State $v \gets \mathbf{0} \in \mathbb{R}^{p-1}, \quad r \gets \tilde{y}, \quad c_j \gets \lVert Z_{:,j} \rVert_2^2$
\For{$t = 1, \dots, T_{\max}$}
    \For{$j \in \{1, \dots, p-1\} : c_j > 0$}
        \State $\rho_j \gets Z_{:,j}^\top r + c_j v_j$
        \State $v_j^{+} \gets \dfrac{\operatorname{sign}(\rho_j)\,\max(|\rho_j| - \lambda_1,\, 0)}{c_j + \lambda_2}$
        \State $r \gets r - Z_{:,j}\,(v_j^{+} - v_j), \quad \delta_j \gets |v_j^{+} - v_j|, \quad v_j \gets v_j^{+}$
    \EndFor
    \If{$\max_j \delta_j \le \varepsilon \max(\lVert v \rVert_\infty, 1)$}
        \State \Break
    \EndIf
\EndFor
\State \Return $w \gets \bigl(\bar{y} - \bar{a}^\top v,\; v\bigr)$
\end{algorithmic}
\end{algorithm}
```

### Gradient

Because the basis functions are explicit polynomials, the gradient is computed analytically:

$$
\nabla \widehat{f}(x) = \sum_{k=1}^{p} w_k \nabla \phi_k(x).
$$

## Radial Basis Function Interpolation

The RBF model is defined as

$$
\widehat{f}(x) = \sum_{i=1}^{n} w_i \, \phi\bigl(\gamma \lVert x - x^{(i)} \rVert\bigr) + q(x),
$$

where $q(x)$ is a polynomial tail and the weights satisfy the moment constraints $P^\top w = 0$.

The backend supports these basis functions:

$$
\phi(r) =
\begin{cases}
e^{-r^2} & \text{Gaussian}, \\
\sqrt{1 + r^2} & \text{Multiquadric}, \\
\frac{1}{\sqrt{1 + r^2}} & \text{Inverse multiquadric}, \\
r & \text{Linear}, \\
r^3 & \text{Cubic}, \\
r^2 \log r & \text{Thin plate spline.}
\end{cases}
$$

The coefficients are obtained from the augmented linear system

$$
\begin{bmatrix}
\Phi + \eta I & P \\
P^\top & 0
\end{bmatrix}
\begin{bmatrix}
w \\
\beta
\end{bmatrix}
=
\begin{bmatrix}
y \\
0
\end{bmatrix}.
$$

```{algorithm}
:name: alg-rbf-training

\begin{algorithm}
\caption{RBF training with optional automatic shape selection}
\begin{algorithmic}[1]
\Require $X \in \mathbb{R}^{n \times d}$, $y \in \mathbb{R}^n$, kernel $\phi$, tail monomials $\pi_1, \dots, \pi_m$, $\eta \ge 0$, $\gamma \in \mathbb{R}_{>0} \cup \{\texttt{auto}\}$
\Ensure $w \in \mathbb{R}^n$, $\beta \in \mathbb{R}^m$
\State $D_{ij} \gets \lVert x^{(i)} - x^{(j)} \rVert_2, \quad P_{ik} \gets \pi_k(x^{(i)})$
\State $M(\gamma) \gets \begin{bmatrix} \phi(\gamma D) + \eta I & P \\ P^\top & 0 \end{bmatrix} \in \mathbb{R}^{(n+m) \times (n+m)}$
\If{$\gamma = \texttt{auto} \;\wedge\; \phi \in \{\text{Gaussian}, \text{MQ}, \text{IMQ}\}$}
    \For{$\gamma_k = 10^{-2 + k/10}, \quad k = 0, \dots, 40$}
        \State $B \gets M(\gamma_k)^{-1}, \quad c \gets B_{:,1:n}\, y$
        \If{$\lVert M(\gamma_k) \rVert_1 \lVert B \rVert_1 > 10^{10}$}
            \State $E_k \gets \infty$
        \Else
            \State $E_k \gets \dfrac{1}{n} \displaystyle\sum_{i=1}^{n} \left( \frac{c_i}{B_{ii}} \right)^{2}$ \Comment{Rippa's LOO error}
        \EndIf
    \EndFor
    \State $\gamma \gets \gamma_{k^\star}, \quad k^\star = \arg\min_k E_k$
\EndIf
\State $\begin{bmatrix} w \\ \beta \end{bmatrix} \gets M(\gamma)^{-1} \begin{bmatrix} y \\ \mathbf{0} \end{bmatrix}$
\State \Return $(w, \beta)$
\end{algorithmic}
\end{algorithm}
```

## Ordinary Kriging

The implemented kriging model assumes a constant mean $\beta$ and a powered-exponential correlation:

$$
R(x, x') = \exp\left(-\sum_{k=1}^{d} \theta_k \lvert x_k - x'_k \rvert^p\right),
\qquad 0 < p \le 2.
$$

For fixed $\theta$, the concentrated log-likelihood is minimized through the equivalent objective

$$
\mathcal{L}(\theta) = n \log \sigma^2(\theta) + \log \det R_\eta(\theta),
$$

where $R_\eta = R + \eta I$ includes the nugget term. The predictor at a new point $x$ is

$$
\widehat{f}(x) = \beta + r(x)^\top R_\eta^{-1}(y - \beta \mathbf{1}),
$$

and the kriging standard deviation is

$$
s(x) = \sqrt{\sigma^2 \left(1 - r(x)^\top R_\eta^{-1} r(x) + \frac{(1 - \mathbf{1}^\top R_\eta^{-1} r(x))^2}{\mathbf{1}^\top R_\eta^{-1} \mathbf{1}}\right)}.
$$

```{algorithm}
:name: alg-kriging-mle

\begin{algorithm}
\caption{Maximum-likelihood estimation of kriging correlation parameters}
\begin{algorithmic}[1]
\Require $X \in \mathbb{R}^{n \times d}$, $y \in \mathbb{R}^n$, $p \in (0, 2]$, $\eta \ge 0$, $[\theta_{\min}, \theta_{\max}]$, $r \in \mathbb{N}$
\Ensure $\theta^\star \in \mathbb{R}^d_{>0}$, $\beta$, $\sigma^2$
\Function{NLL}{$\theta$}
    \State $R_{ij} \gets \exp\!\Bigl(-\sum_{k=1}^{d} \theta_k \bigl|x^{(i)}_k - x^{(j)}_k\bigr|^{p}\Bigr)$
    \State $L L^\top \gets R + \eta I$ \Comment{retry with $\eta \gets 10\eta$ if not SPD}
    \State $\beta \gets \dfrac{\mathbf{1}^\top R_\eta^{-1} y}{\mathbf{1}^\top R_\eta^{-1} \mathbf{1}}$
    \State $\sigma^2 \gets \tfrac{1}{n} (y - \beta\mathbf{1})^\top R_\eta^{-1} (y - \beta\mathbf{1})$
    \State \Return $n \log \sigma^2 + 2 \sum_{i=1}^{n} \log L_{ii}$
\EndFunction
\State $\ell_{\min} \gets \log_{10} \theta_{\min}, \quad \ell_{\max} \gets \log_{10} \theta_{\max}, \quad \Omega \gets [\ell_{\min}, \ell_{\max}]^d$
\State $\ell^{(0)} \gets \tfrac{1}{2}(\ell_{\min} + \ell_{\max})\mathbf{1}, \quad \ell^{(s)} \sim \mathcal{U}(\Omega), \; s = 1, \dots, r$
\State $f^\star \gets \infty$
\For{$s = 0, \dots, r$}
    \State $\hat{\ell} \gets \operatorname{L\text{-}BFGS\text{-}B}\bigl(\ell \mapsto \operatorname{NLL}(10^{\ell}),\; \ell^{(s)},\; \Omega\bigr)$
    \If{$\operatorname{NLL}(10^{\hat{\ell}}) < f^\star$}
        \State $f^\star \gets \operatorname{NLL}(10^{\hat{\ell}}), \quad \ell^\star \gets \hat{\ell}$
    \EndIf
\EndFor
\State $\theta^\star \gets 10^{\ell^\star}$
\State \Return $\theta^\star$ with $(\beta, \sigma^2)$ from \Call{NLL}{$\theta^\star$}
\end{algorithmic}
\end{algorithm}
```

## Weighted Ensemble

If experts $\widehat{f}_1, \dots, \widehat{f}_m$ are trained, the ensemble predictor is

$$
\widehat{f}_{\mathrm{ens}}(x) = \sum_{j=1}^{m} \omega_j \widehat{f}_j(x),
\qquad
\omega_j \ge 0,
\qquad
\sum_{j=1}^{m} \omega_j = 1.
$$

For `weights="cv"`, the backend computes

$$
\omega_j \propto \frac{1}{\operatorname{MSE}_j},
$$

where $\operatorname{MSE}_j$ is the cross-validated mean squared error of expert $j$ ({numref}`alg-ensemble-cv-weights`).

```{algorithm}
:name: alg-ensemble-cv-weights

\begin{algorithm}
\caption{Cross-validated inverse-MSE ensemble weights}
\begin{algorithmic}[1]
\Require $X \in \mathbb{R}^{n \times d}$, $y \in \mathbb{R}^n$, expert training maps $\mathcal{A}_1, \dots, \mathcal{A}_m$, folds $K \ge 2$
\Ensure $\omega \in \Delta^{m-1}$, fitted experts $\widehat{f}_1, \dots, \widehat{f}_m$
\State $\pi \gets$ random permutation of $\{1, \dots, n\}$; $\;(I_1, \dots, I_K) \gets$ contiguous blocks of $\pi$
\State $\operatorname{MSE}_j \gets 0, \quad j = 1, \dots, m$
\For{$k = 1, \dots, K$}
    \For{$j = 1, \dots, m$}
        \State $\widehat{f}_j^{(-k)} \gets \mathcal{A}_j\bigl(X_{\bar{I}_k}, y_{\bar{I}_k}\bigr), \quad \bar{I}_k = \{1, \dots, n\} \setminus I_k$
        \State $\operatorname{MSE}_j \gets \operatorname{MSE}_j + \dfrac{1}{n} \displaystyle\sum_{i \in I_k} \bigl(\widehat{f}_j^{(-k)}(x^{(i)}) - y_i\bigr)^2$
    \EndFor
\EndFor
\State $\omega_j \gets \dfrac{1 / \max(\operatorname{MSE}_j, \epsilon)}{\sum_{l=1}^{m} 1 / \max(\operatorname{MSE}_l, \epsilon)}$ \Comment{$\epsilon$: smallest positive float}
\State $\widehat{f}_j \gets \mathcal{A}_j(X, y), \quad j = 1, \dots, m$
\State \Return $\omega$, $\bigl(\widehat{f}_j\bigr)_{j=1}^{m}$
\end{algorithmic}
\end{algorithm}
```

## Ranking Metrics In The Backend

The dashboard backend reports and ranks models using:

- mean squared error: $\operatorname{MSE} = \frac{1}{n}\sum_i (y_i - \widehat{y}_i)^2$,
- root mean squared error: $\operatorname{RMSE} = \sqrt{\operatorname{MSE}}$,
- mean absolute error: $\operatorname{MAE} = \frac{1}{n}\sum_i |y_i - \widehat{y}_i|$,
- coefficient of determination: $R^2 = 1 - \frac{\sum_i (y_i - \widehat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}$,
- predicted holdout $R^2$ based on the training mean baseline.

For the weighted composite ranking, each metric is normalized to a penalty scale and combined as a convex weighted sum.