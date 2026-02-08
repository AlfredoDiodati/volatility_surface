# Observation-Driven MSM with Fixed Binomial Cascade  
*(p = 0.5, mean zero returns)*

---

## 1. Returns

$$
r_t \mid \mathcal F_{t-1} \sim \mathcal N(0,\sigma_t^2)
$$

The conditional mean is fixed at zero.  
No additional noise or mixture parameters are introduced.

---

## 2. Volatility: multiplicative binomial cascade

$$
\sigma_t^2
=
\bar\sigma^2
\prod_{i=1}^k M_{i,t}
$$

Each multiplier takes values in a two-point support:

$$
M_{i,t}\in\{m_0,m_1\},\qquad 0<m_0<1<m_1
$$

Unconditional binomial law (fixed, MSM-style):

$$
\mathbb P(M_{i,t}=m_1)=\mathbb P(M_{i,t}=m_0)=\tfrac12
$$

Mean-one normalization:

$$
\tfrac12 m_0 + \tfrac12 m_1 = 1
\quad\Longrightarrow\quad
m_1 = 2 - m_0
$$

The unconditional variance level is therefore identified by $\bar\sigma^2$.

---

## 3. Soft-binomial representation

Introduce latent continuous indicators:

$$
z_{i,t}=\Lambda(f_{i,t})=\frac{1}{1+e^{-f_{i,t}}}
$$

Map indicators to multipliers:

$$
M_{i,t}
=
m_0\left(\frac{m_1}{m_0}\right)^{z_{i,t}}
=
m_0\left(\frac{2-m_0}{m_0}\right)^{z_{i,t}}
$$

For moderate or large $|f_{i,t}|$, the multipliers concentrate near
$\{m_0,2-m_0\}$.

Log-variance:

$$
\log\sigma_t^2
=
\log\bar\sigma^2
+
\sum_{i=1}^k
\Big[
\log m_0
+
z_{i,t}\log\!\left(\frac{2-m_0}{m_0}\right)
\Big]
$$

---

## 4. Score

Gaussian score with respect to log-variance:

$$
s_t
=
\frac{\partial \log p(r_t\mid\sigma_t^2)}{\partial \log\sigma_t^2}
=
\tfrac12\left(\frac{r_t^2}{\sigma_t^2}-1\right)
$$

Component score via the chain rule:

$$
\nabla_{i,t}
=
s_t\;
z_{i,t}(1-z_{i,t})
\log\!\left(\frac{2-m_0}{m_0}\right)
$$

---

## 5. Observation-driven dynamics (multifrequency geometry)

Geometric persistence structure:

$$
\phi_i=\exp\!\big(-\lambda_1 b^{\,i-1}\big),
\qquad \lambda_1>0,\ b>1
$$

State update:

$$
f_{i,t+1}
=
\phi_i f_{i,t}
+
\alpha\,\nabla_{i,t}
$$

No intercepts and no per-frequency parameters.

---

## 6. Parameter vector (constant in $k$)

$$
\theta
=
\big(
\log\bar\sigma^2,\;
m_0,\;
\alpha,\;
\lambda_1,\;
b
\big)
$$

All remaining quantities are implied by normalization and structure.

---

## 7. Interpretation

- Exact MSM binomial cascade in the marginal law  
- Fixed unconditional left/right mass ($p=0.5$)  
- Geometric hierarchy of time scales across frequencies  
- Score-driven updates replace Bayesian filtering  
- No latent HMM states  
- Parameter dimension independent of the number of frequencies $k$
