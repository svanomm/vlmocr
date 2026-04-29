regress postestimation — Postestimation tools for regress 2556

# Predictions

## Description for predict

predict creates a new variable containing predictions such as linear predictions, residuals, standardized residuals, Studentized residuals, Cook’s distance, leverage, probabilities, expected values, DFBETAs for *varname*, standard errors, COVRATIOS, DFITS, and Welsch distances.

## Menu for predict

Statistics > Postestimation

## Syntax for predict

predict $[type]$ *newvar* $[if]$ $[in]$ $[, statistic]$

| *statistic* | Description |
| :--- | :--- |
| Main | |
| `xb` | linear prediction; the default |
| `residuals` | residuals |
| `score` | score; equivalent to residuals |
| `rstandard` | standardized residuals |
| `rstudent` | Studentized (jackknifed) residuals |
| `cooksd` | Cook’s distance |
| `leverage` \| `hat` | leverage (diagonal elements of hat matrix) |
| `pr(a,b)` | $\Pr(y_j \mid a < y_j < b)$ |
| `e(a,b)` | $E(y_j \mid a < y_j < b)$ |
| `ystar(a,b)` | $E(y_j^*), y_j^* = \max\{a, \min(y_j, b)\}$ |
| * `dfbeta(varname)` | DFBETA for *varname* |
| `stdp` | standard error of the linear prediction |
| `stdf` | standard error of the forecast |
| `stdr` | standard error of the residual |
| * `covratio` | COVRATIO |
| * `dfits` | DFITS |
| * `welsch` | Welsch distance |

Unstarred statistics are available both in and out of sample; type `predict` ... `if e(sample)` ... if wanted only for the estimation sample. Starred statistics are calculated only for the estimation sample, even when if `e(sample)` is not specified.

`rstandard`, `rstudent`, `cooksd`, `leverage`, `dfbeta()`, `stdf`, `stdr`, `covratio`, `dfits`, and `welsch` are not available if any `vce()` other than `vce(ols)` was specified with `regress`.

`xb`, `residuals`, `score`, and `stdp` are the only options allowed with `svy` estimation results.

where *a* and *b* may be numbers or variables; *a* missing ($a \geq .$) means $-\infty$, and *b* missing ($b \geq .$) means $+\infty$; see [U] 12.2.1 Missing values.
**regress postestimation — Postestimation tools for regress 2557**

## Options for predict

<div style="border: 1px solid black; padding: 5px;">
Main
</div>

**xb**, the default, calculates the linear prediction.

**residuals** calculates the residuals.

**score** is equivalent to **residuals** in linear regression.

**rstandard** calculates the standardized residuals.

**rstudent** calculates the Studentized (jackknifed) residuals.

**cooksd** calculates the Cook’s $D$ influence statistic (Cook 1977).

**leverage** or **hat** calculates the diagonal elements of the projection (“hat”) matrix.

**pr**($a,b$) calculates $\Pr(a < \mathbf{x}_j\mathbf{b} + u_j < b)$, the probability that $y_j|\mathbf{x}_j$ would be observed in the interval $(a, b)$.

$a$ and $b$ may be specified as numbers or variable names; $lb$ and $ub$ are variable names;
**pr**($20,30$) calculates $\Pr(20 < \mathbf{x}_j\mathbf{b} + u_j < 30)$;
**pr**($lb,ub$) calculates $\Pr(lb < \mathbf{x}_j\mathbf{b} + u_j < ub)$; and
**pr**($20,ub$) calculates $\Pr(20 < \mathbf{x}_j\mathbf{b} + u_j < ub)$.

$a$ missing ($a \geq .$) means $-\infty$; **pr**($.,30$) calculates $\Pr(-\infty < \mathbf{x}_j\mathbf{b} + u_j < 30)$;
**pr**($lb,30$) calculates $\Pr(-\infty < \mathbf{x}_j\mathbf{b} + u_j < 30)$ in observations for which $lb \geq .$
and calculates $\Pr(lb < \mathbf{x}_j\mathbf{b} + u_j < 30)$ elsewhere.

$b$ missing ($b \geq .$) means $+\infty$; **pr**($20,.$) calculates $\Pr(+\infty > \mathbf{x}_j\mathbf{b} + u_j > 20)$;
**pr**($20,ub$) calculates $\Pr(+\infty > \mathbf{x}_j\mathbf{b} + u_j > 20)$ in observations for which $ub \geq .$
and calculates $\Pr(20 < \mathbf{x}_j\mathbf{b} + u_j < ub)$ elsewhere.

**e**($a,b$) calculates $E(\mathbf{x}_j\mathbf{b} + u_j | a < \mathbf{x}_j\mathbf{b} + u_j < b)$, the expected value of $y_j|\mathbf{x}_j$ conditional on $y_j|\mathbf{x}_j$
being in the interval $(a, b)$, meaning that $y_j|\mathbf{x}_j$ is truncated. $a$ and $b$ are specified as they are for **pr**().

**ystar**($a,b$) calculates $E(y_j^*)$, where $y_j^* = a$ if $\mathbf{x}_j\mathbf{b} + u_j \leq a$, $y_j^* = b$ if $\mathbf{x}_j\mathbf{b} + u_j \geq b$, and $y_j^* = \mathbf{x}_j\mathbf{b} + u_j$
otherwise, meaning that $y_j^*$ is censored. $a$ and $b$ are specified as they are for **pr**().

**dfbeta**(*varname*) calculates the DFBETA for *varname*, the difference between the regression coefficient
when the $j$th observation is included and excluded, said difference being scaled by the estimated standard error of the coefficient. *varname* must have been included among the regressors in the previously
fitted model. The calculation is automatically restricted to the estimation subsample.

**stdp** calculates the standard error of the prediction, which can be thought of as the standard error of the
predicted expected value or mean for the observation’s covariate pattern. The standard error of the
prediction is also referred to as the standard error of the fitted value.

**stdf** calculates the standard error of the forecast, which is the standard error of the point prediction for
1 observation. It is commonly referred to as the standard error of the future or forecast value. By
construction, the standard errors produced by **stdf** are always larger than those produced by **stdp**;
see *Methods and formulas*.

**stdr** calculates the standard error of the residuals.

**covratio** calculates COVRATIO (Belsley, Kuh, and Welsch 1980), a measure of the influence of the $j$th
observation based on considering the effect on the variance–covariance matrix of the estimates. The
calculation is automatically restricted to the estimation subsample.