```@meta
EditURL = "src/sleepstudy.jl"
```

## Sleep study (linear GEE)

The sleepstudy data are from a study of subjects experiencing sleep
deprivation.  Reaction times were measured at baseline (day 0) and
after each of several consecutive days of sleep deprivation (3 hours
of sleep each night).  This example fits a linear model to the reaction
times, with the mean reaction time being modeled as a linear function
of the number of days since the subject began experiencing sleep
deprivation.  The data are clustered by subject, and since the data
are collected by time, we use a first-order autoregressive working
correlation model.

````julia
using EstimatingEquationsRegression, RDatasets, StatsModels

slp = dataset("lme4", "sleepstudy");

# The data must be sorted by the group id.
slp = sort(slp, :Subject);

m1 = fit(GeneralizedEstimatingEquationsModel,
         @formula(Reaction ~ Days), slp, slp[:, :Subject],
         IdentityLink(), ConstantVar(), AR1Cor())
````

````
StatsModels.TableRegressionModel{EstimatingEquationsRegression.GeneralizedEstimatingEquationsModel{EstimatingEquationsRegression.GEEResp{Float64}, EstimatingEquationsRegression.DensePred{Float64}}, Matrix{Float64}}

Reaction ~ 1 + Days

Coefficients:
────────────────────────────────────────────────────────────────────────
                Coef.  Std. Error      z  Pr(>|z|)  Lower 95%  Upper 95%
────────────────────────────────────────────────────────────────────────
(Intercept)  253.489      6.35647  39.88    <1e-99  241.031      265.948
Days          10.4668     1.43944   7.27    <1e-12    7.64552     13.288
────────────────────────────────────────────────────────────────────────
````

The scale parameter (unexplained standard deviation).

````julia
sqrt(dispersion(m1.model))
````

````
47.76062829893422
````

The AR1 correlation parameter.

````julia
corparams(m1.model)
````

````
0.7670316895600812
````

The results indicate that reaction times become around 10.5 units
slower for each additional day on the study, starting from a baseline
mean value of around 253 units.  There are around 47.8 standard
deviation units of unexplained variation, and the within-subject
autocorrelation of the unexplained variation decays exponentially with
a parameter of around 0.77.

There are several approaches to estimating the covariance of the
parameter estimates, the default is the robust (sandwich) approach.
Other options are the "naive" approach, the "md" (Mancl-DeRouen)
bias-reduced approach, and the "kc" (Kauermann-Carroll) bias-reduced
approach.  Below we use the Mancl-DeRouen approach.  Note that this
does not change the coefficient estimates, but the standard errors,
test statistics (z), and p-values are affected.

````julia
m2 = fit(GeneralizedEstimatingEquationsModel,
         @formula(Reaction ~ Days), slp, slp.Subject,
         IdentityLink(), ConstantVar(), AR1Cor(), cov_type="md")
````

````
StatsModels.TableRegressionModel{EstimatingEquationsRegression.GeneralizedEstimatingEquationsModel{EstimatingEquationsRegression.GEEResp{Float64}, EstimatingEquationsRegression.DensePred{Float64}}, Matrix{Float64}}

Reaction ~ 1 + Days

Coefficients:
────────────────────────────────────────────────────────────────────────
                Coef.  Std. Error      z  Pr(>|z|)  Lower 95%  Upper 95%
────────────────────────────────────────────────────────────────────────
(Intercept)  253.489      6.73038  37.66    <1e-99  240.298      266.681
Days          10.4668     1.52411   6.87    <1e-11    7.47956     13.454
────────────────────────────────────────────────────────────────────────
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

