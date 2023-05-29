```@meta
EditURL = "<unknown>/hospitalstay.jl"
```

## Length of hospital stay

Below we look at data on length of hospital stay for patients
undergoing a cardiovascular procedure.  This example illustrates how
the variance function can be changed to a non-standard form.  Modeling
the variance as μ^p for 1<=p<=2 gives a Tweedie model, and when p=1 or
p=2 we have a Poisson or a Gamma model, respectively.  For 1<p<2, the
inference is via quasi-likelihood as the score equations solved by GEE
do not correspond to the score function of the log-likelihood of the
data (even when there is no dependence within clusters).  In this
example, as is often the case, the parameter estimates and standard
errors are not strongly sensitive to the variance model.

````julia
using GEE, RDatasets, GLM

azpro = dataset("COUNT", "azpro")

azpro[!, :Los] = Float64.(azpro[:, :Los])

azpro = sort(azpro, :Hospital)

m1 = fit(GeneralizedEstimatingEquationsModel,
         @formula(Los ~ Procedure + Sex + Age75), azpro, azpro[:, :Hospital],
         LogLink(), IdentityVar(), IndependenceCor())

m2 = fit(GeneralizedEstimatingEquationsModel,
         @formula(Los ~ Procedure + Sex + Age75), azpro, azpro[:, :Hospital],
         LogLink(), PowerVar(1.5), IndependenceCor())
````

````
StatsModels.TableRegressionModel{GeneralizedEstimatingEquationsModel{GEE.GEEResp{Float64}, GEE.DensePred{Float64}}, Matrix{Float64}}

Los ~ 1 + Procedure + Sex + Age75

Coefficients:
──────────────────────────────────────────────────────────────────────────
                 Coef.  Std. Error      z  Pr(>|z|)   Lower 95%  Upper 95%
──────────────────────────────────────────────────────────────────────────
(Intercept)   1.69053    0.0333798  50.65    <1e-99   1.62511     1.75596
Procedure     0.942344   0.0422612  22.30    <1e-99   0.859514    1.02517
Sex          -0.152185   0.0206432  -7.37    <1e-12  -0.192644   -0.111725
Age75         0.145134   0.0262325   5.53    <1e-07   0.0937191   0.196548
──────────────────────────────────────────────────────────────────────────
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

