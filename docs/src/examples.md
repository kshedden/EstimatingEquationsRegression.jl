# Examples

```@meta
DocTestSetup = quote
    using CategoricalArrays, DataFrames, Distributions, GEE, GLM, RDatasets, StatsModels
end
```

The sleepstudy data contains data from a study of subjects
experiencing sleep deprivation.  Reaction times were measured at
baseline (day 0) and after each of several consecutive days of sleep
deprivation (3 hours of sleep per night).  This example fits a linear
model to the reaction times, with the mean reaction time being modeled
as a linear function of the number of days that a subject has been on
the study.  The data are clustered by subject, and since the data are
collected by time, we use a first-order autoregressive working
correlation model.

```jldoctest
julia> slp = dataset("lme4", "sleepstudy");

julia> slp = sort(slp, :Subject);

julia> head(slp)
6×3 DataFrame
│ Row │ Reaction │ Days  │ Subject │
│     │ Float64  │ Int32 │ Cat…    │
├─────┼──────────┼───────┼─────────┤
│ 1   │ 249.56   │ 0     │ 308     │
│ 2   │ 258.705  │ 1     │ 308     │
│ 3   │ 250.801  │ 2     │ 308     │
│ 4   │ 321.44   │ 3     │ 308     │
│ 5   │ 356.852  │ 4     │ 308     │
│ 6   │ 414.69   │ 5     │ 308     │

julia> m = gee(@formula(Reaction ~ Days), slp, slp.Subject, Normal(), AR1Cor(), IdentityLink())
StatsModels.TableRegressionModel{GeneralizedEstimatingEquationsModel{GEE.GEEResp{Float64},GEE.DensePred{Float64}},Array{Float64,2}}

Reaction ~ 1 + Days

Coefficients:
────────────────────────────────────────────────────────────────────────
                Coef.  Std. Error      z  Pr(>|z|)  Lower 95%  Upper 95%
────────────────────────────────────────────────────────────────────────
(Intercept)  253.489      6.35647  39.88    <1e-99  241.031      265.948
Days          10.4668     1.43944   7.27    <1e-12    7.64552     13.288
────────────────────────────────────────────────────────────────────────

julia> sqrt(dispersion(m.model))
47.760628298934215

julia> corparams(m.model)
0.7670316895600818
```

The results indicate that reaction times become around 10.4 units
slower for each additional day on the study, starting from a baseline
mean value of around 253 units.  There are around 47.8 standard
deviation units of unexplained variation, and the within-subject
autocorrelation of the unexplained variation decays with a parameter
of around 0.77.

The next example uses data from a 1988 survey of contraception use
among women in Bangladesh.  Contraception use may vary by district,
and since there are 60 districts it may not be practical to use fixed
effects, allocating a parameter for every district.  Instead, we fit a
marginal logistic regression model and cluster the results by
district.  To explain the variation in contraceptive use, we use the
woman's age, the number of living children that she has at the time of
the survey, and an indicator of whether the woman lives in an urban
area.  As a working correlation structure, the women are modeled as
being exchangeable within each district.

```jldoctest
julia> con = dataset("mlmRev", "Contraception");

julia> con[:, :Use1] = [x == "Y" ? 1.0 : 0.0 for x in con[:, :Use]];

julia> con = sort(con, :District);

julia> head(con)
6×7 DataFrame
│ Row │ Woman │ District │ Use  │ LivCh │ Age     │ Urban │ Use1    │
│     │ Cat…  │ Cat…     │ Cat… │ Cat…  │ Float64 │ Cat…  │ Float64 │
├─────┼───────┼──────────┼──────┼───────┼─────────┼───────┼─────────┤
│ 1   │ 1     │ 1        │ N    │ 3+    │ 18.44   │ Y     │ 0.0     │
│ 2   │ 2     │ 1        │ N    │ 0     │ -5.5599 │ Y     │ 0.0     │
│ 3   │ 3     │ 1        │ N    │ 2     │ 1.44    │ Y     │ 0.0     │
│ 4   │ 4     │ 1        │ N    │ 3+    │ 8.44    │ Y     │ 0.0     │
│ 5   │ 5     │ 1        │ N    │ 0     │ -13.559 │ Y     │ 0.0     │
│ 6   │ 6     │ 1        │ N    │ 0     │ -11.56  │ Y     │ 0.0     │

julia> m = gee(@formula(Use1 ~ Age + LivCh + Urban), con, con[:, :District], Binomial(), ExchangeableCor(), LogitLink())
StatsModels.TableRegressionModel{GeneralizedEstimatingEquationsModel{GEE.GEEResp{Float64},GEE.DensePred{Float64}},Array{Float64,2}}

Use1 ~ 1 + Age + LivCh + Urban

Coefficients:
────────────────────────────────────────────────────────────────────────────
                  Coef.  Std. Error      z  Pr(>|z|)   Lower 95%   Upper 95%
────────────────────────────────────────────────────────────────────────────
(Intercept)  -1.6052     0.180205    -8.91    <1e-18  -1.95839    -1.252
Age          -0.0253885  0.00674873  -3.76    0.0002  -0.0386158  -0.0121612
LivCh: 1      1.06049    0.188551     5.62    <1e-7    0.690932    1.43004
LivCh: 2      1.31064    0.161219     8.13    <1e-15   0.994655    1.62662
LivCh: 3+     1.28688    0.195082     6.60    <1e-10   0.904523    1.66923
Urban: Y      0.679963   0.157503     4.32    <1e-4    0.371263    0.988663
────────────────────────────────────────────────────────────────────────────

julia> corparams(m.model)
0.0638658332586221
```

Since GEE estimation is based on quasi-likelihood, there is no
likelihood ratio test for comparing nested models.  A score test can
be used instead, as shown below.  Note that the parent model must not
be fit before conducting the score test.

```
julia> m1 = gee(@formula(Use1 ~ Age + LivCh + Urban), con, con[:, :District], Binomial(), ExchangeableCor(), LogitLink(); dofit=false);

julia> m0 = gee(@formula(Use1 ~ Age + Urban), con, con[:, :District], Binomial(), ExchangeableCor(), LogitLink());

julia> scoretest(m1.model, m0.model)
```

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

```jldoctest
julia> azpro = dataset("COUNT", "azpro");

julia> azpro[!, :Los] = Array{Float64}(azpro[:, :Los]);

julia> m1 = gee(@formula(Los ~ Procedure + Sex + Age75), azpro, azpro[:, :Hospital], Poisson(), IndependenceCor(), LogLink())
StatsModels.TableRegressionModel{GeneralizedEstimatingEquationsModel{GEE.GEEResp{Float64},GEE.DensePred{Float64}},Array{Float64,2}}

Los ~ 1 + Procedure + Sex + Age75

Coefficients:
─────────────────────────────────────────────────────────────────────────
                 Coef.  Std. Error      z  Pr(>|z|)  Lower 95%  Upper 95%
─────────────────────────────────────────────────────────────────────────
(Intercept)   1.68953    0.0251049  67.30    <1e-99   1.64033    1.73874
Procedure     0.940554   0.0232797  40.40    <1e-99   0.894926   0.986181
Sex          -0.147551   0.0232818  -6.34    <1e-9   -0.193182  -0.10192
Age75         0.142054   0.024159    5.88    <1e-8    0.094703   0.189405
─────────────────────────────────────────────────────────────────────────

julia> import GLM

julia> GLM.glmvar(::GLM.Poisson, μ::Real) = μ^1.5

julia> m2 = gee(@formula(Los ~ Procedure + Sex + Age75), azpro, azpro[:, :Hospital], Poisson(),IndependenceCor(), LogLink())
StatsModels.TableRegressionModel{GeneralizedEstimatingEquationsModel{GEE.GEEResp{Float64},GEE.DensePred{Float64}},Array{Float64,2}}

Los ~ 1 + Procedure + Sex + Age75

Coefficients:
──────────────────────────────────────────────────────────────────────────
                 Coef.  Std. Error      z  Pr(>|z|)   Lower 95%  Upper 95%
──────────────────────────────────────────────────────────────────────────
(Intercept)   1.69051    0.0255617  66.13    <1e-99   1.64041     1.74061
Procedure     0.942381   0.0232502  40.53    <1e-99   0.896812    0.987951
Sex          -0.152183   0.0236146  -6.44    <1e-9   -0.198467   -0.105899
Age75         0.145119   0.0244092   5.95    <1e-8    0.0972776   0.19296
──────────────────────────────────────────────────────────────────────────
```
