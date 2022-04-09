# Examples

```@meta
DocTestSetup = quote
    using CategoricalArrays, DataFrames, Distributions, GEE, GLM, RDatasets, StatsModels
end
```

## Sleep study (linear GEE)

The sleepstudy data are from a study of subjects experiencing sleep
deprivation.  Reaction times were measured at baseline (day 0) and
after each of several consecutive days of sleep deprivation (3 hours
of sleep per night).  This example fits a linear model to the reaction
times, with the mean reaction time being modeled as a linear function
of the number of days since the subject began experiencing sleep
deprivation.  The data are clustered by subject, and since the data
are collected by time, we use a first-order autoregressive working
correlation model.

```jldoctest sleep
julia> slp = dataset("lme4", "sleepstudy");

julia> slp = sort(slp, :Subject);

julia> first(slp, 6)
6×3 DataFrame
 Row │ Reaction  Days   Subject
     │ Float64   Int32  Cat…
─────┼──────────────────────────
   1 │  249.56       0  308
   2 │  258.705      1  308
   3 │  250.801      2  308
   4 │  321.44       3  308
   5 │  356.852      4  308
   6 │  414.69       5  308

julia> m = gee(@formula(Reaction ~ Days), slp, slp.Subject, Normal(), AR1Cor(), IdentityLink())
StatsModels.TableRegressionModel{GeneralizedEstimatingEquationsModel{GEE.GEEResp{Float64}, GEE.DensePred{Float64}}, Matrix{Float64}}

Reaction ~ 1 + Days

Coefficients:
────────────────────────────────────────────────────────────────────────
                Coef.  Std. Error      z  Pr(>|z|)  Lower 95%  Upper 95%
────────────────────────────────────────────────────────────────────────
(Intercept)  253.489      6.35647  39.88    <1e-99  241.031      265.948
Days          10.4668     1.43944   7.27    <1e-12    7.64552     13.288
────────────────────────────────────────────────────────────────────────

julia> round(sqrt(dispersion(m.model)), digits=3)
47.761

julia> round(corparams(m.model), digits=3)
0.767
```

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

```jldoctest sleep
julia> m = gee(@formula(Reaction ~ Days), slp, slp.Subject, Normal(), AR1Cor(), IdentityLink(), cov_type="md")
StatsModels.TableRegressionModel{GeneralizedEstimatingEquationsModel{GEE.GEEResp{Float64}, GEE.DensePred{Float64}}, Matrix{Float64}}

Reaction ~ 1 + Days

Coefficients:
────────────────────────────────────────────────────────────────────────
                Coef.  Std. Error      z  Pr(>|z|)  Lower 95%  Upper 95%
────────────────────────────────────────────────────────────────────────
(Intercept)  253.489      6.73038  37.66    <1e-99  240.298      266.681
Days          10.4668     1.52411   6.87    <1e-11    7.47956     13.454
────────────────────────────────────────────────────────────────────────
```

## Contraception use (logistic GEE)

The next example uses data from a 1988 survey of contraception use
among women in Bangladesh.  Since contraception use is binary, it is
natural to use logistic regression.  Contraceptive use is coded 'Y'
and 'N' and we need to recode it as numeric below.

Contraception use may vary by the district in which a woman lives, and
since there are 60 districts it may not be practical to use fixed
effects (allocating a parameter for every district).  Instead, we fit
a marginal logistic regression model using GEE and cluster the results
by district.

To explain the variation in contraceptive use, we use the woman's age,
the number of living children that she has at the time of the survey,
and an indicator of whether the woman lives in an urban area.  As a
working correlation structure, the women are modeled as being
exchangeable within each district.

```jldoctest
julia> con = dataset("mlmRev", "Contraception");

julia> con[:, :Use1] = [x == "Y" ? 1.0 : 0.0 for x in con[:, :Use]];

julia> con = sort(con, :District);

julia> first(con, 6)
6×7 DataFrame
 Row │ Woman  District  Use   LivCh  Age       Urban  Use1
     │ Cat…   Cat…      Cat…  Cat…   Float64   Cat…   Float64
─────┼────────────────────────────────────────────────────────
   1 │ 1      1         N     3+      18.44    Y          0.0
   2 │ 2      1         N     0       -5.5599  Y          0.0
   3 │ 3      1         N     2        1.44    Y          0.0
   4 │ 4      1         N     3+       8.44    Y          0.0
   5 │ 5      1         N     0      -13.559   Y          0.0
   6 │ 6      1         N     0      -11.56    Y          0.0

julia> m = gee(@formula(Use1 ~ Age + LivCh + Urban), con, con[:, :District], Binomial(), ExchangeableCor(), LogitLink())
StatsModels.TableRegressionModel{GeneralizedEstimatingEquationsModel{GEE.GEEResp{Float64}, GEE.DensePred{Float64}}, Matrix{Float64}}

Use1 ~ 1 + Age + LivCh + Urban

Coefficients:
────────────────────────────────────────────────────────────────────────────
                  Coef.  Std. Error      z  Pr(>|z|)   Lower 95%   Upper 95%
────────────────────────────────────────────────────────────────────────────
(Intercept)  -1.60517    0.180197    -8.91    <1e-18  -1.95835    -1.25199
Age          -0.0253875  0.00674856  -3.76    0.0002  -0.0386144  -0.0121605
LivCh: 1      1.06048    0.188543     5.62    <1e-07   0.690944    1.43002
LivCh: 2      1.31064    0.161215     8.13    <1e-15   0.994662    1.62661
LivCh: 3+     1.28683    0.195078     6.60    <1e-10   0.904483    1.66918
Urban: Y      0.680084   0.15749      4.32    <1e-04   0.371409    0.988758
────────────────────────────────────────────────────────────────────────────

julia> round(corparams(m.model), digits=3)
0.064
```

We see that older women are less likely to use contraception than
younger women.  With each additional year of age, the log odds of
contraception use decreases by 0.03.  The `LivCh` variable (number of
living children) is categorical, and the reference level is 0,
i.e. the woman has no living children.  We see that women with living
children are more likely than women with no living children to use
contraception, especially if the woman has 2 or more living children.
Furthermore, we see that women living in an urban environment are more
likely to use contraception.

The exchangeable correlation parameter is 0.064, meaning that there is
a small tendency for women living in the same district to have similar
contraceptive-use behavior.  In other words, some districts have
greater rates of contraception use and other districts have lower
rates of contraceptive use.  This is likely due to variables
characterizing the residents of different districts that we did not
include in the model as covariates.

Since GEE estimation is based on quasi-likelihood, there is no
likelihood ratio test for comparing nested models.  A score test can
be used instead, as shown below.  Note that the parent model must not
be fit before conducting the score test (use dofit=false to create a
model without fitting it).

```
julia> m1 = gee(@formula(Use1 ~ Age + LivCh + Urban), con, con[:, :District], Binomial(), ExchangeableCor(), LogitLink(); dofit=false);

julia> m0 = gee(@formula(Use1 ~ Age + Urban), con, con[:, :District], Binomial(), ExchangeableCor(), LogitLink());

julia> scoretest(m1.model, m0.model)
(DoF = 3, Stat = 29.73395517089856, Pvalue = 1.569826834191268e-6)
```

The score test above is used to assess whether the `LivCh` variable
contributes to the variation in contraceptive use.  A score test is
useful here because `LivCh` is a categorical variable and is coded
using multiple categorical indicators.  The score test is an omnibus
test assessing whether any of these indicators contributes to
explaining the variation in the response.  The small p-value shown
above strongly suggests that `LivCh` is a relevant variable.

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

```jldoctest
julia> azpro = dataset("COUNT", "azpro");

julia> azpro[!, :Los] = Float64.(azpro[:, :Los]);

julia> azpro = sort(azpro, :Hospital);

julia> GLM.glmvar(::GLM.Poisson, μ::Real) = μ

julia> m1 = gee(@formula(Los ~ Procedure + Sex + Age75), azpro, azpro[:, :Hospital], Poisson(), IndependenceCor(), LogLink())
StatsModels.TableRegressionModel{GeneralizedEstimatingEquationsModel{GEE.GEEResp{Float64}, GEE.DensePred{Float64}}, Matrix{Float64}}

Los ~ 1 + Procedure + Sex + Age75

Coefficients:
─────────────────────────────────────────────────────────────────────────
                 Coef.  Std. Error      z  Pr(>|z|)  Lower 95%  Upper 95%
─────────────────────────────────────────────────────────────────────────
(Intercept)   1.68953    0.0335989  50.29    <1e-99   1.62368    1.75538
Procedure     0.940554   0.0424738  22.14    <1e-99   0.857307   1.0238
Sex          -0.147551   0.0217237  -6.79    <1e-10  -0.190129  -0.104973
Age75         0.142054   0.0248406   5.72    <1e-07   0.093367   0.190741
─────────────────────────────────────────────────────────────────────────

julia> GLM.glmvar(::GLM.Poisson, μ::Real) = μ^1.5

julia> m2 = gee(@formula(Los ~ Procedure + Sex + Age75), azpro, azpro[:, :Hospital], Poisson(), IndependenceCor(), LogLink())
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
```
