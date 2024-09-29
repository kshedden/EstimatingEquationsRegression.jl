```@meta
EditURL = "src/contraception.jl"
```

## Contraception use (logistic GEE)

This example uses data from a 1988 survey of contraception use
among women in Bangladesh.  Contraception use is binary, so it is
natural to use logistic regression.  Contraceptive use is coded 'Y'
and 'N' and we will recode it as numeric (Y=1, N=0) below.

Contraception use may vary by the district in which a woman lives, and
since there are 60 districts it may not be practical to use fixed
effects (allocating a parameter for every district).  Therefore, we fit
a marginal logistic regression model using GEE and cluster the results
by district.

To explain the variation in contraceptive use, we use the woman's age,
the number of living children that she has at the time of the survey,
and an indicator of whether the woman lives in an urban area.  As a
working correlation structure, the women are modeled as being
exchangeable within each district.

````julia
using EstimatingEquationsRegression, RDatasets, StatsModels, Distributions

con = dataset("mlmRev", "Contraception")

con[!, :Use1] = [x == "Y" ? 1.0 : 0.0 for x in con[:, :Use]]

con = sort(con, :District)

# There are two equivalent ways to fit a GEE model.  First we
# demonstrate the quasi-likelihood approach, in which we specify
# the link function, variance function, and working correlation structure.
m1 = fit(GeneralizedEstimatingEquationsModel,
         @formula(Use1 ~ Age + LivCh + Urban),
         con, con[:, :District],
         LogitLink(), BinomialVar(), ExchangeableCor())

# This is the distribution-based approach to fit a GEE model, in
# which we specify the distribution family, working correlation
# structure, and link function.
m2 = fit(GeneralizedEstimatingEquationsModel,
         @formula(Use1 ~ Age + LivCh + Urban),
         con, con[:, :District],
         Binomial(), ExchangeableCor(), LogitLink())
````

````
StatsModels.TableRegressionModel{EstimatingEquationsRegression.GeneralizedEstimatingEquationsModel{EstimatingEquationsRegression.GEEResp{Float64}, EstimatingEquationsRegression.DensePred{Float64}}, Matrix{Float64}}

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
````

There is a moderate level of correlation between women
living in the same district:

````julia
corparams(m1.model)
````

````
0.06367178989068953
````

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
be used instead, as shown below.  Note that the parent model need not
be fit before conducting the score test.

````julia
m3 = fit(GeneralizedEstimatingEquationsModel,
         @formula(Use1 ~ Age + LivCh + Urban),
         con, con[:, :District],
         LogitLink(), BinomialVar(), ExchangeableCor();
         dofit=false)

m4 = fit(GeneralizedEstimatingEquationsModel,
         @formula(Use1 ~ Age + Urban),
         con, con[:, :District],
         LogitLink(), BinomialVar(), ExchangeableCor())

st = scoretest(m3.model, m4.model)
pvalue(st)
````

````
1.569826834191268e-6
````

The score test above is used to assess whether the `LivCh` variable
contributes to the variation in contraceptive use.  A score test is
useful here because `LivCh` is a categorical variable and is coded
using multiple categorical indicators.  The score test is an omnibus
test assessing whether any of these indicators contributes to
explaining the variation in the response.  The small p-value shown
above strongly suggests that `LivCh` is a relevant variable.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

