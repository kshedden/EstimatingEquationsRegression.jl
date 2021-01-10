# Examples

```@meta
DocTestSetup = quote
    using CategoricalArrays, DataFrames, Distributions, GEE, RDatasets, StatsModels
end
```

```jldoctest
julia> using DataFrames, GEE, RDatasets

julia> slp = dataset("lme4", "sleepstudy")
180×3 DataFrame
│ Row │ Reaction │ Days  │ Subject │
│     │ Float64  │ Int32 │ Cat…    │
├─────┼──────────┼───────┼─────────┤
│ 1   │ 249.56   │ 0     │ 308     │
│ 2   │ 258.705  │ 1     │ 308     │
│ 3   │ 250.801  │ 2     │ 308     │
│ 4   │ 321.44   │ 3     │ 308     │
│ 5   │ 356.852  │ 4     │ 308     │
│ 6   │ 414.69   │ 5     │ 308     │
│ 7   │ 382.204  │ 6     │ 308     │
│ 8   │ 290.149  │ 7     │ 308     │
│ 9   │ 430.585  │ 8     │ 308     │
│ 10  │ 466.353  │ 9     │ 308     │
│ 11  │ 222.734  │ 0     │ 309     │
│ 12  │ 205.266  │ 1     │ 309     │
⋮
│ 168 │ 304.631  │ 7     │ 371     │
│ 169 │ 350.781  │ 8     │ 371     │
│ 170 │ 369.469  │ 9     │ 371     │
│ 171 │ 269.412  │ 0     │ 372     │
│ 172 │ 273.474  │ 1     │ 372     │
│ 173 │ 297.597  │ 2     │ 372     │
│ 174 │ 310.632  │ 3     │ 372     │
│ 175 │ 287.173  │ 4     │ 372     │
│ 176 │ 329.608  │ 5     │ 372     │
│ 177 │ 334.482  │ 6     │ 372     │
│ 178 │ 343.22   │ 7     │ 372     │
│ 179 │ 369.142  │ 8     │ 372     │
│ 180 │ 364.124  │ 9     │ 372     │

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

julia> corparams(m.model)
0.7670316895600818
```
