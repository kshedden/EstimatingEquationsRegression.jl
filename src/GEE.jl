module GEE

using Distributions, LinearAlgebra, StatsBase
using LinearAlgebra: BlasReal, diag
using StatsBase: CoefTable, StatisticalModel, RegressionModel
using GLM: Link, LinPredModel, LinPred, ModResp, linkfun, linkinv, glmvar, mueta
using GLM: GeneralizedLinearModel, dispersion_parameter, canonicallink

import StatsBase: coef, coeftable, vcov, stderr, dof, dof_residual, fit

export fit, fit!, GeneralizedEstimatingEquationsModel, vcov, stderr, coef
export CorStruct, IndependenceCor, ExchangeableCor, AR1Cor, corparams, dispersion, dof
export scoretest, modelmatrix, gee

const FP = AbstractFloat
const FPVector{T<:FP} = AbstractArray{T,1}

include("corstruct.jl")
include("linpred.jl")
include("geefit.jl")
include("scoretest.jl")
end
