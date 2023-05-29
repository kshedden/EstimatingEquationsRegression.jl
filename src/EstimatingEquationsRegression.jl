module EstimatingEquationsRegression

import StatsAPI: coef, coeftable, coefnames, vcov, stderr, dof, dof_residual
import StatsAPI: HypothesisTest, fit, predict, pvalue, residuals

using Distributions, LinearAlgebra, DataFrames, StatsModels
using StatsBase: CoefTable, StatisticalModel, RegressionModel

using GLM: Link, LinPredModel, LinPred, ModResp, linkfun, linkinv, glmvar, mueta
using GLM: IdentityLink, LogLink, LogitLink
using GLM: GeneralizedLinearModel, dispersion_parameter, canonicallink

# From StatsAPI
export fit, vcov, stderr, coef, coefnames, modelmatrix, predict, coeftable, pvalue
export dof, residuals

export fit!, GeneralizedEstimatingEquationsModel, resid_pearson
export CorStruct, IndependenceCor, ExchangeableCor, OrdinalIndependenceCor, AR1Cor
export corparams, dispersion, dof, scoretest, gee
export expand_ordinal, GEEE, geee

# GLM exports
export IdentityLink, LogLink, LogitLink

# QIF exports
export QIF, qif, QIFBasis, QIFIdentityBasis, QIFHollowBasis, QIFSubdiagonalBasis

# Variance functions
export Varfunc, geevar, ConstantVar, IdentityVar, BinomialVar, PowerVar

const FP = AbstractFloat
const FPVector{T<:FP} = AbstractArray{T,1}

include("varfunc.jl")
include("corstruct.jl")
include("linpred.jl")
include("geefit.jl")
include("scoretest.jl")
include("utils.jl")
include("expectiles.jl")
include("qif.jl")
end
