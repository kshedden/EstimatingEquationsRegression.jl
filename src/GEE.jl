module GEE

using Distributions, LinearAlgebra, StatsBase, DataFrames, StatsModels
using LinearAlgebra: BlasReal, diag
using StatsBase: CoefTable, StatisticalModel, RegressionModel
import StatsBase: coef, coeftable, vcov, stderr, dof, dof_residual, fit, predict
using GLM: Link, LinPredModel, LinPred, ModResp, linkfun, linkinv, glmvar, mueta
using GLM: IdentityLink, LogLink, LogitLink
using GLM: GeneralizedLinearModel, dispersion_parameter, canonicallink

export fit, fit!, GeneralizedEstimatingEquationsModel, vcov, stderr, coef
export CorStruct, IndependenceCor, ExchangeableCor, OrdinalIndependenceCor, AR1Cor
export corparams, dispersion, dof, scoretest, modelmatrix, gee
export expand_ordinal, GEEE, geee, coefnames, coeftable

export QIF, qif, QIFBasis, QIFIdentityBasis, QIFHollowBasis, QIFSubdiagonalBasis

export predict
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
#include("gee2.jl")
end
