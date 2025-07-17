module EstimatingEquationsRegression

import Base: show
import StatsAPI: coef, coeftable, coefnames, vcov, stderror, dof, dof_residual
import StatsAPI: HypothesisTest, fit, predict, pvalue, residuals, nobs

using Distributions, LinearAlgebra, DataFrames, StatsModels
using StatsBase: CoefTable, StatisticalModel, RegressionModel
using PrettyTables

using GLM: Link, LinPredModel, LinPred, ModResp, mueta
using GLM: IdentityLink, LogLink, LogitLink
using GLM: GeneralizedLinearModel, dispersion_parameter

# Need to extend these
# TODO mueta is removed from recent GLM
import GLM: linkfun, linkinv, mueta, canonicallink

export show

# From StatsAPI
export fit, vcov, stderror, coef, coefnames, modelmatrix, predict, coeftable, pvalue
export dof, residuals, nobs

# Correlation structure exports
export CorStruct, IndependenceCor, ExchangeableCor, OrdinalIndependenceCor, AR1Cor

export fit!, GeneralizedEstimatingEquationsModel, resid_pearson
export corparams, dispersion, dof, scoretest, gee, NoDistribution
export expand_ordinal

# GEEE exports
export GEEE, geee

# GEE2 exports
export GeneralizedEstimatingEquations2Model, SigmoidLink

# GLM exports
export IdentityLink, LogLink, LogitLink, canonicallink

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
include("gee2.jl")

end
