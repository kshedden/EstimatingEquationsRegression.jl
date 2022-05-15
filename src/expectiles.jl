using LinearAlgebra, BlockDiagonals

"""
    GEEEResp

The response vector and group labels for GEE expectile analysis.
"""
struct GEEEResp{T<:Real} <: ModResp

    # n-dimensional vector of responses
    y::Vector{T}

    # Group labels, sorted
    grp::Vector
end

abstract type GEEELinPred <: LinPred end

struct GEEEDensePred{T<:Real} <: GEEELinPred

    # p x n covariate matrix, variables are rows, cases are columns
    X::Matrix{T}
end

function update_score!(
    pp::GEEEDensePred,
    i1::Int,
    i2::Int,
    cor::CorStruct,
    linpred::AbstractVector{T},
    sd::AbstractVector{T},
    resid::AbstractVector{T},
    f::T,
    scr,
) where {T<:Real}
    x = @view pp.X[:, i1:i2]
    scr .+= f * x * covsolve(cor, linpred, sd, zeros(0), resid)
end

function update_denom!(
    pp::GEEEDensePred,
    i1::Int,
    i2::Int,
    cor::CorStruct,
    linpred::AbstractVector{T},
    sd::AbstractVector{T},
    resid::AbstractVector{T},
    f::T,
    denom,
) where {T<:Real}
    x = @view pp.X[:, i1:i2]
    denom .+= f * x * diagm(covsolve(cor, linpred, sd, zeros(0), resid)) * x'
end

"""
    GEEE

Fit expectile regression models using GEE.
"""
mutable struct GEEE{T<:Real} <: AbstractGEE

    # The response
    rr::GEEEResp{T}

    # The covariates
    pp::GEEEDensePred{T}

    # Each column contains the first and last index of a group.
    grpix::Matrix{Int}

    # Vector of tau values for which expectiles are jointly estimated
    tau::Vector{Float64}

    # Weight for each tau value
    tau_wt::Vector{Float64}

    # The parameters to be estimated
    beta::Matrix{Float64}

    # The working correlation model
    cor::Vector{CorStruct}

    # Map conditional mean to conditional variance
    varfunc::Function

    # The variance/covariance matrix of the parameter estimates
    vcov::Matrix{Float64}

    # The size of the biggest group.
    mxgrp::Int
end

function GEEE(
    y,
    X,
    grp,
    tau;
    cor = CorStruct[IndependenceCor() for _ in eachindex(tau)],
    varfunc = nothing,
    tau_wt = nothing,
)
    @assert length(y) == size(X, 2) == length(grp)

    if !issorted(grp)
        error("Group vector is not sorted")
    end
    gix, mx = groupix(grp)

    if isnothing(tau_wt)
        tau_wt = ones(length(tau))
    end

    # Default variance function
    if isnothing(varfunc)
        varfunc = x -> 1.0
    end

    # Number of expectiles
    q = length(tau)

    # Number of covariates
    p = size(X, 1)

    # Each column contains the covariates at an expectile
    beta = zeros(p, q)

    # The variance/covariance matrix of all parameters
    vcov = zeros(p * q, p * q)

    rr = GEEEResp(y, grp)
    pp = GEEEDensePred(X)

    return GEEE(rr, pp, gix, tau, tau_wt, beta, cor, varfunc, vcov, mx)
end

# Place the linear predictor for the j^th tau value and
# for group g into linpred.
function linpred!(geee::GEEE, tauj::Int, g::Int, linpred::T) where {T<:AbstractVector}
    i1, i2 = geee.grpix[:, g]
    linpred!(geee.pp, i1, i2, geee.beta[:, tauj], linpred)
end

function linpred!(
    pp::GEEEDensePred{T},
    i1::Int,
    i2::Int,
    beta::AbstractVector{T},
    linpred::AbstractVector{T},
) where {T<:Real}
    linpred .= pp.X[:, i1:i2]' * beta
end

# Multiply the residuals in-place by their own check function.
function check_resid_mul!(resid::AbstractVector{S}, tau::T) where {S,T<:Real}
    for j in eachindex(resid)
        if resid[j] < 0
            resid[j] *= 1 - tau
        else
            resid[j] *= tau
        end
    end
end

# Place the score for the jth expectile into 'scr'.
function score!(
    geee::GEEE,
    j::Int,
    scr::Vector{T},
    denom::Matrix{T},
    linpred::Vector{T},
    resid::Vector{T},
) where {T<:Real}

    scr .= 0
    denom .= 0
    p, n = size(geee.pp.X)
    grpix = geee.grpix

    # Backing storage
    cresid = zeros(geee.mxgrp)

    # Loop over the groups
    for (g, (i1, i2)) in enumerate(eachcol(grpix))

        # Group size
        gs = i2 - i1 + 1

        # Update the linear predictor and residuals
        linpred1 = @view linpred[i1:i2]
        resid1 = @view resid[i1:i2]
        cresid1 = @view cresid[1:gs]
        linpred!(geee, j, g, linpred1)
        resid1 .= geee.rr.y[i1:i2] - linpred1

        # The checked residuals
        cresid1 .= resid1
        check!(cresid1, geee.tau[j])

        # The product of the checked residuals and the residuals
        check_resid_mul!(resid1, geee.tau[j])

        # The conditional standard deviations of the observations
        sd = sqrt.(geee.varfunc.(linpred1))

        # Update the score function
        update_score!(geee.pp, i1, i2, geee.cor[j], linpred1, sd, resid1, 1.0, scr)

        # The conditional standard deviations of the observations
        sd = sqrt.(geee.varfunc.(linpred1))

        # Update the denominator for the parameter update
        update_denom!(geee.pp, i1, i2, geee.cor[j], linpred1, sd, cresid1, 1.0, denom)
    end
end

# Apply the check function in-place to v.
function check!(v::AbstractVector{T}, tau::Float64) where {T<:Real}
    for j in eachindex(v)
        if v[j] < 0
            v[j] = 1 - tau
        else
            v[j] = tau
        end
    end
end

# Update the parameter estimates for the j^th expectile.
function update!(geee::GEEE, j::Int, upcor::Bool)::Float64
    p, n = size(geee.pp.X)
    score = zeros(p)
    denom = zeros(p, p)
    linpred = zeros(n)
    resid = zeros(n)
    score!(geee, j, score, denom, linpred, resid)
    step = denom \ score
    geee.beta[:, j] .+= step
    sresid = resid ./ sqrt.(geee.varfunc.(linpred))
    if upcor
        updatecor(geee.cor[j], sresid, geee.grpix, p)
    end
    return norm(step)
end

# Estimate the coefficients for the j^th expectile.
function fit_tau!(geee::GEEE, j::Int; maxiter::Int = 100, tol::Real = 1e-6)

    # Fit without covariance updates.
    for itr in 1:maxiter
        ss = update!(geee, j, false)
        if ss < tol
            break
        end
    end

    # Fit with covariance updates.
    for itr = 1:maxiter
        ss = update!(geee, j, true)
        if ss < tol
            break
        end
    end
end

# Calculate the robust covariance matrix for the parameter estimates.
function set_vcov!(geee::GEEE)

    # Number of covariates
    p, n = size(geee.pp.X)

    # Number of expectiles being jointly estimated
    q = length(geee.tau)

    # Factors in the covariance matrix
    D1 = [zeros(p, p) for _ in eachindex(geee.tau)]
    D0 = zeros(p * q, p * q)

    # Get D1 and D0, factors of the covariance matrix
    linpred0 = zeros(n, q)
    resid0 = zeros(n, q)
    cresid0 = zeros(n, q)
    sd0 = zeros(n, q)
    vv = zeros(p * q)
    for (g, (i1, i2)) in enumerate(eachcol(geee.grpix))

        # Get the residual and checked residual vector for all tau values.
        vv .= 0
        for j = 1:q
            resid = @view resid0[i1:i2, j]
            cresid = @view cresid0[i1:i2, j]
            linpred = @view linpred0[i1:i2, j]
            sd = @view sd0[i1:i2, j]

            # The residuals
            linpred!(geee, j, g, linpred)
            resid .= geee.rr.y[i1:i2] - linpred

            # The checked residuals
            cresid .= resid
            check!(cresid, geee.tau[j])

            # The product of the checked residuals and the residuals
            check_resid_mul!(resid, geee.tau[j])

            # The conditional standard deviations
            sd .= sqrt.(geee.varfunc.(linpred))

            # Update D1
            update_denom!(
                geee.pp,
                i1,
                i2,
                geee.cor[j],
                linpred,
                sd,
                cresid,
                geee.tau_wt[j],
                D1[j],
            )

            # Update D0
            jj = (j - 1) * p
            update_score!(
                geee.pp,
                i1,
                i2,
                geee.cor[j],
                linpred,
                sd,
                resid,
                geee.tau_wt[j],
                @view(vv[jj+1:jj+p])
            )

        end
        D0 .+= vv * vv'
    end

    # Normalize the block for each expectile by the sample size
    D0 ./= n
    n = length(geee.rr.y)
    for j = 1:q
        D1[j] ./= n
    end

    vcov = BlockDiagonal(D1) \ D0 / BlockDiagonal(D1)
    vcov ./= n
    geee.vcov = vcov
end

function dispersion(geee::GEEE, tauj::Int)::Float64

    # The dispersion parameter estimate
    sig2 = 0.0

    # Backing storage
    resid_b = zeros(geee.mxgrp)
    linpred_b = zeros(geee.mxgrp)

    # Loop over groups
    for (g, (i1, i2)) in enumerate(eachcol(geee.grpix))
        # Size of current group
        gs = i2 - i1 + 1

        # Get the residual vector for this group
        resid = @view resid_b[1:gs]
        linpred = @view linpred_b[1:gs]
        linpred!(geee, tauj, g, linpred)
        resid .= geee.y[i1:i2] - linpred
        check_resid_mul!(resid, geee.tau[tauj])
        sig2 += sum(abs2, resid)
    end

    p, n = size(geee.X)
    sig2 /= (n - p)
    return sig2
end

function startingvalues(pp::GEEEDensePred{T}, m::Int, y::Vector{T}) where {T<:Real}
    u, s, v = svd(pp.X')
    b = v * diagm(s) \ u' * y
    c = hcat([b for _ in 1:m]...)
    return c
end

function StatsBase.fit!(geee::GEEE)

    geee.beta .= startingvalues(geee.pp, length(geee.tau), geee.rr.y)

    # Fit all coefficients
    for j in eachindex(geee.tau)
        fit_tau!(geee, j)
    end

    set_vcov!(geee)

    return geee
end

function fit(
    ::Type{GEEE},
    X::AbstractMatrix{T},
    y::AbstractVector{T},
    g::AbstractVector,
    tau::Vector{Float64},
    c::CorStruct = IndependenceCor();
    dofit::Bool = true,
    fitargs...,
) where {T<:Real}

    c = CorStruct[copy(c) for _ in eachindex(tau)]
    geee = GEEE(y, copy(X'), g, tau; cor = c)

    return dofit ? fit!(geee) : geee
end

geee(F, D, args...; kwargs...) = fit(GEEE, F, D, args...; kwargs...)

function StatsBase.coef(m::GEEE)
    return m.beta[:]
end

function StatsBase.vcov(m::GEEE)
    return m.vcov
end

function StatsBase.coefnames(
    m::StatsModels.TableRegressionModel{GEEE{T},Matrix{T}},
) where {T}
    return repeat(coefnames(m.mf), length(m.model.tau))
end

function StatsBase.coeftable(
    m::StatsModels.TableRegressionModel{GEEE{T},Matrix{T}},
) where {T}
    ct = coeftable(m.model)
    ct.rownms = coefnames(m)
    return ct
end

function StatsBase.coeftable(mm::GEEE; level::Real = 0.95)
    cc = coef(mm)
    se = sqrt.(diag(mm.vcov))
    zz = cc ./ se
    p = 2 * ccdf.(Ref(Normal()), abs.(zz))
    ci = se * quantile(Normal(), (1 - level) / 2)
    levstr = isinteger(level * 100) ? string(Integer(level * 100)) : string(level * 100)
    na = ["x$i" for i = 1:size(mm.pp.X, 1)]
    q = length(mm.tau)
    na = repeat(na, q)
    tau = kron(mm.tau, ones(size(mm.pp.X, 1)))
    CoefTable(
        hcat(tau, cc, se, zz, p, cc + ci, cc - ci),
        ["tau", "Coef.", "Std. Error", "z", "Pr(>|z|)", "Lower $levstr%", "Upper $levstr%"],
        na,
        5,
        4,
    )
end

corparams(m::StatsModels.TableRegressionModel) = corparams(m.model)
corparams(m::GEEE) = [corparams(c) for c in m.cor]
