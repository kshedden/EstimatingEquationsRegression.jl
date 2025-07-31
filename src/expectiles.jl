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

    # Each column is the linear predictor for one expectile
    linpred::Matrix{T}

    # Each column contains the residuals for one expectile
    resid::Matrix{T}

    # Each column contains the checked residuals for one expectile
    cresid::Matrix{T}

    # Each column contains the product of the residual and checked
    # residual for one expectile
    cresidx::Matrix{T}

    # Each column contains the standard deviations for one expectile
    sd::Matrix{T}
end

abstract type GEEELinPred <: LinPred end

# TODO: make this universal for this package
struct GEEEDensePred{T<:Real} <: GEEELinPred

    # The number of observations
    n::Int

    # The number of covariates
    p::Int

    # Each column contains the first and last index of a group.
    gix::Matrix{Int}

    # n x p covariate matrix, observations are rows, variables are columns
    X::Matrix{T}
end

# Compute the product X * v where X is the design matrix.
function xtv(pp::GEEEDensePred, rhs::T) where {T<:AbstractArray}
    return pp.X * rhs
end

# Compute the product X * v where X is the submatrix of the design matrix
# containing data for group 'g'.
function xtvg(pp::GEEEDensePred, g::Int, rhs::T) where {T<:AbstractArray}
    i1, i2 = pp.gix[:, g]
    return pp.X[i1:i2, :]' * rhs
end

"""
    GEEE

Fit expectile regression models using GEE.
"""
mutable struct GEEE{T<:Real,L<:LinPred} <: AbstractGEE

    # The response
    rr::GEEEResp{T}

    # The covariates
    pp::L

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
    varfunc::Varfunc

    # The variance/covariance matrix of the parameter estimates
    vcov::Matrix{Float64}

    # The size of the biggest group.
    mxgrp::Int

    # Was the model fit and converged
    converged::Vector{Bool}
end

function update_score_group!(
    pp::T,
    g::Int,
    cor::CorStruct,
    linpred::AbstractVector,
    sd::AbstractVector,
    resid::AbstractVector,
    f,
    scr,
) where {T<:LinPred}
    scr .+= xtvg(pp, g, f * covsolve(cor, linpred, sd, resid))
end

function update_denom_group!(
    pp::T,
    g::Int,
    cor::CorStruct,
    linpred::AbstractVector,
    sd::AbstractVector,
    cresid::AbstractVector,
    f,
    denom,
) where {T<:LinPred}
    u = xtvg(pp, g, Diagonal(cresid))
    denom .+= xtvg(pp, g, f * covsolve(cor, linpred, sd, u'))
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
    @assert length(y) == size(X, 1) == length(grp)

    if !issorted(grp)
        error("Group vector is not sorted")
    end
    gix, mx = groupix(grp)

    if isnothing(tau_wt)
        tau_wt = ones(length(tau))
    end

    # Default variance function
    if isnothing(varfunc)
        varfunc = ConstantVar()
    end

    # Number of expectiles
    q = length(tau)

    # Number of observations
    n = size(X, 1)

    # Number of covariates
    p = size(X, 2)

    # Each column contains the covariates at an expectile
    beta = zeros(p, q)

    # The variance/covariance matrix of all parameters
    vcov = zeros(p * q, p * q)

    T = eltype(y)
    rr = GEEEResp(
        y,
        grp,
        zeros(T, length(y), length(tau)),
        zeros(T, length(y), length(tau)),
        zeros(T, length(y), length(tau)),
        zeros(T, length(y), length(tau)),
        zeros(T, length(y), length(tau)),
    )
    pp = GEEEDensePred(n, p, gix, X)

    return GEEE(
        rr,
        pp,
        gix,
        tau,
        tau_wt,
        beta,
        cor,
        varfunc,
        vcov,
        mx,
        zeros(Bool, length(tau)),
    )
end

# Place the linear predictor for the j^th tau value into linpred.
function linpred!(geee::GEEE, j::Int)
    geee.rr.linpred[:, j] .= xtv(geee.pp, geee.beta[:, j])
end

function iterprep!(geee::GEEE, j::Int)

    # Update the linear predictor for the j'th expectile.
    linpred!(geee, j)

    # Update the residuals for the j'th expectile
    geee.rr.resid[:, j] .= geee.rr.y - geee.rr.linpred[:, j]

    # Update the checked residuals for the j'th expectile
    geee.rr.cresid[:, j] .= geee.rr.resid[:, j]
    check!(@view(geee.rr.cresid[:, j]), geee.tau[j])

    # Update the products of the residuals and checked residuals for the j'th expectile
    geee.rr.cresidx[:, j] .= geee.rr.resid[:, j] .* geee.rr.cresid[:, j]

    # Analytic weights not yet implemented TODO
    awts = ones(size(geee.rr.resid, 1))

    # Update the conditional standard deviations for the j'th expectile
    geee.rr.sd[:, j] .= sqrt.(geevar.(NoDistribution(), geee.varfunc, geee.rr.linpred[:, j], awts))
end

# Place the score and denominator for the jth expectile into 'scr' and 'denom'.
function score!(geee::GEEE, j::Int, scr::Vector{T}, denom::Matrix{T}) where {T<:Real}

    scr .= 0
    denom .= 0
    iterprep!(geee, j)

    # Loop over the groups
    for (g, (i1, i2)) in enumerate(eachcol(geee.grpix))

        # Quantities for the current group
        linpred1 = @view geee.rr.linpred[i1:i2, j]
        cresid1 = @view geee.rr.cresid[i1:i2, j]
        cresidx1 = @view geee.rr.cresidx[i1:i2, j]
        sd1 = @view geee.rr.sd[i1:i2, j]

        # Update the score function
        update_score_group!(geee.pp, g, geee.cor[j], linpred1, sd1, cresidx1, 1.0, scr)

        # Update the denominator for the parameter update
        update_denom_group!(geee.pp, g, geee.cor[j], linpred1, sd1, cresid1, 1.0, denom)
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

# Update the parameter estimates for the j^th expectile.  If 'upcor' is true,
# first update the correlation parameters.
function update_params!(geee::GEEE, j::Int, upcor::Bool)::Float64

    if upcor
        p = length(geee.beta)
        iterprep!(geee, j)
        sresid = geee.rr.resid[:, j] ./ geee.rr.sd[:, j]
        updatecor(geee.cor[j], sresid, geee.grpix, p)
    end

    p = geee.pp.p
    score = zeros(p)
    denom = zeros(p, p)
    score!(geee, j, score, denom)
    step = denom \ score
    geee.beta[:, j] .+= step

    return norm(step)
end

# Estimate the coefficients for the j^th expectile.
function fit_tau!(
    geee::GEEE,
    j::Int;
    maxiter::Int = 100,
    tol::Real = 1e-8,
    updatecor::Bool = true,
    fitargs...,
)

    # Fit with covariance updates.
    for itr = 1:maxiter

        # Let the parameters catch up
        ss = 0.0
        for itr1 = 1:maxiter
            ss = update_params!(geee, j, false)
            if ss < tol
                break
            end
        end

        if !updatecor
            # Don't update the correlation parameters
            if ss < tol
                geee.converged[j] = true
            end
            return
        end

        ss = update_params!(geee, j, true)
        if ss < tol
            geee.converged[j] = true
            return
        end
    end
end

# Calculate the robust covariance matrix for the parameter estimates.
function set_vcov!(geee::GEEE)

    # Number of covariates
    (; n, p) = geee.pp

    # Number of expectiles being jointly estimated
    q = length(geee.tau)

    for j = 1:q
        iterprep!(geee, j)
    end

    # Factors in the covariance matrix
    D1 = [zeros(p, p) for _ in eachindex(geee.tau)]
    D0 = zeros(p * q, p * q)

    vv = zeros(p * q)
    for (g, (i1, i2)) in enumerate(eachcol(geee.grpix))

        vv .= 0
        for j = 1:q
            linpred = @view geee.rr.linpred[i1:i2, j]
            sd = @view geee.rr.sd[i1:i2, j]
            cresid = @view geee.rr.cresid[i1:i2, j]
            cresidx = @view geee.rr.cresidx[i1:i2, j]

            # Update D1
            update_denom_group!(
                geee.pp,
                g,
                geee.cor[j],
                linpred,
                sd,
                cresid,
                geee.tau_wt[j],
                D1[j],
            )

            jj = (j - 1) * p
            update_score_group!(
                geee.pp,
                g,
                geee.cor[j],
                linpred,
                sd,
                cresidx,
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

function GLM.dispersion(geee::GEEE, tauj::Int)::Float64

    n, p = size(geee.pp.X)
    iterprep!(geee, tauj)

    # The dispersion parameter estimate
    sig2 = sum(abs2, geee.rr.cresidx ./ geee.rr.sd)
    sig2 /= (n - p)
    return sig2
end

function startingvalues(pp::GEEEDensePred{T}, m::Int, y::Vector{T}) where {T<:Real}
    u, s, v = svd(pp.X)
    b = v * (Diagonal(s) \ (u' * y))
    c = hcat([b for _ = 1:m]...)
    return c
end

function fit!(geee::GEEE; fitargs...)

    geee.beta .= startingvalues(geee.pp, length(geee.tau), geee.rr.y)

    # Fit all coefficients
    for j in eachindex(geee.tau)
        fit_tau!(geee, j; fitargs...)
    end

    if !all(geee.converged)
        @warn("One or more expectile GEE models did not converge")
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
    geee = GEEE(y, X, g, tau; cor = c)

    return dofit ? fit!(geee; fitargs...) : geee
end

geee(F, D, args...; kwargs...) = fit(GEEE, F, D, args...; kwargs...)

function coef(m::GEEE)
    return m.beta[:]
end

function vcov(m::GEEE)
    return m.vcov
end

function coefnames(m::StatsModels.TableRegressionModel{GEEE{S, GEEEDensePred{S}}, Matrix{S}}) where {S}
    return repeat(coefnames(m.mf), length(m.model.tau))
end

function coeftable(m::StatsModels.TableRegressionModel{GEEE{S, GEEEDensePred{S}}, Matrix{S}}) where {S}
    ct = coeftable(m.model)
    ct.rownms = coefnames(m)
    return ct
end

function coeftable(mm::GEEE; level::Real = 0.95)
    cc = coef(mm)
    se = sqrt.(diag(mm.vcov))
    zz = cc ./ se
    p = 2 * ccdf.(Ref(Normal()), abs.(zz))
    ci = se * quantile(Normal(), (1 - level) / 2)
    levstr = isinteger(level * 100) ? string(Integer(level * 100)) : string(level * 100)
    na = ["x$i" for i = 1:size(mm.pp.X, 2)]
    q = length(mm.tau)
    na = repeat(na, q)
    tau = kron(mm.tau, ones(size(mm.pp.X, 2)))
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

function predict(mm::GEEE, newX::AbstractMatrix; tauj::Int = 1)
    p = mm.pp.p
    jj = p * (tauj - 1)
    cf = coef(mm)[jj+1:jj+p]
    vc = vcov(mm)[jj+1:jj+p, jj+1:jj+p]
    eta = newX * cf
    va = newX * vc * newX'
    sd = sqrt.(diag(va))
    return (prediction = eta, lower = eta - 2 * sd, upper = eta + 2 * sd)
end

function predict(mm::GEEE; tauj::Int = 1)
    p = mm.pp.p
    jj = p * (tauj - 1)
    cf = coef(mm)[jj+1:jj+p]
    vc = vcov(mm)[jj+1:jj+p, jj+1:jj+p]
    eta = xtv(mm.pp, cf)
    va = xtv(mm.pp, vc)
    va = xtv(mm.pp, va')
    sd = sqrt.(diag(va))
    return (prediction = eta, lower = eta - 2 * sd, upper = eta + 2 * sd)
end

residuals(rr::GEEEResp) = rr.y - rr.mu
