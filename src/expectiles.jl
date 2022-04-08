using LinearAlgebra

mutable struct GEEE{T<:Real} <: AbstractGEE

    # n-dimensional vector of responses
    y::AbstractVector{T}

    # p x n covariate matrix, variables are rows, cases are columns
    X::AbstractMatrix{T}

    # Group labels, sorted
    grp::AbstractVector

    # Each column contains the first and last index of a group.
    grpix::Matrix{Int}

    # Vector of tau values for which expectiles are jointly estimated
    tau::Vector{Float64}

    # Weight for each tau value
    tau_wt::Vector{Float64}

    # The parameters to be estimated
    beta::Matrix{Float64}

    # The working correlation model
    corstr::CorStruct

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
    corstr = IndependenceCor(),
    varfunc = nothing,
    tau_wt = nothing,
)

    @assert length(y) == size(X, 2) == length(grp)

    if !issorted(grp)
        error("Group vector is not sorted")
    end
    gi, mx = groupix(grp)

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

    return GEEE(y, X, grp, gi, tau, tau_wt, beta, corstr, varfunc, vcov, mx)
end

# Place the linear predictor for the j^th tau value and
# for group g into linpred.
function linpred!(geee::GEEE, tauj::Int, g::Int, linpred::AbstractVector{Float64})
    i1, i2 = geee.grpix[:, g]
    x = geee.X[:, i1:i2]
    linpred .= x' * geee.beta[:, tauj]
end

# Multiply the residuals in-place by their own check function.
function check_resid_mul!(resid::AbstractVector{T}, tau) where {T<:Real}
    for j in eachindex(resid)
        if resid[j] < 0
            resid[j] *= 1 - tau
        else
            resid[j] *= tau
        end
    end
end

# Place the score for the jth expectile into 'scr'.
function score!(geee::GEEE, j::Int, scr::Vector{T}) where {T<:Real}

    scr .= 0
    linpred0 = zeros(geee.mxgrp)
    resid0 = zeros(geee.mxgrp)
    p = size(geee.X, 1)
    grpix = geee.grpix

    # Loop over the groups
    for (g, (i1, i2)) in enumerate(eachcol(grpix))
        gs = i2 - i1 + 1
        linpred = @view linpred0[1:gs]
        resid = @view resid0[1:gs]
        x = @view geee.X[:, i1:i2]
        linpred!(geee, j, g, linpred)
        resid .= geee.y[i1:i2] - linpred
        check_resid_mul!(resid, geee.tau[j])
        sd = sqrt.(geee.varfunc.(linpred))
        scr .+= x * covsolve(geee.corstr, linpred, sd, zeros(0), resid)
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

# Calculate the denominator matrix used in the fitting iterations.
function denom!(geee::GEEE, j::Int, denom::Matrix{T}) where {T<:Real}
    p = size(geee.X, 1)
    tau = geee.tau[j]
    linpred0 = zeros(geee.mxgrp)
    resid0 = zeros(geee.mxgrp)
    grpix = geee.grpix

    # Loop over the groups to build up the denominator matrix
    for (g, (i1, i2)) in enumerate(eachcol(grpix))
        i1, i2 = grpix[:, g]
        gs = i2 - i1 + 1
        linpred = @view linpred0[1:gs]
        resid = @view resid0[1:gs]
        x = @view geee.X[:, i1:i2]
        linpred!(geee, j, g, linpred)
        resid .= geee.y[i1:i2] - linpred
        check!(resid, tau)
        sd = sqrt.(geee.varfunc.(linpred))
        denom .+= x * covsolve(geee.corstr, linpred, sd, zeros(0), diagm(resid)) * x'
    end
end

# Update the parameter estimates for the j^th expectile.
function update!(geee::GEEE, j::Int)::Float64
    p = size(geee.X, 1)
    score = zeros(p)
    denom = zeros(p, p)
    denom!(geee, j, denom)
    score!(geee, j, score)
    step = denom \ score
    geee.beta[:, j] .+= step
    return norm(step)
end

# Estimate the coefficients for the j^th expectile.
function fit_tau!(geee::GEEE, j::Int; maxiter = 100)
    for itr = 1:maxiter
        ss = update!(geee, j)
        if ss < 1e-6
            break
        end
    end
end

# Calculate the robust covariance matrix for the parameter estimates.
function set_vcov!(geee::GEEE)

    # Number of covariates
    p = size(geee.X, 1)

    # Number of expectiles being jointly estimated
    q = length(geee.tau)

    # Factors in the covariance matrix
    D1 = zeros(p * q, p * q)
    D0 = zeros(p * q, p * q)

    # Get D1 and D0, factors of the covariance matrix
    linpred0 = zeros(q * geee.mxgrp)
    resid0 = zeros(q * geee.mxgrp)
    cresid0 = zeros(q * geee.mxgrp)
    va0 = zeros(q * geee.mxgrp, q * geee.mxgrp)
    for (g, (i1, i2)) in enumerate(eachcol(geee.grpix))

        # Size of current group
        gs = i2 - i1 + 1

        # Get the checked residual vector for all tau values.
        resid = @view resid0[1:q*gs]
        cresid = @view cresid0[1:q*gs]
        linpred = @view linpred0[1:q*gs]
        va = @view va0[1:q*gs, 1:q*gs]
        va .= 0
        jj = 0
        for j in eachindex(geee.tau)

            # Get the residuals
            linpred!(geee, j, g, @view(linpred[jj+1:jj+gs]))
            resid[jj+1:jj+gs] .= geee.y[i1:i2] - linpred[jj+1:jj+gs]

            # Get the checked residuals
            cresid[jj+1:jj+gs] .= resid[jj+1:jj+gs]
            check!(@view(cresid[jj+1:jj+gs]), geee.tau[j])

            # The working covariance matrix, currently independent.  There
            # is no need to scale by sigma^2 since it would cancel out below.
            va[jj+1:jj+gs, jj+1:jj+gs] .= diagm(geee.varfunc.(linpred[jj+1:jj+gs]))

            jj += gs
        end

        # Update D1
        x = geee.X[:, i1:i2]
        xw = kron(diagm(geee.tau_wt), x')
        xi = kron(I(q), x')
        D1 .+= xw' * (va \ diagm(cresid)) * xi

        # Update D0
        jj = 0
        for j in eachindex(geee.tau)
            check_resid_mul!(@view(resid[jj+1:jj+gs]), geee.tau[j])
            jj += gs
        end
        v = xw' * (va \ resid)
        D0 .+= v * v'
    end

    # Total sample size
    n = length(geee.y)
    D0 ./= n
    D1 ./= n

    vcov = D1 \ D0 / D1
    vcov ./= n
    geee.vcov = vcov
end

function dispersion(geee::GEEE, tauj::Int)::Float64
    sig2 = 0.0
    resid_b = zeros(geee.mxgrp)
    linpred_b = zeros(geee.mxgrp)
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

function StatsBase.fit!(geee::GEEE)

    # Fit all coefficients
    for j in eachindex(geee.tau)
        fit_tau!(geee, j)
    end

    set_vcov!(geee)

    return geee
end

function fit(
    ::Type{GEEE},
    X::Matrix{T},
    y::AbstractVector{T},
    g::AbstractVector,
    tau::Vector{Float64},
    c::CorStruct = IndependenceCor();
    dofit::Bool = true,
    fitargs...,
) where {T<:Real}

    geee = GEEE(y, copy(X'), g, tau)

    return dofit ? fit!(geee) : geee
end

geee(F, D, args...; kwargs...) = fit(GEEE, F, D, args...; kwargs...)

function StatsBase.coef(m::GEEE)
    return m.beta[:]
end

function StatsBase.vcov(m::GEEE)
    return m.vcov
end

function StatsBase.coefnames(m::StatsModels.TableRegressionModel{GEEE{T},Matrix{T}}) where T
	return repeat(coefnames(m.mf), length(m.model.tau))	
end

function StatsBase.coeftable(m::StatsModels.TableRegressionModel{GEEE{T},Matrix{T}}) where T
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
    na = ["x$i" for i = 1:size(mm.X, 1)]
    q = length(mm.tau)
    na = repeat(na, q)
    tau = kron(mm.tau, ones(size(mm.X, 1)))
    CoefTable(
        hcat(tau, cc, se, zz, p, cc + ci, cc - ci),
        ["tau", "Coef.", "Std. Error", "z", "Pr(>|z|)", "Lower $levstr%", "Upper $levstr%"],
        na,
        5,
        4,
    )
end
