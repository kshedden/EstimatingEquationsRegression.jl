using Optim

"""
    QIFResp

n-dimensional vectors related to the QIF response variable.
"""
struct QIFResp{T<:Real} <: ModResp
    # The response data
    y::Vector{T}

    # The linear predictor
    eta::Vector{T}

    # The residuals
    resid::Vector{T}

    # The fitted mean
    mu::Vector{T}

    # The standard deviations
    sd::Vector{T}

    # The standardized residuals
    sresid::Vector{T}

    # The derivative of mu with respect to eta
    dmudeta::Vector{T}

    # The second derivative of mu with respect to eta
    d2mudeta2::Vector{T}
end

"""
    QIFLinPred

Represent a design matrix for QIF analysis.  The design matrix
is stored with the variables as rows and the observations as
columns.
"""
abstract type QIFLinPred{T} <: GLM.LinPred end

struct QIFDensePred{T<:Real} <: QIFLinPred{T}
    X::Matrix{T}
end

"""
    QIFBasis

A basis matrix for representing the inverse working correlation matrix.
"""
abstract type QIFBasis end

"""
    QIF

Quadratic Inference Function (QIF) is an approach to fitting marginal regression
models with correlated data.
"""
mutable struct QIF{T<:Real,L<:Link,V<:Varfunc} <: AbstractMarginalModel

    # The response y and related information
    rr::QIFResp{T}

    # The covariates X.
    pp::QIFLinPred{T}

    # The coefficients being estimated
    beta::Vector{T}

    # Each column contains the first and last index of one group
    gix::Matrix{Int}

    # The group labels, sorted
    grp::Vector

    # The link function
    link::L

    # The variance function
    varfunc::V

    # The empirical covariance of the score vectors
    scov::Matrix{T}

    # The basis vectors defining
    basis::Vector

    # True if the model was fit and converged
    converged::Bool
end


function QIFResp(y::Vector{T}) where {T<:Real}
    n = length(y)
    e = eltype(y)
    return QIFResp(
        y,
        zeros(e, n),
        zeros(e, n),
        zeros(e, n),
        zeros(e, n),
        zeros(e, n),
        zeros(e, n),
        zeros(e, n),
    )
end

# Multiply the design matrix for one group along the observations.
function rmul!(
    pp::QIFDensePred{T},
    v::Vector{T},
    r::Vector{T},
    i1::Int,
    i2::Int,
) where {T<:Real}
    r .+= pp.X[i1:i2, :] * v
end

# Multiply the design matrix for one group along the variables.
function lmul!(
    pp::QIFDensePred,
    v::Vector{T},
    r::AbstractVector{T},
    i1::Int,
    i2::Int,
) where {T<:Real}
    r .+= pp.X[i1:i2, :]' * v
end

struct QIFHollowBasis <: QIFBasis end

struct QIFSubdiagonalBasis <: QIFBasis
    d::Int
end

struct QIFIdentityBasis <: QIFBasis end

function rbasis(::QIFIdentityBasis, T::Type, d::Int)
    return I(d)
end

function rbasis(::QIFHollowBasis, T::Type, d::Int)
    return ones(T, d, d) - I(d)
end

function rbasis(b::QIFSubdiagonalBasis, T::Type, d::Int)
    m = zeros(T, d, d)
    j = 1
    for i = b.d:d-1
        m[i+1, j] = 1
        m[j, i+1] = 1
        j += 1
    end
    return m
end

function mueta2(::IdentityLink, eta::T) where {T<:Real}
    return zero(typeof(eta))
end

function mueta2(::LogLink, eta::T) where {T<:Real}
    return exp(eta)
end

# Weights are not implemented
function weights(qif::QIF{T}; type::Symbol=:analytic) where {T<:Real}
    n = length(qif.rr.mu)
    return ones(n)
end

# Update the linear predictor and related n-dimensional quantities
# that do not depend on thr grouping or correlation structures.
function iterprep!(qif::QIF{T}, beta::Vector{T}) where {T<:Real}
    qif.rr.eta .= 0
    rmul!(qif.pp, beta, qif.rr.eta, 1, length(qif.rr.eta))
    qif.rr.mu .= linkinv.(qif.link, qif.rr.eta)
    qif.rr.resid .= qif.rr.y - qif.rr.mu
    qif.rr.dmudeta .= mueta.(qif.link, qif.rr.eta)
    qif.rr.d2mudeta2 .= mueta2.(qif.link, qif.rr.eta)
    awts = weights(qif; type=:analytic)
    qif.rr.sd .= sqrt.(geevar.(NoDistribution(), qif.varfunc, qif.rr.mu, awts))
    qif.rr.sresid .= qif.rr.resid ./ qif.rr.sd
end

# Calculate the score for group 'g' and add it to the current value of 'scr'.
function score!(qif::QIF{T}, g::Int, scr::Vector{T}) where {T<:Real}
    i1, i2 = qif.gix[:, g]
    gs = i2 - i1 + 1
    p = length(qif.beta)
    sd = @view(qif.rr.sd[i1:i2])
    dmudeta = @view(qif.rr.dmudeta[i1:i2])
    sresid = @view(qif.rr.sresid[i1:i2])

    jj = 0
    for b in qif.basis
        rb = rbasis(b, T, gs)
        rhs = Diagonal(dmudeta ./ sd) * rb * sresid
        lmul!(qif.pp, rhs, @view(scr[jj+1:jj+p]), i1, i2)
        jj += p
    end
end

# Calculate the average score function.
function score!(qif::QIF{T}, scr::Vector{T}) where {T<:Real}
    ngrp = size(qif.gix, 2)
    scr .= 0
    for g = 1:ngrp
        score!(qif, g, scr)
    end

    scr ./= ngrp
end

# Calculate the Jacobian of the score function for group 'g' and add it to
# the current value of 'scd'.
function scorederiv!(qif::QIF{T}, g::Int, scd::Matrix{T}) where {T<:Real}
    i1, i2 = qif.gix[:, g]
    p = length(qif.beta)
    gs = i2 - i1 + 1
    sd = @view(qif.rr.sd[i1:i2])
    awts = weights(qif; type=:analytic)
    vd = geevarderiv.(NoDistribution(), qif.varfunc, qif.rr.mu[i1:i2], awts[i1:i2])
    dmudeta = @view(qif.rr.dmudeta[i1:i2])
    d2mudeta2 = @view(qif.rr.d2mudeta2[i1:i2])
    sresid = @view(qif.rr.sresid[i1:i2])

    x = qif.pp.X[i1:i2, :]

    jj = 0
    for b in qif.basis
        rb = rbasis(b, T, gs)
        for j = 1:p
            scd[jj+1:jj+p, j] .+= x' * Diagonal(d2mudeta2 .* x[:, j] ./ sd) * rb * sresid
            scd[jj+1:jj+p, j] .-=
                0.5 * x' * Diagonal(dmudeta .^ 2 .* vd .* x[:, j] ./ sd .^ 3) * rb * sresid
            scd[jj+1:jj+p, j] .-=
                0.5 *
                x' *
                Diagonal(dmudeta ./ sd) *
                rb *
                (vd .* dmudeta .* x[:, j] .* sresid ./ sd .^ 2)
            scd[jj+1:jj+p, j] .-=
                x' * Diagonal(dmudeta ./ sd) * rb * (dmudeta .* x[:, j] ./ sd)
        end
        jj += p
    end
end

# Calculate the Jacobian of the average score function.
function scorederiv!(qif::QIF{T}, scd::Matrix{T}) where {T<:Real}
    ngrp = size(qif.gix, 2)
    scd .= 0
    for g = 1:ngrp
        scorederiv!(qif, g, scd)
    end
    scd ./= ngrp
end

function iterate!(
    qif::QIF{T};
    gtol::Float64 = 1e-4,
    verbose::Bool = false,
)::Bool where {T<:Real}

    p = length(qif.beta)

    ngrp = size(qif.gix, 2)
    m = p * length(qif.basis)
    scr = zeros(m)
    scd = zeros(m, p)

    # Get the search direction in the beta space
    iterprep!(qif, qif.beta)
    score!(qif, scr)
    scorederiv!(qif, scd)
    grad = scd' * (qif.scov \ scr) / ngrp

    if norm(grad) < gtol
        if verbose
            println(@sprintf("Final |grad|=%f", norm(grad)))
        end
        return true
    end

    f0 = scr' * (qif.scov \ scr) / ngrp
    beta0 = copy(qif.beta)

    step = 1.0
    while true
        beta = beta0 - step * grad
        iterprep!(qif, beta)
        score!(qif, scr)
        f1 = scr' * (qif.scov \ scr) / ngrp
        if f1 < f0
            qif.beta .= beta
            break
        end
        step /= 2
        if step < 1e-14
            println("Failed to find downhill step")
            break
        end
    end

    if verbose
        println(@sprintf("Current |grad|=%f", norm(grad)))
    end
    return false
end

function updateCov!(qif::QIF)
    ngrp = size(qif.gix, 2)
    qif.scov .= 0
    p = length(qif.beta)
    nb = length(qif.basis)
    scr = zeros(p * nb)
    iterprep!(qif, qif.beta)
    for g = 1:ngrp
        scr .= 0
        score!(qif, g, scr)
        qif.scov .+= scr * scr'
    end
    qif.scov ./= ngrp
end

function get_fungrad(qif::QIF, scov::Matrix)

    p = length(qif.beta)      # number of parameters
    m = p * length(qif.basis) # number of score equations
    ngrp = size(qif.gix, 2)   # number of groups

    # Objective function to minimize.
    fun = function (beta)
        iterprep!(qif, beta)
        scr = zeros(m)
        score!(qif, scr)
        return scr' * (scov \ scr) / ngrp
    end

    grad! = function (G, beta)
        iterprep!(qif, beta)
        scr = zeros(m)
        score!(qif, scr)
        scd = zeros(m, p)
        scorederiv!(qif, scd)
        G .= 2 * scd' * (scov \ scr) / ngrp
    end

    return fun, grad!
end

function fitbeta!(qif::QIF, start; verbose::Bool = false, g_tol = 1e-5)

    fun, grad! = get_fungrad(qif, qif.scov)

    opts = Optim.Options(show_trace = verbose, g_tol = g_tol)
    r = optimize(fun, grad!, start, LBFGS(), opts)

    if !Optim.converged(r)
        @warn("fitbeta did not converged")
    end

    qif.beta = Optim.minimizer(r)
    return Optim.converged(r)
end

function fit!(
    qif::QIF;
    g_tol::Float64 = 1e-5,
    verbose::Bool = false,
    maxiter::Int = 5,
)
    start = zeros(length(qif.beta))
    cnv = false
    for k = 1:maxiter
        if verbose
            println(@sprintf("=== Outer iteration %d:", k))
        end
        cnv = fitbeta!(qif, start; verbose = verbose, g_tol = g_tol)
        updateCov!(qif)
    end

    qif.converged = cnv

    return qif
end

function fit(
    ::Type{QIF},
    X::AbstractMatrix,
    y::AbstractVector,
    grp::AbstractVector;
    basis::AbstractVector = QIFBasis[QIFIdentityBasis()],
    link::Link = IdentityLink(),
    varfunc::Varfunc = ConstantVar(),
    verbose::Bool = false,
    dofit::Bool = true,
    start = nothing,
    kwargs...,
)
    p = size(X, 2)
    if length(y) != size(X, 1) != length(grp)
        msg = @sprintf(
            "Length of 'y' (%d) and length of 'grp' (%d) must equal the number of rows in 'X' (%d)\n",
            length(y),
            length(grp),
            size(X, 1)
        )
        @error(msg)
    end

    # TODO maybe a better way to do this
    T = promote_type(eltype(y), eltype(X))
    if eltype(y) != T
        y = T.(y)
    end
    if eltype(X) != T
        X = T.(X)
    end

    gix, mxgrp = groupix(grp)
    rr = QIFResp(y)
    pp = QIFDensePred(X)
    q = length(basis)
    m = QIF(
        rr,
        pp,
        zeros(p),
        gix,
        grp,
        link,
        varfunc,
        Matrix{Float64}(I(p * q)),
        basis,
        false,
    )

    if !isnothing(start)
        m.beta .= start
    end
    return dofit ? fit!(m; verbose = verbose, kwargs...) : m
end

"""
    qif(F, D, args...; kwargs...)
Fit a generalized linear model to data using quadratic inference functions.
Alias for `fit(QIF, ...)`.
See [`fit`](@ref) for documentation.
"""
qif(F, D, args...; kwargs...) = fit(QIF, F, D, args...; kwargs...)

function coef(m::QIF)
    return m.beta
end

function coefnames(m::QIF)
    p = length(coef(m))
    xn = ["x$(j)" for j in 1:p]
    return xn
end

function vcov(m::QIF)
    p = length(m.beta)
    q = length(m.basis)
    scd = zeros(p * q, p)
    ngrp = size(m.gix, 2)
    scorederiv!(m, scd)
    return inv(scd' * (m.scov \ scd)) / ngrp
end

function coeftable(mm::QIF; level::Real = 0.95)
    cc = coef(mm)
    se = sqrt.(diag(vcov(mm)))
    zz = cc ./ se
    pv = 2 * ccdf.(Ref(Normal()), abs.(zz))
    ci = se * quantile(Normal(), (1 - level) / 2)
    levstr = isinteger(level * 100) ? string(Integer(level * 100)) : string(level * 100)
    na = ["x$i" for i in eachindex(mm.beta)]
    CoefTable(
        hcat(cc, se, zz, pv, cc + ci, cc - ci),
        ["Coef.", "Std. Error", "z", "Pr(>|z|)", "Lower $levstr%", "Upper $levstr%"],
        na,
        4,
        3,
    )
end
