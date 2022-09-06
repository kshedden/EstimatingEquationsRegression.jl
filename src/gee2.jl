"""
    GEE2Resp

The response vector and group labels for GEE2 analysis.  All
vectors here are n-dimensional and aligned with `y`.
"""
struct GEE2Resp{T<:Real} <: ModResp

    # n-dimensional vector of responses
    y::Vector{T}

    # The means
    mu::Vector{T}

    # The linear predictors
    eta::Vector{T}

    # The residuals
    resid::Vector{T}

    # The standard deviations
    sd::Vector{T}

    # The derivative of mu with respect to linear eta
    dmudeta1::Vector{T}

    # The derivative of mu with respect to square eta
    dmudeta2::Vector{T}

    # Group labels, sorted
    grp::Vector

    # 2 x n matrix, each column contains the first and last index of a group
    gix::Matrix{Int}
end

"""
    GEE2

Fit regression models using GEE2.
"""
mutable struct GEE2{T<:Real,L<:Link,P<:LinPred,V1<:Varfunc,V2<:Varfunc} <: AbstractGEE

    # Response information for y, centered y^2, and standardized y*y products
    rr_lin::GEE2Resp{T}
    rr_sqr::GEE2Resp{T}
    rr_prod::GEE2Resp{T}

    # Linear predictors for y, centered y^2, and standardized y*y products
    pp_lin::P
    pp_sqr::P
    pp_prod::P

    # Map linear predictor to conditional mean
    link::L

    # Map conditional mean of linear terms to conditional variance of linear terms
    varfunc::V1

    # Map conditional mean of square terms to conditional variance of square terms
    varfunc2::V2

    # The working correlation model, cor[1] is the working correlation model
    # for the linear parameters, cor[2] is the working correlation model
    # for the squared parameters.
    cor::Vector{CorStruct}

    # The parameters to be estimated
    beta::Vector{Float64}

    # The variance/covariance matrix of the parameter estimates
    vcov::Matrix{Float64}

    # The size of the biggest group.
    mxgrp::Int
end

function GEE2Resp(y, grp, gix)
    n = length(y)
    return GEE2Resp(y, zeros(n), zeros(n), zeros(n), zeros(n), zeros(n), zeros(n), grp, gix)
end

function update_eta!(pp::P, eta::Vector{T}, beta::Vector{T}) where {P<:LinPred,T<:Real}
    eta .= pp.X * beta
end

function iterprep_lin!(gee::GEE2, beta::Vector{Vector{T}}) where {T<:Real}
    update_eta!(gee.pp_lin, gee.rr_lin.eta, beta[1])
    gee.rr_lin.mu .= linkinv.(gee.link, gee.rr_lin.eta)
    gee.rr_lin.dmudeta1 .= mueta.(gee.link, gee.rr_lin.eta)
    gee.rr_lin.resid .= gee.rr_lin.y - gee.rr_lin.mu
    gee.rr_lin.sd .= geevar.(gee.varfunc, gee.rr_lin.mu)
end

function iterprep_sqr!(gee::GEE2, beta::Vector{Vector{T}}) where {T<:Real}
    update_eta!(gee.pp_sqr, gee.rr_sqr.eta, beta[2])
    gv = geevar.(gee.varfunc, gee.rr_lin.mu)
    ee = exp.(gee.rr_sqr.eta)
    gee.rr_sqr.mu .= gv .* ee + gee.rr_lin.mu .^ 2

    gee.rr_sqr.dmudeta1 .=
        geevarderiv.(gee.varfunc, gee.rr_lin.mu) .* gee.rr_lin.mu .* ee +
        gee.rr_lin.mu .* gee.rr_lin.dmudeta1
    gee.rr_sqr.dmudeta2 .= gv .* ee .* gee.rr_sqr.eta

    gee.rr_sqr.resid .= gee.rr_sqr.y - gee.rr_sqr.mu
    gee.rr_sqr.sd .= geevar.(gee.varfunc2, gee.rr_sqr.mu)
end

function updateDeriv!(
    gee::GEE2,
    pp::LinPred,
    g::Int,
    beta::Vector{Vector{T}},
    D::Matrix{T},
) where {T<:Real}

    D .= 0
    p1, p2, p3 = length(beta[1]), length(beta[2]), length(bega[3])
    i1, i2 = pp.gix[:, g]
    gs = i2 - i1 + 1

    # Derivative of the linear terms with respect to linear parameters
    D[1:p1, 1:gs] .= hmul(gee.pp_lin, i1, i2, gee.rr_lin.dmudeta1[i1:i2])

    # Derivatives of the squared terms with respect to linear parameters
    D[1:p1, gs+1:2*gs] .= hmul(gee.pp_lin, i1, i2, gee.rr_sqr.dmudeta1[i1:i2])

    # Derivatives of the squared terms with respect to squared parameters
    D[p1+1:p1+p2, gs+1:2*gs] .+= hmul(gee.pp_sqr, i1, i2, gee.rr_sqr.dmudeta2[i1:i2])
end

function iterprep!(gee::GEE2, beta::Vector{Vector{T}}) where {T<:Real}
    iterprep_lin!(gee, beta)
    iterprep_sqr!(gee, beta)
    #    iterprep_prod!(gee, beta)
end

function update!(gee::GEE2, beta::Vector{Vector{T}}) where {T<:Real}

    p1, p2 = length(beta[1]), length(beta[2])
    iterprep!(gee, beta)
    score = zeros(p1 + p2)
    denom = zeros(p1 + p2, p1 + p2)
    for g in 1:size(gee.rr_lin.gix, 2)
        i1, i2 = gee.rr_lin.gix[:, g]
        gs = i2 - i1 + 1
        gsc = div(gs * (gs - 1), 2)
        D = zeros(p1 + p2, 2 * gs + gsc)
        vir = zeros(2 * gs + gsc)
        updateDeriv!(gee, gee.pp_lin, g, beta, D)
        w = zeros(0)
        vir[1:gs] = covsolve(
            gee.cor[1],
            gee.rr_lin.mu[i1:i2],
            gee.rr_lin.sd[i1:i2],
            w,
            gee.rr_lin.resid[i1:i2],
        )
        vir[gs+1:2*gs] = covsolve(
            gee.cor[2],
            gee.rr_sqr.mu[i1:i2],
            gee.rr_sqr.sd[i1:i2],
            w,
            gee.rr_sqr.resid[i1:i2],
        )
        score .+= D * vir

        vid = zeros(2 * gs, p1 + p2)
        vid[1:gs, :] =
            covsolve(gee.cor[1], gee.rr_lin.mu[i1:i2], gee.rr_lin.sd[i1:i2], w, D[:, 1:gs]')
        vid[gs+1:2*gs, :] = covsolve(
            gee.cor[1],
            gee.rr_lin.mu[i1:i2],
            gee.rr_lin.sd[i1:i2],
            w,
            D[:, gs+1:2*gs]',
        )
        denom .+= D * vid
    end

    u = denom \ score
    beta[1] .+= u[1:p1]
    beta[2] .+= u[p1+1:p1+p2]

end

# Expand y to include all distinct products of elements within the same group,
# and expand X accordingly to include the absolute difference and sum of the
# covariate vectors for each pair.
function expand_cor(y::Vector, X::AbstractMatrix, grp::Vector, gix::Matrix{Int})
    # Calculate the sample size after expanding pairs
    n = 0
    for (i1, i2) in eachcol(gix)
        m = i2 - i1 + 1
        n += div(m * (m - 1), 2)
    end

    yy = zeros(eltype(y), n)
    q = size(X, 2)
    XX = zeros(eltype(X), n, 2 * q)
    grp2 = zeros(eltype(grp), n)
    gix2 = zeros(Int, size(gix)...)

    ii = 1
    for (j, (i1, i2)) in enumerate(eachcol(gix))
        # Group size
        m = i2 - i1 + 1

        if m == 1
            # No products for groups of size 1.
            gix[:, j] .= -1
            continue
        end

        # Expanded group size
        mx = div(m * (m - 1), 2)

        # Expand the group labels and indices
        grp2[ii:ii+mx-1] .= grp[i1]
        gix2[:, j] = [ii, ii + mx - 1]

        # Products of pairs of observations
        for j1 = 1:m
            for j2 = 1:j1-1
                yy[ii] = y[i1+j1-1] * y[i1+j2-1]
                XX[ii, 1:q] = abs.(X[i1+j1-1, :] - X[i1+j2-1, :])
                XX[ii, q+1:end] = (X[i1+j1-1, :] + X[i1+j2-1, :]) / 2
                ii += 1
            end
        end
    end

    return yy, XX, grp2, gix2
end

function expand_y!(gee::GEE2, beta)

    gee.rr_sqr.y .= (gee.rr_lin.y - gee.rr_lin.mu).^2

    ii = 1
    yy = gee.rr_prod.y
    mu1 = gee.rr_lin.mu
    va1 = gee.rr_sqr.mu
    for (j, (i1, i2)) in enumerate(eachcol(gee.rr_lin.gix))
        # Group size
        m = i2 - i1 + 1

        # Expanded group size
        mx = div(m * (m - 1), 2)

        # Products of pairs of observations
        for j1 = 1:m
            for j2 = 1:j1-1
                yy[ii] = (y[i1+j1-1] - mu1[i1+j1-1]) * (y[i1+j2-1] - mu1[i1+j2-1])
                yy[ii] /= sqrt(va[i1+j1-1] * va[i1+j2-1])
                ii += 1
            end
        end
    end

end

function StatsBase.fit!(gee::GEE2; maxiter::Int = 100)

    (; n, p) = gee.pp_lin
    beta = [randn(p), randn(p), randn(2 * p)]

    for k = 1:maxiter
        expand!(gee, beta)
        iterprep!(gee, beta)
        update!(gee, beta)
    end

end

function fit(
    ::Type{GEE2},
    X::AbstractMatrix{T},
    y::AbstractVector{T},
    g::AbstractVector,
    c::CorStruct = IndependenceCor(),
    l::Link = IdentityLink(),
    vf::Varfunc = ConstantVar(),
    vf2::Varfunc = PowerVar(2);
    dofit::Bool = true,
    fitargs...,
) where {T<:Real}

    gee2 = GEE2(y, copy(X), g, l, vf, vf2, c)

    return dofit ? fit!(gee2) : geee
end


function GEE2(
    y,
    X,
    grp;
    l::Link = IdentityLink(),
    vf::Varfunc = ConstantVar(),
    vf2::Varfunc = PowerVar(2),
    corlin::CorStruct = IndependenceCor(),
    corsqr::CorStruct = IndependenceCor(),
)
    if !(length(y) == size(X, 1) == length(grp))
        msg = @sprintf(
            "length of y (%d), number of rows of X (%d) and length of grp (%d) must agree.\n",
            length(y),
            size(X, 1),
            length(grp)
        )
        error(msg)
    end

    if !issorted(grp)
        error("Group vector is not sorted")
    end
    gix, mxgrp = groupix(grp)

    yc, Xc, grpc, gixc = expand_cor(y, X, grp, gix)

    rr_mn = GEE2Resp(y, grp, gix)
    rr_var = GEE2Resp(y .^ 2, grp, gix)
    rr_cor = GEE2Resp(yc, grpc, gixc)

    n, p = size(X)
    Xt = copy(X)
    pp_mn = GEEEDensePred(n, p, gix, Xt)
    pp_var = GEEEDensePred(n, p, gix, Xt)
    pp_cor = GEEEDensePred(size(Xc, 1), 2*p, gixc, Xc)

    q = 4 * size(X, 2)
    cor = CorStruct[corlin, corsqr]
    return GEE2(
        rr_mn,
        rr_var,
        rr_cor,
        pp_mn,
        pp_var,
        pp_cor,
        l,
        vf,
        vf2,
        cor,
        zeros(q),
        zeros(q, q),
        mxgrp,
    )
end
