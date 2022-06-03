using Printf, GLM

abstract type AbstractMarginalModel <: GLM.AbstractGLM end
abstract type AbstractGEE <: AbstractMarginalModel end

"""
    GEEResp

The response vector, grouping information, and vectors derived from
the response.  Vectors here are all n-dimensional.
"""
struct GEEResp{T<:Real} <: ModResp

    "`y`: response vector"
    y::Vector{T}

    "`grpix`: group positions, each column contains positions i1, i2 spanning one group"
    grpix::Matrix{Int}

    "`wts`: case weights"
    wts::Vector{T}

    "`η`: the linear predictor"
    η::Vector{T}

    "`mu`: the mean, use `mu` instead of `μ` for compatibility with GLM"
    mu::Vector{T}

    "`resid`: residuals"
    resid::Vector{T}

    "`sresid`: standardized (Pearson) residuals"
    sresid::Vector{T}

    "`sd`: the standard deviation of the observations"
    sd::Vector{T}

    "`dμdη`: derivative of mean with respect to linear predictor"
    dμdη::Vector{T}

    "`viresid`: whitened residuals"
    viresid::Vector{T}

    "`offset`: offset is added to the linear predictor"
    offset::Vector{T}
end

"""
    GEEprop

Properties that define a GLM fit using GEE - link, distribution, and
working correlation structure.
"""
struct GEEprop{D<:UnivariateDistribution,L<:Link,R<:CorStruct}

    "`L`: the link function (maps from mean to linear predictor)"
    link::L

    "`varfunc`: used to determine the variance, only one of varfunc and `dist` should be specified"
    varfunc::Varfunc

    "`cor`: the working correlation structure"
    cor::R

    "`dist`: the distribution family, used only to determine the variance, not used if varfunc is provided."
    dist::D

    "`ddof`: adjustment to the denominator degrees of freedom for estimating
     the scale parameter, this value is subtracted from the sample size to
     obtain the degrees of freedom."
    ddof::Int

    "`cov_type`: the type of parameter covariance (default is robust)"
    cov_type::String
end

function GEEprop(link, varfunc, cor, dist, ddof; cov_type = "robust")
    GEEprop(link, varfunc, cor, dist, ddof, cov_type)
end

"""
    GEECov

Covariance matrices for the parameter estimates.
"""
mutable struct GEECov

    "`cov`: the parameter covariance matrix"
    cov::Matrix{Float64}

    "`rcov`: the robust covariance matrix"
    rcov::Matrix{Float64}

    "`nacov`: the naive (model-dependent) covariance matrix"
    nacov::Matrix{Float64}

    "`mdcov`: the Mancel-DeRouen bias-reduced robust covariance matrix"
    mdcov::Matrix{Float64}

    "`kccov`: the Kauermann-Carroll bias-reduced robust covariance matrix"
    kccov::Matrix{Float64}

    "`scrcov`: the empirical Gram matrix of the score vectors (not scaled by n)"
    scrcov::Matrix{Float64}
end


function GEECov(p::Int)
    GEECov(zeros(p, p), zeros(p, p), zeros(p, p), zeros(p, p), zeros(p, p), zeros(p, p))
end

"""
    GeneralizedEstimatingEquationsModel <: AbstractGEE

Type representing a GLM to be fit using generalized estimating
equations (GEE).
"""
mutable struct GeneralizedEstimatingEquationsModel{G<:GEEResp,L<:LinPred} <: AbstractGEE
    rr::G
    pp::L
    qq::GEEprop
    cc::GEECov
    fit::Bool
    converged::Bool
end

function GEEResp(
    y::Vector{T},
    g::Matrix{Int},
    wts::Vector{T},
    off::Vector{T},
) where {T<:Real}
    return GEEResp{T}(
        y,
        g,
        wts,
        similar(y),
        similar(y),
        similar(y),
        similar(y),
        similar(y),
        similar(y),
        similar(y),
        off,
    )
end

# Preliminary calculations for one iteration of GEE fitting.
function _iterprep(p::LinPred, r::GEEResp, q::GEEprop)

    # Update the linear predictor
    updateη!(p, r.η, r.offset)

    # Update the conditional means
    r.mu .= linkinv.(q.link, r.η)

    # Update the raw residuals
    r.resid .= r.y .- r.mu

    # The variance can be determined either by the family, or supplied directly.
    r.sd .= if typeof(q.varfunc) <: NullVar
        glmvar.(q.dist, r.mu)
    else
        geevar.(q.varfunc, r.mu)
    end
    r.sd .= sqrt.(r.sd)

    # Update the standardized residuals
    r.sresid .= r.resid ./ r.sd

    # Update the derivative of the mean function with respect to the linear predictor
    r.dμdη .= mueta.(q.link, r.η)
end

function _iterate(p::LinPred, r::GEEResp, q::GEEprop, c::GEECov, last::Bool) where {T<:Real}

    p.score .= 0
    c.nacov .= 0
    if last
        c.scrcov .= 0
    end

    for (g, (i1, i2)) in enumerate(eachcol(r.grpix))
        updateD!(p, r.dμdη[i1:i2], i1, i2)
        w = length(r.wts) > 0 ? r.wts[i1:i2] : zeros(0)
        r.viresid[i1:i2] .= covsolve(q.cor, r.mu[i1:i2], r.sd[i1:i2], w, r.resid[i1:i2])
        p.score_obs .= p.D' * r.viresid[i1:i2]
        p.score .+= p.score_obs
        c.nacov .+= p.D' * covsolve(q.cor, r.mu[i1:i2], r.sd[i1:i2], w, p.D)

        if last
            # Only compute on final iteration
            c.scrcov .+= p.score_obs * p.score_obs'
        end
    end

    if last
        c.scrcov .= Symmetric((c.scrcov + c.scrcov') / 2)
    end
    c.nacov .= Symmetric((c.nacov + c.nacov') / 2)
end

# Calculate the Mancl-DeRouen and Kauermann-Carroll bias-corrected
# parameter covariance matrices.  This must run after the parameter
# fitting because it requires the naive covariance matrix and scale
# parameter estimates.
function _update_bc!(p::LinPred, r::GEEResp, q::GEEprop, c::GEECov, di::Float64)::Int

    m = size(p.X, 2)
    bcm_md = zeros(m, m)
    bcm_kc = zeros(m, m)
    nfail = 0

    for (g, (i1, i2)) in enumerate(eachcol(r.grpix))

        # Computation of common quantities
        w = length(r.wts) > 0 ? r.wts[i1:i2] : zeros(0)
        updateD!(p, r.dμdη[i1:i2], i1, i2)
        vid = covsolve(q.cor, r.mu[i1:i2], r.sd[i1:i2], w, p.D)
        vid .= vid ./ di

        # This is m x m, where m is the group size.
        # It could be large.
        h = p.D * c.nacov * vid'

        m = i2 - i1 + 1
        eval, evec = eigen(I(m) - h)
        if minimum(abs, eval) < 1e-14
            nfail += 1
            continue
        end

        # Kauermann-Carroll
        eval .= (eval + abs.(eval)) ./ 2
        eval2 = 1 ./ sqrt.(eval)
        eval2[eval.==0] .= 0
        ar = evec * diagm(eval2) * evec' * r.resid[i1:i2]
        sr = covsolve(q.cor, r.mu[i1:i2], r.sd[i1:i2], w, real(ar))
        sr = p.D' * sr
        bcm_kc .+= sr * sr'

        # Mancl-DeRouen
        ar = (I(m) - h) \ r.resid[i1:i2]
        sr = covsolve(q.cor, r.mu[i1:i2], r.sd[i1:i2], w, ar)
        sr = p.D' * sr
        bcm_md .+= sr * sr'
    end

    bcm_md .= bcm_md ./ di^2
    bcm_kc .= bcm_kc ./ di^2
    c.mdcov .= c.nacov * bcm_md * c.nacov
    c.kccov .= c.nacov * bcm_kc * c.nacov

    return nfail
end

# Project x to be a positive semi-definite matrix, also return
# a boolean indicating whether the matrix had non-negligible
# negative eigenvalues.
function pcov(x::Matrix)
    x = Symmetric((x + x') / 2)
    a, b = eigen(x)
    f = minimum(a) <= -1e-8
    a = clamp.(a, 0, Inf)
    return Symmetric(b * diagm(a) * b'), f
end

function _fit!(
    m::AbstractGEE,
    verbose::Bool,
    maxiter::Integer,
    atol::Real,
    rtol::Real,
    start,
    fitcoef::Bool,
    fitcor::Bool,
    bccor::Bool,
)
    m.fit && return m

    (; pp, rr, qq, cc) = m
    (; y, grpix, η, mu, sd, dμdη, viresid, resid, sresid) = rr
    (; link, dist, cor, ddof) = qq
    (; scrcov, nacov) = cc
    score = pp.score

    # GEE update of coef is not needed in this case
    independence = typeof(cor) <: IndependenceCor && isnothing(start)

    if isnothing(start)
        # The default maxiter for GLM seems to be too small
        maxit = min(maxiter, 100)
        gm = StatsBase.fit(
            GeneralizedLinearModel,
            pp.X,
            y,
            dist,
            link;
            wts = m.rr.wts,
            maxiter = maxit,
        )
        start = coef(gm)
    end

    pp.beta0 = start

    n, p = size(pp.X)
    last = false || !fitcoef || independence
    cvg = false || !fitcoef || independence

    for iter = 1:maxiter
        _iterprep(pp, rr, qq)
        fitcor && updatecor(cor, sresid, grpix, ddof)
        _iterate(pp, rr, qq, cc, last)

        fitcoef || break

        updateβ!(pp, score, nacov)

        if last
            break
        end
        nrm = norm(pp.delbeta)
        verbose && println("Iteration $iter, step norm=$nrm")
        cvg = nrm < atol
        last = (iter == maxiter - 1) || cvg
    end

    # Robust covariance
    m.cc.rcov, f = pcov(nacov \ scrcov / nacov)
    if f
        @warn("Robust covariance matrix is not positive definite.")
    end

    # Naive covariance
    m.cc.nacov, f = pcov(inv(nacov) .* dispersion(m))
    if f
        @warn("Naive covariance matrix is not positive definite")
    end

    if cvg
        m.converged = true
    else
        @warn("Warning: GEE failed to converge.")
    end

    # The model has been fit
    m.fit = true

    # Update the bias-corrected parameter covariances
    di = dispersion(m)

    if bccor
        nfail = _update_bc!(pp, rr, qq, cc, di)
        if nfail > 0
            @warn "Failures in $(nfail) groups when computing bias-corrected standard errors"
        end
    end

    # Set the default covariance
    cc.cov = vcov(m, cov_type = qq.cov_type)

    return m
end

Distributions.Distribution(q::GEEprop) = q.dist
Distributions.Distribution(m::GeneralizedEstimatingEquationsModel) = Distribution(m.qq)

Corstruct(m::GeneralizedEstimatingEquationsModel{G,L}) where {G,L} = m.qq.cor

function dispersion(m::AbstractGEE)
    r = m.rr.sresid
    if dispersion_parameter(m.qq.dist)
        if length(m.rr.wts) > 0
            w = m.rr.wts
            d = sum(w) - size(m.pp.X, 2)
            s = sum(i -> w[i] * r[i]^2, eachindex(r)) / d
        else
            s = sum(i -> r[i]^2, eachindex(r)) / dof_residual(m)
        end
    else
        one(eltype(r))
    end
end

function StatsBase.vcov(m::AbstractGEE; cov_type::String = "")
    if cov_type == ""
        # Default covariance
        return m.cc.cov
    elseif cov_type == "robust"
        return m.cc.rcov
    elseif cov_type == "naive"
        return m.cc.nacov
    elseif cov_type == "md"
        return m.cc.mdcov
    elseif cov_type == "kc"
        return m.cc.kccov
    else
        warning("Unknown cov_type '$(cov_type)'")
        return nothing
    end
end

function StatsBase.stderror(m::AbstractGEE; cov_type::String = "robust")::Array{Float64,1}
    v = diag(vcov(m; cov_type = cov_type))
    ii = findall((v .>= -1e-10) .& (v .<= 0))
    if length(ii) > 0
        v[ii] .= 0
        @warn "Estimated parameter covariance matrix is not positive definite"
    end
    return sqrt.(v)
end

function StatsBase.coeftable(
    mm::GeneralizedEstimatingEquationsModel;
    level::Real = 0.95,
    cov_type::String = "",
)
    cov_type = (cov_type == "") ? mm.qq.cov_type : cov_type
    cc = coef(mm)
    se = stderror(mm; cov_type = cov_type)
    zz = cc ./ se
    p = 2 * ccdf.(Ref(Normal()), abs.(zz))
    ci = se * quantile(Normal(), (1 - level) / 2)
    levstr = isinteger(level * 100) ? string(Integer(level * 100)) : string(level * 100)
    CoefTable(
        hcat(cc, se, zz, p, cc + ci, cc - ci),
        ["Coef.", "Std. Error", "z", "Pr(>|z|)", "Lower $levstr%", "Upper $levstr%"],
        ["x$i" for i = 1:size(mm.pp.X, 2)],
        4,
        3,
    )
end

dof(x::GeneralizedEstimatingEquationsModel) =
    dispersion_parameter(x.qq.dist) ? length(coef(x)) + 1 : length(coef(x))


# Ensure that X, y, wts, and offset have the same type
function prepargs(X, y, g, wts, offset)

    (gi, mg) = groupix(g)

    if !(size(X, 1) == length(y) == length(g))
        m = @sprintf(
            "Number of rows in X (%d), y (%d), and g (%d) must match",
            size(X, 1),
            length(y),
            length(g)
        )
        throw(DimensionMismatch(m))
    end

    tl = [typeof(first(X)), typeof(first(y))]
    if length(wts) > 0
        push!(tl, typeof(first(wts)))
    end
    if length(offset) > 0
        push!(tl, typeof(first(offset)))
    end
    t = promote_type(tl...)
    X = t.(X)
    y = t.(y)
    wts = t.(wts)
    offset = t.(offset)
    return X, y, wts, offset, gi, mg
end

"""
    fit(GeneralizedEstimatingEquationsModel, X, y, g, l, v, [c = IndependenceCor()]; <keyword arguments>)

Fit a generalized linear model to data using generalized estimating equations (GEE).  This
interface emphasizes the "quasi-likelihood" framework for GEE and requires direct specification
of the link and variance function, without reference to any distribution/family.
"""
function fit(
    ::Type{GeneralizedEstimatingEquationsModel},
    X::AbstractMatrix,
    y::AbstractVector,
    g::AbstractVector,
    l::Link,
    v::Varfunc,
    c::CorStruct = IndependenceCor();
    cov_type::String = "robust",
    dofit::Bool = true,
    wts::AbstractVector{<:Real} = similar(y, 0),
    offset::AbstractVector{<:Real} = similar(y, 0),
    ddof_scale::Union{Int,Nothing} = nothing,
    fitargs...,
)
    d = Normal() # Not used, only a placeholder

    X, y, wts, offset, gi, mg = prepargs(X, y, g, wts, offset)

    rr = GEEResp(y, gi, wts, offset)
    p = size(X, 2)
    ddof = isnothing(ddof_scale) ? p : ddof_scale
    res = GeneralizedEstimatingEquationsModel(
        rr,
        DensePred(X, mg),
        GEEprop(l, v, c, d, ddof; cov_type),
        GEECov(p),
        false,
        false,
    )

    return dofit ? fit!(res; fitargs...) : res
end

"""
    fit(GeneralizedEstimatingEquationsModel, X, y, g, d, c, [l = canonicallink(d)]; <keyword arguments>)

Fit a generalized linear model to data using generalized estimating
equations.  `X` and `y` can either be a matrix and a vector,
respectively, or a formula and a data frame. `g` is a vector
containing group labels, and elements in a group must be consecutive
in the data.  `d` must be a `UnivariateDistribution`, `c` must be a
`CorStruct` and `l` must be a [`Link`](@ref), if supplied.

# Keyword Arguments
- `cov_type::String`: Type of covariance estimate for parameters. Defaults
to "robust", other options are "naive", "md" (Mancl-DeRouen debiased) and
"kc" (Kauermann-Carroll debiased).xs
- `dofit::Bool=true`: Determines whether model will be fit
- `wts::Vector=similar(y,0)`: Not implemented.
Can be length 0 to indicate no weighting (default).
- `offset::Vector=similar(y,0)`: offset added to `Xβ` to form `eta`.  Can be of
length 0
- `verbose::Bool=false`: Display convergence information for each iteration
- `maxiter::Integer=100`: Maximum number of iterations allowed to achieve convergence
- `atol::Real=1e-6`: Convergence is achieved when the relative change in
`β` is less than `max(rtol*dev, atol)`.
- `rtol::Real=1e-6`: Convergence is achieved when the relative change in
`β` is less than `max(rtol*dev, atol)`.
- `start::AbstractVector=nothing`: Starting values for beta. Should have the
same length as the number of columns in the model matrix.
- `fitcoef::Bool=true`: If false, set the coefficients equal to the GLM coefficients
or to `start` if provided, and update the correlation parameters and dispersion without
using GEE iterations to update the coefficients.`
- `fitcor::Bool=true`: If false, hold the correlation parameters equal to their starting
values.
- `bccor::Bool=true`: If false, do not compute the Kauermann-Carroll and Mancel-DeRouen
covariances.
"""
function fit(
    ::Type{GeneralizedEstimatingEquationsModel},
    X::AbstractMatrix,
    y::AbstractVector,
    g::AbstractVector,
    d::UnivariateDistribution = Normal(),
    c::CorStruct = IndependenceCor(),
    l::Link = canonicallink(d);
    cov_type::String = "robust",
    dofit::Bool = true,
    wts::AbstractVector{<:Real} = similar(y, 0),
    offset::AbstractVector{<:Real} = similar(y, 0),
    ddof_scale::Union{Int,Nothing} = nothing,
    fitargs...,
)
    X, y, wts, offset, gi, mg = prepargs(X, y, g, wts, offset)

    rr = GEEResp(y, gi, wts, offset)
    p = size(X, 2)
    ddof = isnothing(ddof_scale) ? p : ddof_scale
    res = GeneralizedEstimatingEquationsModel(
        rr,
        DensePred(X, mg),
        GEEprop(l, NullVar(), c, d, ddof; cov_type),
        GEECov(p),
        false,
        false,
    )

    return dofit ? fit!(res; fitargs...) : res
end

"""
    gee(F, D, args...; kwargs...)
Fit a generalized linear model to data using generalized estimating
equations. Alias for `fit(GeneralizedEstimatingEquationsModel, ...)`.
See [`fit`](@ref) for documentation.
"""
gee(F, D, args...; kwargs...) =
    fit(GeneralizedEstimatingEquationsModel, F, D, args...; kwargs...)


function StatsBase.fit!(
    m::AbstractGEE;
    verbose::Bool = false,
    maxiter::Integer = 50,
    atol::Real = 1e-6,
    rtol::Real = 1e-6,
    start = nothing,
    fitcoef::Bool = true,
    fitcor::Bool = true,
    bccor::Bool = true,
    kwargs...,
)
    _fit!(m, verbose, maxiter, atol, rtol, start, fitcoef, fitcor, bccor)
end

"""
    corparams(m::AbstractGEE)

Return the parameters that define the working correlation structure.
"""
function corparams(m::AbstractGEE)
    return corparams(m.qq.cor)
end

GLM.Link(m::GeneralizedEstimatingEquationsModel) = m.qq.link
