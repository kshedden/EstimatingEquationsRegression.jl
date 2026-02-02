abstract type AbstractMarginalModel <: GLM.AbstractGLM end
abstract type AbstractGEE <: AbstractMarginalModel end

"""
    GEEResp

The response vector, grouping information, and vectors derived from
the response.  Vectors here are all n-dimensional.
"""
mutable struct GEEResp{T<:Real} <: ModResp

    "`y`: response vector"
    y::Vector{T}

    "`grpix`: delineate the groups"
    grpix::Vector{UnitRange{Int}}

    "`awts`: analytic weights, the model variance is scaled by the reciprocal of awts"
    awts::Vector{T}

    "`fwts`: frequency weights, each observation is replicated by the given weight value"
    fwts::Vector{T}

    "`η`: the linear predictor"
    η::Vector{T}

    "`mu`: the mean"
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
struct GEEprop

    "`L`: the link function (maps from mean to linear predictor)"
    link::Link

    "`varfunc`: used to determine the variance, only one of varfunc and `dist` should be specified"
    varfunc::Varfunc

    "`cor`: the working correlation structure"
    cor::CorStruct

    "`dist`: the distribution family, used only to determine the variance, not used if varfunc is provided."
    dist::UnivariateDistribution

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

    "`DtViD`: the unscaled inverse of the naive (model-dependent) covariance matrix for one group"
    DtViD_grp::Matrix{Float64}

    "`DtViD_sum`: the cumulative sum of DtViD"
    DtViD_sum::Matrix{Float64}

    "`mdcov`: the Mancel-DeRouen bias-reduced robust covariance matrix"
    mdcov::Matrix{Float64}

    "`kccov`: the Kauermann-Carroll bias-reduced robust covariance matrix"
    kccov::Matrix{Float64}

    "`scrcov`: the empirical Gram matrix of the score vectors (not scaled by n)"
    scrcov::Matrix{Float64}
end


function GEECov(p::Int)
    GEECov(zeros(p, p), zeros(p, p), zeros(p, p), zeros(p, p), zeros(p, p), zeros(0, 0), zeros(0, 0), zeros(p, p))
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
    isfitted::Bool
    converged::Bool
end

function GEEResp(y::Vector{T}, g::Vector{UnitRange{Int}}, awts::Vector{T}, fwts::Vector{T}, off::Vector{T}) where {T<:Real}
    return GEEResp{T}(y, g, awts, fwts, similar(y), similar(y), similar(y), similar(y),
                      similar(y), similar(y), similar(y), off)
end

# Preliminary calculations for one iteration of GEE fitting.
function _iterprep(mod::M) where{M<:AbstractGEE}

    (; pp, rr, qq, cc) = mod

    # Update the linear predictor
    updateη!(pp, rr.η, rr.offset)

    # Update the conditional means
    rr.mu .= linkinv.(qq.link, rr.η)

    # Update the raw residuals
    rr.resid .= rr.y .- rr.mu

    # Get the standard deviation, based on the variance and analytic weights
    rr.sd .= geevar.(qq.dist, qq.varfunc, rr.mu, rr.awts)
    if minimum(rr.sd) < 1e-10
        @warn("Some observation variances are nearly zero")
        rr.sd .= clamp.(rr.sd, 1e-10, Inf)
    end
    rr.sd .= sqrt.(rr.sd)

    # Update the standardized residuals
    rr.sresid .= rr.resid ./ rr.sd

    # Update the derivative of the mean function with respect to the linear predictor
    rr.dμdη .= mueta.(qq.link, rr.η)

    cc.DtViD_sum .= 0
    pp.score .= 0
    cc.scrcov .= 0
end

function _update_group(mod::M, j::Int, last::Bool) where{M<:AbstractGEE}

    (; pp, rr, qq, cc) = mod

    gr = rr.grpix[j]

    fwt = weights(mod; type=:frequency)[gr]
    updateD!(pp, rr.dμdη[gr], gr)
    rr.viresid[gr] .= covsolve(qq.cor, rr.mu[gr], rr.sd[gr], rr.resid[gr])
    pp.score_grp .= pp.D' * Diagonal(fwt) * rr.viresid[gr]
    pp.score .+= pp.score_grp
    cc.DtViD_grp .= pp.D' * Diagonal(fwt) * covsolve(qq.cor, rr.mu[gr], rr.sd[gr], pp.D)
    cc.DtViD_sum .+= cc.DtViD_grp

    if last
        # Only compute on final iteration
        cc.scrcov .+= pp.score_grp * pp.score_grp'
    end
end

function _iterate(mod::M, last::Bool) where{M<:AbstractGEE}

    (; pp, rr, qq, cc) = mod

    pp.score .= 0
    cc.DtViD_sum .= 0
    if last
        cc.scrcov .= 0
    end

    for j in eachindex(rr.grpix)
        _update_group(mod, j, last)
    end

    if last
        cc.scrcov .= Symmetric((cc.scrcov + cc.scrcov') / 2)
    end
    cc.DtViD_sum .= Symmetric((cc.DtViD_sum + cc.DtViD_sum') / 2)
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

    for (g, gr) in enumerate(r.grpix)

        # Computation of common quantities
        awts = r.awts[gr]
        updateD!(p, r.dμdη[gr], gr)
        vid = covsolve(q.cor, r.mu[gr], r.sd[gr], p.D)
        vid .= vid ./ di

        # Group size
        m = first(size(gr))

        # This is m x m, where m is the group size.
        # It could be large.
        h = p.D * c.nacov * vid'

        eval, evec = eigen(I(m) - h)
        if minimum(abs, eval) < 1e-14
            nfail += 1
            continue
        end

        # Kauermann-Carroll
        eval .= (eval + abs.(eval)) ./ 2
        eval2 = 1 ./ sqrt.(eval)
        eval2[eval.==0] .= 0
        ar = evec * diagm(eval2) * evec' * r.resid[gr]
        sr = covsolve(q.cor, r.mu[gr], r.sd[gr], real(ar))
        sr = p.D' * sr
        bcm_kc .+= sr * sr'

        # Mancl-DeRouen
        ar = (I(m) - h) \ r.resid[gr]
        sr = covsolve(q.cor, r.mu[gr], r.sd[gr], ar)
        sr = p.D' * sr
        bcm_md .+= sr * sr'
    end

    bcm_md .= bcm_md ./ di^2
    bcm_kc .= bcm_kc ./ di^2
    c.mdcov = c.nacov * bcm_md * c.nacov
    c.kccov = c.nacov * bcm_kc * c.nacov

    return nfail
end

# Project x to be a positive semi-definite matrix, also return
# a boolean indicating whether the matrix had non-negligible
# negative eigenvalues.
function pcov(x::T) where {T<:AbstractMatrix}
    x = Symmetric((x + x') / 2)
    a, b = eigen(x)
    f = minimum(a) <= -1e-8
    a = clamp.(a, 0, Inf)
    return Symmetric(b * diagm(a) * b'), f
end

function get_start(mod::M; maxiter=10, tol=1e-3, verbosity::Int=0) where{M<:AbstractGEE}

    (; pp, rr, qq, cc) = mod
    (; y, offset, awts) = rr
    (; X) = pp
    (; link, dist) = qq

    # Weighted least squares
    if typeof(link) <: IdentityLink
        yy = length(offset) > 0 ? y - offset : y
        W = Diagonal(awts)
        return (X' * W * X) \ (X' * W * yy)
    end

    if typeof(link) <: SigmoidLink
        return zeros(size(X, 2))
    end

    for iter = 1:maxiter
        _iterprep(mod)
        _iterate(mod, false)
        if norm(pp.score) < tol
            break
        end
        if verbosity > 2
            println("Starting values iteration=$(iter)")
        end
        update_coef!(pp, pp.score, cc.DtViD_sum; diagonalize=iter < 5, bclip=0.5)
    end

    return pp.beta0
end

function invert_scaling!(m)

    (; pp, rr, qq, cc) = m
    (; cov, rcov, nacov, mdcov, kccov, scrcov) = cc
    (; xscale, beta0) = pp

    if length(xscale) == 0
        # No scaling was done
        return
    end

    # Adjust the model parameters
    beta0 ./= xscale[:]

    # Adjust all the covariances
    xxscale = xscale' * xscale
    cov ./= xxscale
    rcov ./= xxscale
    nacov ./= xxscale
    mdcov ./= xxscale
    kccov ./= xxscale
    scrcov ./= xxscale
end

function fit!(mod::M; verbosity::Int=0, maxiter::Integer=100, atol::Float64=1e-6, rtol::Float64=1e-6,
               start::Vector=Float64[], fitcoef::Bool=true, fitcor::Bool=true, bccor::Bool=false, inference::Bool=true) where{M<:AbstractGEE}

    # Don't refit if the model has already been fit.
    if mod.isfitted
        return mod
    end

    (; pp, rr, qq, cc) = mod
    (; score) = pp
    (; y, grpix, η, mu, sd, dμdη, viresid, resid, sresid, offset) = rr
    (; link, dist, cor, ddof) = qq
    (; scrcov, nacov, DtViD_sum, DtViD_grp) = cc

    # GEE update of coef is not needed in this case
    independence = typeof(cor) <: IndependenceCor && length(start) == 0

    pp.beta0 = length(start) == 0 ? get_start(mod; verbosity=verbosity) : copy(start)
    if verbosity > 0
        println("Starting values: ", pp.beta0)
    end

    n, p = size(pp.X)
    last = !fitcoef # Indicates that we are on the final iteration
    cvg = false

    for iter in 1:maxiter
        _iterprep(mod)
        fitcor && updatecor(cor, sresid, grpix, ddof)
        _iterate(mod, last)

        if !fitcoef
            # Run _iterprep and _iterate once for side effects, then exit.
            break
        end

        update_coef!(pp, score, DtViD_sum)

        if last
            break
        end
        nrm = norm(pp.delbeta)
        verbosity > 0 && println("iteration $(iter), |step|=$(nrm)")
        cvg = nrm < atol
        last = (iter == maxiter - 1) || cvg
    end

    verbosity > 0 && println("completed iterations")

    # The model has now been fit
    mod.isfitted = true

    if !inference
        invert_scaling!(mod)
        return mod
    end

    nacov .= try
        inv(DtViD_sum)
    catch e
        # Don't compute standard errors
        @warn("Naive covariance matrix is singular")
        mm = Inf * ones(size(nacov)...)
        mod.cc.nacov = mm
        mod.cc.DtViD_sum = mm
        mod.cc.rcov = mm
        mod.cc.mdcov = mm
        mod.cc.kccov = mm
        mod.cc.cov = mm
        invert_scaling!(mod)
        return mod
    end

    # Robust covariance
    sw = try
        Symmetric(DtViD_sum \ scrcov / DtViD_sum)
    catch e
        mm = Inf * ones(size(nacov)...)
        mod.cc.rcov = mm
        mod.cc.mdcov = mm
        mod.cc.kccov = mm
        mod.cc.cov = mm
        invert_scaling!(mod)
        return mod
    end
    mod.cc.rcov, f = pcov(sw)
    if f
        @warn("Robust covariance matrix is not positive definite.")
    end

    # Naive covariance
    verbosity > 0 && println("Computing naive covariance")
    mod.cc.nacov, f = pcov(nacov .* dispersion(mod))
    if f
        @warn("Naive covariance matrix is not positive definite")
    end

    if cvg
        mod.converged = true
    else
        @warn("Warning: GEE fitting failed to converge.")
    end

    # Update the bias-corrected parameter covariances
    verbosity > 0 && println("Computing dispersion")
    di = dispersion(mod)

    if bccor
        verbosity > 0 && println("Computing bias corrected vcov")
        nfail = _update_bc!(pp, rr, qq, cc, di)
        if nfail > 0
            @warn "Failures in $(nfail) groups when computing bias-corrected standard errors"
        end
    end

    # Set the default covariance
    cc.cov = vcov(mod, cov_type = qq.cov_type)

    invert_scaling!(mod)

    return mod
end

Distributions.Distribution(q::GEEprop) = q.dist
Distributions.Distribution(m::GeneralizedEstimatingEquationsModel) = Distribution(m.qq)

corstruct(m::GeneralizedEstimatingEquationsModel{G,L}) where {G,L} = m.qq.cor
varfunc(m::GeneralizedEstimatingEquationsModel{G,L}) where {G,L} = m.qq.varfunc

function weights(m::GeneralizedEstimatingEquationsModel{G,L}; type::Symbol=:analytic) where {G,L}
    if type == :analytic
        return m.rr.awts
    elseif type == :frequency
        return m.rr.fwts
    else
        error("Weight type $(type) is not supported")
    end
end

function GLM.dispersion(m::AbstractGEE)
    r = m.rr.sresid
    if dispersion_parameter(m.qq.dist)
        awts = weights(m; type=:analytic)
        fwts = weights(m; type=:frequency)
        X = modelmatrix(m)
        d = sum(fwts) - size(X, 2)
        s = sum(i -> fwts[i] * r[i]^2 / awts[i], eachindex(r)) / d
    else
        one(eltype(r))
    end
end

function vcov(m::GeneralizedEstimatingEquationsModel; cov_type::String = "")

    if cov_type == ""
        # Default covariance
        return m.cc.cov
    elseif cov_type == "robust"
        return m.cc.rcov
    elseif cov_type == "naive"
        return m.cc.nacov
    elseif cov_type == "md"
        if size(m.cc.mdcov, 1) == 0
            error("Bias corrected covariance matrices not computed")
        end
        return m.cc.mdcov
    elseif cov_type == "kc"
        if size(m.cc.mdcov, 1) == 0
            error("Bias corrected covariance matrices not computed")
        end
        return m.cc.kccov
    else
        warning("Unknown cov_type '$(cov_type)'")
        return nothing
    end
end

function stderror(m::AbstractGEE; cov_type::String = "robust")
    v = diag(vcov(m; cov_type = cov_type))
    ii = findall((v .>= -1e-10) .& (v .<= 0))
    if length(ii) > 0
        v[ii] .= 0
        @warn "Estimated parameter covariance matrix is not positive definite"
    end
    return sqrt.(v)
end

function nobs(mm::GeneralizedEstimatingEquationsModel)
    return nobs(mm.pp)
end

function coeftable(
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
function prepargs(X, y, g, awts, fwts, offset)

    gi, mg = groupix(g)

    if !(size(X, 1) == length(y) == length(g) == length(awts) == length(fwts))
        m = @sprintf(
            "Number of rows in X (%d), y (%d), g (%d), awts (%d), and fwts (%d) must match",
            size(X, 1),
            length(y),
            length(g),
            length(awts),
            length(fwts)
        )
        throw(DimensionMismatch(m))
    end

    if any(awts .< 0)
        error("Negative 'awts' not allowed.")
    end

    if any(fwts .< 0)
        error("Negative 'fwts' not allowed.")
    end

    tl = [typeof(first(X)), typeof(first(y))]
    if length(awts) > 0
        push!(tl, typeof(first(awts)))
    end
    if length(offset) > 0
        push!(tl, typeof(first(offset)))
    end
    t = promote_type(tl...)
    X = t.(X)
    y = t.(y)
    awts = t.(awts)
    fwts = t.(fwts)
    offset = t.(offset)
    return X, y, awts, fwts, offset, gi, mg
end

# Fake distribution to indicate that the GEE was specified using link and variance
# function not the distribution.
struct QuasiLikelihood <: ContinuousUnivariateDistribution end

function fit_quasi(
    X::AbstractMatrix,
    y::AbstractVector,
    g::AbstractVector,
    l::Link,
    v::Varfunc,
    c::CorStruct;
    cov_type::String = "robust",
    dofit::Bool = true,
    awts::AbstractVector{<:Real} = ones(eltype(y), length(y)),
    fwts::AbstractVector{<:Real} = ones(eltype(y), length(y)),
    offset::AbstractVector{<:Real} = similar(y, 0),
    ddof_scale::Union{Int,Nothing} = nothing,
    scalex::Bool=false,
    start::Vector=Float64[],
    fitcoef::Bool=true,
    fitcor::Bool=true,
    bccor::Bool=false,
    inference::Bool=true,
    verbosity::Int=0,
    maxiter::Int=100,
    atol::Float64=1e-6,
    rtol::Float64=1e-6,
    fitargs...,
)
    X, y, awts, fwts, offset, gi, mg = prepargs(X, y, g, awts, fwts, offset)
    rr = GEEResp(y, gi, awts, fwts, offset)
    p = size(X, 2)
    ddof = isnothing(ddof_scale) ? p : ddof_scale
    mod = GeneralizedEstimatingEquationsModel(
        rr,
        DensePred(X, mg; scalex=scalex),
        GEEprop(l, v, c, NoDistribution(), ddof; cov_type),
        GEECov(p),
        false,
        false,
    )

    return dofit ? fit!(mod; verbosity=verbosity, maxiter=maxiter, atol=atol, rtol=rtol, start=start, fitcoef=fitcoef,
                        fitcor=fitcor, bccor=bccor, inference=inference) : mod
end

# Traditional GEE fitting function
function fit_gee(
    X::AbstractMatrix,
    y::AbstractVector,
    g::AbstractVector,
    d::UnivariateDistribution,
    c::CorStruct,
    l::Link;
    cov_type::String = "robust",
    dofit::Bool = true,
    awts::AbstractVector{<:Real} = ones(eltype(y), length(y)),
    fwts::AbstractVector{<:Real} = ones(eltype(y), length(y)),
    offset::AbstractVector{<:Real} = similar(y, 0),
    ddof_scale::Union{Int,Nothing} = nothing,
    scalex::Bool=false,
    start::Vector=Float64[],
    fitcoef::Bool=true,
    fitcor::Bool=true,
    bccor::Bool=false,
    inference::Bool=true,
    verbosity::Int=0,
    maxiter::Int=100,
    atol::Float64=1e-6,
    rtol::Float64=1e-6,
    fitargs...,
)
    X, y, awts, fwts, offset, gi, mg = prepargs(X, y, g, awts, fwts, offset)

    rr = GEEResp(y, gi, awts, fwts, offset)
    p = size(X, 2)
    ddof = isnothing(ddof_scale) ? p : ddof_scale
    mod = GeneralizedEstimatingEquationsModel(
        rr,
        DensePred(X, mg; scalex=scalex),
        GEEprop(l, DefaultVar(), c, d, ddof; cov_type),
        GEECov(p),
        false,
        false,
    )

    return dofit ? fit!(mod; verbosity=verbosity, maxiter=maxiter, atol=atol, rtol=rtol, start=start, fitcoef=fitcoef,
                        fitcor=fitcor, bccor=bccor, inference=inference) : mod
end

struct NoDistribution <: ContinuousUnivariateDistribution end

canonicallink(::NoDistribution) = IdentityLink()

"""
    fit(GeneralizedEstimatingEquationsModel, X, y, g; d, c, [l = canonicallink(d)], <keyword arguments>)

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
- `awts`: Analytic weights (variance is scaled by 1/awts).
- `fwts`: Frequency weights (each observation is replicated by the given weight).
- `offset::Vector=similar(y,0)`: offset added to `Xβ` to form `eta`.  Can be of
length 0 to indicate no offset (defaiult)
- `verbosity::Int=0`: Display convergence information for each iteration
- `maxiter::Integer=100`: Maximum number of iterations allowed to achieve convergence
- `atol::Real=1e-6`: Convergence is achieved when the relative change in
`β` is less than `max(rtol*dev, atol)`.
- `rtol::Real=1e-6`: Convergence is achieved when the relative change in
`β` is less than `max(rtol*dev, atol)`.
- `start::Vector=[]`: Starting values for coefficients. Must have the
same length as the number of columns in the model matrix, or empty.
- `fitcoef::Bool=true`: If false, set the coefficients equal to the GLM coefficients
or to `start` if provided, and update the correlation parameters and dispersion without
using GEE iterations to update the coefficients.`
- `fitcor::Bool=true`: If false, hold the correlation parameters equal to their starting
values.
- `bccor::Bool=false`: If false, do not compute the Kauermann-Carroll and Mancel-DeRouen
covariances.
"""
function fit(
    ::Type{GeneralizedEstimatingEquationsModel},
    X::AbstractMatrix,
    y::AbstractVector,
    g::AbstractVector;
    d::UnivariateDistribution = Normal(),
    c::CorStruct = IndependenceCor(),
    l::Link = canonicallink(d),
    v::Varfunc = DefaultVar(),
    cov_type::String = "robust",
    dofit::Bool = true,
    awts::AbstractVector{<:Real} = ones(eltype(y), length(y)),
    fwts::AbstractVector{<:Real} = ones(eltype(y), length(y)),
    offset::AbstractVector{<:Real} = similar(y, 0),
    ddof_scale::Union{Int,Nothing} = nothing,
    scalex::Bool=false,
    start::Vector=Float64[],
    fitcoef::Bool=true,
    fitcor::Bool=true,
    bccor::Bool=false,
    verbosity::Int=0,
    atol::Float64=1e-6,
    rtol::Float64=1e-6,
    fitargs...,
)

    if (typeof(d) <: NoDistribution) && (typeof(v) <: DefaultVar)
        error("If distribution family 'd' is NoDistribution, variance function 'v' must be provided")
    end

    if typeof(d) <: NoDistribution
        return fit_quasi(X, y, g, l, v, c; cov_type=cov_type, dofit=dofit, awts=awts, fwts=fwts, offset=offset,
                         ddof_scale=ddof_scale, scalex=scalex, start=start, fitcoef=fitcoef,
                         fitcor=fitcor, bccor=bccor, fitargs=fitargs, verbosity=verbosity, atol=atol, rtol=rtol)
    else
        return fit_gee(X, y, g, d, c, l; cov_type=cov_type, dofit=dofit, awts=awts, fwts=fwts, offset=offset,
                       ddof_scale=ddof_scale, scalex=scalex, start=start, fitcoef=fitcoef,
                       fitcor=fitcor, bccor=bccor, fitargs=fitargs, verbosity=verbosity, atol=atol, rtol=rtol)
    end
end

"""
    gee(F, D, args...; kwargs...)
Fit a generalized linear model to data using generalized estimating
equations. Alias for `fit(GeneralizedEstimatingEquationsModel, ...)`.
See [`fit`](@ref) for documentation.
"""
gee(F, D, args...; kwargs...) =
    fit(GeneralizedEstimatingEquationsModel, F, D, args...; kwargs...)


"""
    corparams(m::AbstractGEE)

Return the parameters that define the working correlation structure.
"""
function corparams(m::AbstractGEE)
    return corparams(m.qq.cor)
end

GLM.Link(m::GeneralizedEstimatingEquationsModel) = m.qq.link

function coefnames(m::GeneralizedEstimatingEquationsModel)
    p = size(m.pp.X, 2)
    return ["X" * string(j) for j in 1:p]
end

function residuals(m::AbstractGEE)
    return m.rr.resid
end

"""
    resid_pearson(m::AbstractGEE)

Return the Pearson residuals, which are the observed data
minus the mean, divided by the square root of the variance
function.  The scale parameter is not included so the Pearson
residuals should have constant variance but not necessarily
unit variance.
"""
function resid_pearson(m::AbstractGEE)
    return m.rr.sresid
end

"""
    predict(m::AbstractGEE; type=:linear)

Return the fitted values from the fitted model.  If
type is :linear returns the linear predictor, if
type is :response returns the fitted mean.
"""
function predict(m::AbstractGEE; type=:linear)
    if type == :linear
        m.rr.η
    elseif type == :response
        m.rr.mu
    else
        error("Unknown type='$(type)' in predict")
    end
end

offset(gee::GeneralizedEstimatingEquationsModel) = gee.rr.offset

function predict(m::AbstractGEE, newX::AbstractMatrix; type=:linear, offset=nothing)

    (; pp, qq) = m

    if length(EstimatingEquationsRegression.offset(m)) > 0 && isnothing(offset)
        @warn("Predicting on a model that was fit with an offset, but no offset was provided to predict.")
    end

    lp = newX * pp.beta0
    if !isnothing(offset)
        lp += offset
    end

    pr = if type == :linear
        lp
    elseif type == :response
        linkinv.(qq.link, lp)
    else
        error("Unknown type='$(type)' in predict")
    end

    return pr
end

isfitted(gee::GeneralizedEstimatingEquationsModel) = gee.isfitted
varfunc(gee::GeneralizedEstimatingEquationsModel) = gee.qq.varfunc
