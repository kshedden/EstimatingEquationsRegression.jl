
abstract type UngroupedPred <: LinPred end

struct UngroupedDensePred{S<:Real} <: UngroupedPred

    "`X`: the regression design matrix"
    X::Matrix{S}

    "`score`: the current score vector"
    score::Vector{Float64}

    "`D`: the Jacobian of the mean with respect to the coefficients"
    D::Matrix{Float64}

    "`beta0`: the current parameter estimate"
    beta0::Vector{Float64}
end

function UngroupedDensePred(X)
    n, p = size(X)
    pred = UngroupedDensePred(X, zeros(p), zeros(n, p), zeros(p))
    return pred
end

struct UngroupedResp <: LinPred

    "`y`: the responses"
    y::Vector{Float64}

    "`lp`: the linear predictor"
    lp::Vector{Float64}

    "`mn`: the current mean"
    mu::Vector{Float64}

    "`sd`: the current standard deviation"
    sd::Vector{Float64}

    "`resid`: the current raw residuals"
    resid::Vector{Float64}

    "`dmu_dlp`: the current values of the derivative of the mean with respect to the linear predictor"
    dmu_dlp::Vector{Float64}

    "`hat`: the current diagonal of the hat matrix"
    hat::Vector{Float64}

    "`awts`: analytic weights, the model variance is scaled by the reciprocal of awts"
    awts::Vector{Float64}

    "`fwts`: frequency weights, each observation is replicated by the given weight value"
    fwts::Vector{Float64}

    "`viresid`: residuals divided by variance"
    viresid::Vector{Float64}

    "`offset`: offset is added to the linear predictor"
    offset::Vector{Float64}
end

function UngroupedResp(y, awts, fwts, offset)
    n = length(y)
    resp = UngroupedResp(y, zeros(n), zeros(n), zeros(n), zeros(n), zeros(n), zeros(n), awts, fwts, zeros(n), offset)
    return resp
end

function nobs(pred::UngroupedDensePred)
    return size(pred.X, 1)
end

function nvar(pred::UngroupedDensePred)
    return size(pred.X, 2)
end

mutable struct FirthModel{S<:UngroupedResp,T<:UngroupedPred} <: AbstractMarginalModel

    "`resp`: data and working values related to the response"
    resp::S

    "`pred`: data related to the predictors"
    pred::T

    "`link`: the link function (maps from mean to linear predictor)"
    link::Link

    "`varfunc`: used to determine the variance as a function of the means"
    varfunc::Varfunc

    "`vcov`: the estimated variance/covariance matrix of the parameter estimates"
    vcov::Matrix{Float64}

    "`converged`: indicates whether the model has been successfully fit"
    converged::Bool
end

function FirthModel(resp, pred, l::Link, v::Varfunc)
    p = nvar(pred)
    return FirthModel(resp, pred, l, v, zeros(p, p), false)
end

function nvar(mm::FirthModel)
    return nvar(mm.pred)
end

function nobs(mm::FirthModel)
    return nobs(mm.pred)
end

function get_linpred(pred::UngroupedDensePred, coef::Vector{T}) where {T<:Real}
    return pred.X * coef
end

function updateD!(pred::UngroupedDensePred, dmu_dlp::FPVector)
    pred.D .= Diagonal(dmu_dlp) * pred.X
end

function update_lp!(p::UngroupedDensePred, lp::Vector{Float64}, off::Vector{T}) where {T<:Real}
    lp .= p.X * p.beta0
    if length(off) > 0
        lp .+= off
    end
end

function _update!(mod::FirthModel)

    (; pred, resp, varfunc) = mod

    # Update the linear predictor
    update_lp!(pred, resp.lp, resp.offset)

    # Update the conditional means
    resp.mu .= linkinv.(mod.link, resp.lp)

    # Update the raw residuals
    resp.resid .= resp.y .- resp.mu

    # Update the variances
    resp.sd .= geevar.(varfunc, resp.mu, resp.awts)
    if minimum(resp.sd) < 1e-10
        @warn("Some observation variances are nearly zero")
        resp.sd .= clamp.(resp.sd, 1e-10, Inf)
    end
    resp.sd .= sqrt.(resp.sd)

    # Update the derivative of the mean function with respect to the linear predictor
    for i in eachindex(resp.lp)
        resp.dmu_dlp[i] = inverselink(mod.link, resp.lp[i])[2]
    end

    fwt = weights(mod; type=:frequency)
    updateD!(pred, resp.dmu_dlp)
    resp.viresid .= resp.resid ./ resp.sd.^2
    pred.score .= pred.D' * Diagonal(fwt) * resp.viresid
end

function set_hat!(mod::FirthModel)

    (; resp, pred) = mod
    (; dmu_dlp) = resp
    (; X) = pred

    n = nobs(mod)
    fwt = weights(mod; type=:frequency)

    # The meat of the hat values
    M = X' * Diagonal(dmu_dlp .* fwt) * X
    MI = inv(M)

    for i in 1:n
        resp.hat[i] = fwt[i] * dmu_dlp[i] * X[i, :]' * MI * X[i, :]
    end
end

function fit!(mod::FirthModel; verbosity::Int=0, maxiter::Integer=100, atol::Float64=1e-6, rtol::Float64=1e-6,
              start::Vector=Float64[])

    (; pred, resp) = mod

    n = nobs(mod)
    p = nvar(mod)
    w = zeros(n)
    H = zeros(p, p)
    Ustar = zeros(p)
    step = zeros(p)
    sm = zeros(n)

    fwt = weights(mod; type=:frequency)

    # Starting values
    pred.beta0 .= 0
    converged = false

    for itr in 1:maxiter

        _update!(mod)
        set_hat!(mod)

        # The expected Fisher information matrix
        w .= (resp.dmu_dlp ./ resp.sd).^2
        w .*= (resp.fwts + resp.hat)
        H .= pred.X' * Diagonal(w) * pred.X

        # Firth's additive modification to the score.
        sm .= inverselinkderiv2.(mod.link, resp.lp)
        sm ./= resp.dmu_dlp
        sm .*= resp.hat

        # The Firth score vector
        Ustar .= pred.score + 0.5 * pred.X' * sm

        # Update the coefficients with Fisher scoring
        step .= H \ Ustar
        mx = maximum(abs.(step))
        if mx > 1
            step ./= mx
        end
        pred.beta0 .+= step

        if norm(step) < rtol
            converged = true
            break
        end
    end

    mod.vcov .= inv(H)

    if !converged
        @warn "Firth regression did not converge"
    end
end

function fit(
    ::Type{FirthModel},
    X::AbstractMatrix,
    y::AbstractVector;
    l::Link = IdentityLink(),
    v::Varfunc = ConstantVar(),
    awts::AbstractVector{<:Real} = ones(eltype(y), length(y)),
    fwts::AbstractVector{<:Real} = ones(eltype(y), length(y)),
    offset::AbstractVector{<:Real} = similar(y, 0))

    n = length(y)
    p = size(X, 2)

    if size(X, 1) != n
        error("Length of 'y' ($(length(y))) must equal the leading dimension of 'X' ($(size(X, 1))).")
    end

    pred = UngroupedDensePred(X)

    y = convert(Vector{Float64}, y)
    awts = convert(Vector{Float64}, awts)
    fwts = convert(Vector{Float64}, fwts)
    resp = UngroupedResp(y, awts, fwts, offset)

    mod = FirthModel(resp, pred, l, v)

    fit!(mod)
    return mod
end

function weights(m::FirthModel; type::Symbol=:analytic)
    if type == :analytic
        return m.resp.awts
    elseif type == :frequency
        return m.resp.fwts
    else
        error("Weight type $(type) is not supported")
    end
end

function coef(mod::FirthModel)
    return mod.pred.beta0
end

function vcov(mod::FirthModel)
    return mod.vcov
end

function modelmatrix(mm::FirthModel)
    return mm.pred.X
end

function coeftable(mm::FirthModel; level::Real = 0.95)
    cc = coef(mm)
    se = stderror(mm)
    zz = cc ./ se
    p = 2 * ccdf.(Ref(Normal()), abs.(zz))
    ci = se * quantile(Normal(), (1 - level) / 2)
    levstr = isinteger(level * 100) ? string(Integer(level * 100)) : string(level * 100)
    CoefTable(
        hcat(cc, se, zz, p, cc + ci, cc - ci),
        ["Coef.", "Std. Error", "z", "Pr(>|z|)", "Lower $levstr%", "Upper $levstr%"],
        ["x$i" for i = 1:nvar(mm)],
        4,
        3,
    )
end

function show(io::IO, mm::FirthModel)

    ta = [@sprintf("Num obs: %d", nobs(mm))]

    tf = TextTableFormat(borders=text_table_borders__compact)
    pretty_table(io, ta; table_format=tf, show_column_labels=false)
    show(io, MIME(Symbol("text/plain")), coeftable(mm))
end
