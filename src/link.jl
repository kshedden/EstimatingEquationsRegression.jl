# Most link infrastructure comes from GLM, but we add to it here.

struct SigmoidLink <: Link
    lower::Float64
    upper::Float64
    range::Float64
    eps::Float64
end

SigmoidLink(l, u) = SigmoidLink(l, u, u-l, (u-l)/10000)

function linkfun(sl::SigmoidLink, x::Real)
    x = clamp(x, sl.lower+sl.eps, sl.upper-sl.eps)
    return log(x - sl.lower) - log(sl.upper - x)
end

function linkinv(sl::SigmoidLink, x::Real)
    return sl.range / (1 + exp(-x)) + sl.lower
end

function inverselinkderiv2(::IdentityLink, eta::T) where{T <: Real}
    return zero(typeof(eta))
end

function inverselinkderiv2(::LogLink, eta::T) where {T<:Real}
    return exp(eta)
end

function inverselinkderiv2(::LogitLink, eta::T) where {T<:Real}
    u = exp(-eta)
    return u * (u - 1) / (1 + u)^3
end
