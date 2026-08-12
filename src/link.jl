# Most link infrastructure comes from GLM, but we need second derivatives.

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

