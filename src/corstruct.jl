abstract type CorStruct end

"""
    IndependenceCor <: CorStruct

Type that represents a GEE working correlation structure in which the
observations within a group are modeled as being independent.
"""
struct IndependenceCor <: CorStruct end

Base.copy(c::IndependenceCor) = IndependenceCor()

"""
    ExchangeableCor <: CorStruct

Type that represents a GEE working correlation structure in which the
observations within a group are modeled as exchangeably correlated.
Any two observations in a group have the same correlation between
them, which can be estimated from the data.
"""
mutable struct ExchangeableCor <: CorStruct
    aa::Float64

    # The correlation is never allowed to exceed this value.
    cap::Float64
end

Base.copy(c::ExchangeableCor) = ExchangeableCor(c.aa, c.cap)

function ExchangeableCor()
    ExchangeableCor(0.0, 0.999)
end

function ExchangeableCor(aa)
    ExchangeableCor(aa, 0.999)
end

"""
    AR1Cor <: CorStruct

Type that represents a GEE working correlation structure in which the
observations within a group are modeled as being serially correlated
according to their order in the dataset, with the correlation between
two observations that are j positions apart being `r^j` for a real
parameter `r` that can be estimated from the data.
"""
mutable struct AR1Cor <: CorStruct
    aa::Float64
end

function AR1Cor()
    AR1Cor(0.0)
end

"""
    OrdinalIndependenceCor <: CorStruct

Type that represents a GEE working correlation structure in which the
ordinal observations within a group are modeled as being independent.
Each ordinal observation is converted to a set of binary indicators,
and the indicators derived from a common ordinal value are modeled as
correlated, with the correlations determined from the marginal means.
"""
mutable struct OrdinalIndependenceCor <: CorStruct

    # The number of binary indicators derived from each
    # observed ordinal variable.
    numind::Int
end

function updatecor(c::AR1Cor, sresid::FPVector, g::Matrix{Int}, ddof::Int)

    lag0, lag1 = 0.0, 0.0
    for i = 1:size(g, 2)
        i1, i2 = g[1, i], g[2, i]
        q = i2 - i1 + 1 # group size
        if q < 2
            continue
        end
        s0, s1 = 0.0, 0.0
        for j1 = i1:i2
            s0 += sresid[j1]^2
            if j1 < i2
                s1 += sresid[j1] * sresid[j1+1]
            end
        end
        lag0 += s0 / q
        lag1 += s1 / (q - 1)
    end

    c.aa = lag1 / lag0

end

# Nothing to do for independence model.
function updatecor(c::IndependenceCor, sresid::FPVector, g::Matrix{Int}, ddof::Int) end
function updatecor(c::OrdinalIndependenceCor, sresid::FPVector, g::Matrix{Int}, ddof::Int) end

function updatecor(c::ExchangeableCor, sresid::FPVector, g::Matrix{Int}, ddof::Int)

    sxp, ssr = 0.0, 0.0
    npr, n = 0, 0

    for i = 1:size(g, 2)
        i1, i2 = g[1, i], g[2, i]
        for j1 = i1:i2
            ssr += sresid[j1]^2
            for j2 = j1+1:i2
                sxp += sresid[j1] * sresid[j2]
            end
        end
        q = i2 - i1 + 1 # group size
        n += q
        npr += q * (q - 1) / 2
    end

    scale = ssr / (n - ddof)
    sxp /= scale
    c.aa = sxp / (npr - ddof)
    c.aa = clamp(c.aa, 0, c.cap)
end

function covsolve(
    c::IndependenceCor,
    mu::AbstractVector{T},
    sd::AbstractVector{T},
    w::AbstractVector{T},
    z::AbstractArray{T},
) where {T<:Real}
    if length(w) > 0
        return w .* z ./ sd .^ 2
    else
        return z ./ sd .^ 2
    end
end

function covsolve(
    c::OrdinalIndependenceCor,
    mu::AbstractVector{T},
    sd::AbstractVector{T},
    w::AbstractVector{T},
    z::AbstractArray{T},
) where {T<:Real}

    p = length(mu)
    numind = c.numind
    @assert p % numind == 0
    q = div(p, numind)
    ma = zeros(p, p)
    ii = 0
    for k = 1:q
        for i = 1:numind
            for j = 1:numind
                ma[ii+i, ii+j] = min(mu[ii+i], mu[ii+j]) - mu[ii+i] * mu[ii+j]
            end
        end
        ii += numind
    end

    return ma \ z
end

function covsolve(
    c::ExchangeableCor,
    mu::AbstractVector{T},
    sd::AbstractVector{T},
    w::AbstractVector{T},
    z::AbstractArray{T},
) where {T<:Real}
    p = length(sd)
    a = c.aa
    f = a / ((1 - a) * (1 + a * (p - 1)))
    if length(w) > 0
        di = Diagonal(w ./ sd)
    else
        di = Diagonal(1 ./ sd)
    end
    x = di * z
    u = x ./ (1 - a)
    if length(size(z)) == 1
        u .= u .- f * sum(x) * ones(p)
    else
        u .= u .- f .* ones(p) * sum(x, dims = 1)
    end
    di * u
end

function covsolve(
    c::AR1Cor,
    mu::AbstractVector{T},
    sd::AbstractVector{T},
    w::AbstractVector{T},
    z::AbstractArray{T},
) where {T<:Real}

    r = c.aa[1]
    d = size(z, 1)
    q = length(size(z))

    if length(w) > 0
        z = Diagonal(w) * z
    end

    if d == 1
        # 1x1 case
        return z ./ sd .^ 2
    elseif d == 2
        # 2x2 case
        sp = sd[1] * sd[2]
        z1 = zeros(size(z))
        z1[1, :] .= z[1, :] / sd[1]^2 - r * z[2, :] / sp
        z1[2, :] .= -r * z[1, :] / sp + z[2, :] / sd[2]^2
        z1 .= z1 ./ (1 - r^2)
        return z1
    else
        # General case
        z1 = (z' ./ sd')'

        c0 = (1.0 + r^2) / (1.0 - r^2)
        c1 = 1.0 / (1.0 - r^2)
        c2 = -r / (1.0 - r^2)

        y = c0 * z1
        y[1:end-1, :] .= y[1:end-1, :] + c2 * z1[2:end, :]
        y[2:end, :] .= y[2:end, :] + c2 * z1[1:end-1, :]
        y[1, :] = c1 * z1[1, :] + c2 * z1[2, :]
        y[end, :] = c1 * z1[end, :] + c2 * z1[end-1, :]

        y = (y' ./ sd')'

        # If z is a vector, return a vector
        return q == 1 ? y[:, 1] : y
    end
end

function corparams(c::IndependenceCor) end

function corparams(c::OrdinalIndependenceCor) end

function corparams(c::ExchangeableCor)
    return c.aa
end

function corparams(c::AR1Cor)
    return c.aa
end
