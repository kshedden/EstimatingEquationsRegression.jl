abstract type CorStruct end

struct IndependenceCor <: CorStruct end

mutable struct ExchangeableCor <: CorStruct
    aa::Float64
end

function ExchangeableCor()
    ExchangeableCor(0.0)
end

mutable struct AR1Cor <: CorStruct
    aa::Float64
end

function AR1Cor()
    AR1Cor(0.0)
end

function updatecor(c::AR1Cor, sresid::FPVector, g::Array{Int,2})

    aa, sp = 0.0, 0.0
    ma, ms = 0, 0

    for i = 1:size(g, 1)
        i1, i2 = g[i, 1], g[i, 2]
        for j1 = i1:i2
            sp += sresid[j1]^2
            if j1 < i2
                aa += sresid[j1] * sresid[j1+1]
            end
        end
        q = i2 - i1 + 1
        ms += q
        ma += q - 1
    end

    c.aa = (aa / ma) / (sp / ms)

end

function updatecor(c::IndependenceCor, sresid::FPVector, g::Array{Int,2}) end

function updatecor(c::ExchangeableCor, sresid::FPVector, g::Array{Int,2})

    aa, sp = 0.0, 0.0
    ma, ms = 0, 0

    for i = 1:size(g, 1)
        i1, i2 = g[i, 1], g[i, 2]
        for j1 = i1:i2
            sp += sresid[j1]^2
            for j2 = j1+1:i2
                aa += sresid[j1] * sresid[j2]
            end
        end
        q = i2 - i1 + 1
        ms += q
        ma += q * (q - 1) / 2
    end

    c.aa = (aa / ma) / (sp / ms)

end

function covsolve(c::IndependenceCor, sd::Array{T}, w::Array{T}, z::Array{T}) where {T<:Real}
    if length(w) > 0
        return Diagonal(w ./ sd .^ 2) * z
    else
        return Diagonal(1 ./ sd .^ 2) * z
    end
end

function covsolve(c::ExchangeableCor, sd::Array{T}, w::Array{T}, z::Array{T}) where {T<:Real}
    a = c.aa
    p = length(sd)
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

function covsolve(c::AR1Cor, sd::Array{T}, w::Array{T}, z::Array{T}) where {T<:Real}

    r = c.aa[1]
    d = size(z, 1)
    q = length(size(z))

    if length(w) > 0
        z = Diagonal(w) * z
    end

    if d == 1
        # 1x1 case
        return z ./ sd.^2
    elseif d == 2
        # 2x2 case
        sp = sd[1] * sd[2]
        z1 = zeros(size(z))
        z1[1, :] .= z[1, :]/sd[1]^2 - r*z[2, :]/sp
        z1[2, :] .= -r*z[1, :]/sp + z[2, :]/sd[2]^2
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
        y[1, :] = c1*z1[1, :] + c2*z1[2, :]
        y[end, :] = c1*z1[end, :] + c2*z1[end-1, :]

        y = (y' ./ sd')'

        # If z is a vector, return a vector
        return q == 1 ? y[:, 1] : y
    end
end

function corparams(c::IndependenceCor) end

function corparams(c::ExchangeableCor)
    return c.aa
end

function corparams(c::AR1Cor)
    return c.aa
end
