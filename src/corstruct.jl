abstract type CorStruct end

struct IndependenceCor <: CorStruct end

mutable struct ExchangeableCor <: CorStruct
    ar::Float64
end

function ExchangeableCor()
    ExchangeableCor(0.0)
end

function updatecor(c::IndependenceCor, sresid::FPVector, g::Array{Int,2}) end

function updatecor(c::ExchangeableCor, sresid::FPVector, g::Array{Int,2})

    ar, sp = 0.0, 0.0
    ma, ms = 0, 0

    for i = 1:size(g, 1)
        i1, i2 = g[i, 1], g[i, 2]
        for j1 = i1:i2
            sp += sresid[j1]^2
            for j2 = j1+1:i2
                ar += sresid[j1] * sresid[j2]
            end
        end
        q = i2 - i1 + 1
        ms += q
        ma += q * (q - 1) / 2
    end

    c.ar = (ar / ma) / (sp / ms)

end

function covsolve(c::IndependenceCor, sd::Array{T}, z::Array{T}) where {T<:Real}
    return Diagonal(1 ./ sd .^ 2) * z
end

function covsolve(c::ExchangeableCor, sd::Array{T}, z::Array{T}) where {T<:Real}
    a = c.ar
    p = length(sd)
    f = a / ((1 - a) * (1 + a * (p - 1)))
    di = Diagonal(1 ./ sd)
    x = di * z
    u = x ./ (1 - a)
    if length(size(z)) == 1
        u .= u .- f * sum(x) * ones(p)
    else
        u .= u .- f .* ones(p) * sum(x, dims = 1)
    end
    di * u
end

function corparams(c::IndependenceCor) end

function corparams(c::ExchangeableCor)
    return c.ar
end
