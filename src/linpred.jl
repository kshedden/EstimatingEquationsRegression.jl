

mutable struct DensePred{T<:BlasReal} <: LinPred

    "`X`: the regression design matrix"
    X::Matrix{T}

    "`beta0`: the current parameter estimate"
    beta0::Vector{T}

    "`delbeta`: the increment to the parameter estimate"
    delbeta::Vector{T}

    "`mxg`: the maximum group size"
    mxg::Int

    "`D`: the Jacobian of the mean with respect to the coefficients"
    D::Matrix{T}

    "`score`: the current score vector"
    score::Vector{T}

    "`score_obs`: the score vector for the current group"
    score_obs::Vector{T}
end

function DensePred(X::Matrix{T}, mxg::Int) where {T<:BlasReal}
    p = size(X, 2)
    return DensePred{T}(
        X,
        vec(zeros(T, p)),
        vec(zeros(T, p)),
        mxg,
        zeros(0, 0),
        zeros(p),
        zeros(p),
    )
end

function updateη!(p::DensePred, η::FPVector, off::FPVector)
    η .= p.X * p.beta0
    if length(off) > 0
        η .+= off
    end
end

function updateβ!(p::DensePred, numer::Array{T}, denom::Array{T}) where {T<:Real}
    p.delbeta .= denom \ numer
    p.beta0 .= p.beta0 + p.delbeta
end

function updateD!(p::DensePred, dμdη::FPVector, i1::Int, i2::Int)
    p.D = Diagonal(dμdη) * p.X[i1:i2, :]
end
