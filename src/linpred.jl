

mutable struct DensePred{T<:Real} <: LinPred

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

    "`score_grp`: the score vector for the current group"
    score_grp::Vector{T}

    "`xscale`: the columns of X are scaled by these values"
    xscale::Matrix{T}
end

function DensePred(X::Matrix{T}, mxg::Int; scalex::Bool=false) where {T<:Real}

    p = size(X, 2)

    if scalex
        xscale = std(X; dims=1)
        xscale[abs.(xscale) .< 1e-8] .= 1
        X ./= xscale
    else
        xscale = zeros(0, 0)
    end

    return DensePred{T}(X, vec(zeros(T, p)), vec(zeros(T, p)), mxg, zeros(0, 0),
                        zeros(p), zeros(p), xscale)
end

function nobs(pp::DensePred)
    return size(pp.X, 1)
end

function updateη!(p::DensePred, η::FPVector, off::FPVector)
    η .= p.X * p.beta0
    if length(off) > 0
        η .+= off
    end
end

# Update the model coefficients by taking a Gauss-Seidel step.  If `diagonalize` is true this is an approximate
# stepoo using only the diagonal elements of the Hessian matrix.  If bclip is positive, the elements of the step vector
# are truncated.
function update_coef!(p::DensePred, numer::Vector{T}, denom::Matrix{T}; diagonalize::Bool=false, bclip::Float64=-1.0) where {T<:Real}
    if diagonalize
        denom = Diagonal(diag(denom))
    end
    try
        p.delbeta .= denom \ numer
    catch e
        if isa(e, SingularException)
            @warn("Singularity encountered in update_coef!, using pseudo-inverse")
            p.delbeta .= pinv(denom) * numer
        else
            throw(e)
        end
    end
    if bclip >= 0
        p.delbeta .= clamp.(p.delbeta, -bclip, bclip)
    end
    p.beta0 .+= p.delbeta
end

function updateD!(p::DensePred, dμdη::FPVector, gr::UnitRange{Int})
    p.D = Diagonal(dμdη) * p.X[gr, :]
end
