mutable struct GeneralizedEstimatingEquations2Model <: AbstractGEE

    # GEE model for the mean structure
    mean_model::GeneralizedEstimatingEquationsModel

    # GEE model for the variance structure
    var_model::GeneralizedEstimatingEquationsModel

    # GEE model for the correlation structure
    cor_model::Union{GeneralizedEstimatingEquationsModel,Nothing}

    # True if the model has been fit
    fit::Bool

    # True if the model has been fit and the fit was successful
    converged::Bool
end

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

# TODO GLM no longer has mueta
function mueta(sl::SigmoidLink, x::Real)
    return sl.range / ((1 + exp(-x)) * (1 + exp(x)))
end

function build_rcov(Xr, gi, make_rcov)

    n = (gi[2, :] - gi[1, :]) .+ 1
    m = Int(sum(x->x*(x-1)/2, n))

    # Make 1 row to get the length
    u = make_rcov(Xr[1, :], Xr[2, :])
    p = length(u)

    X = zeros(m, p)
    gr = zeros(m)

    ii = 1
    for j in 1:size(gi, 2)
        for i1 in gi[1, j]:gi[2, j]
            for i2 in i1+1:gi[2, j]
                X[ii, :] = make_rcov(Xr[i1, :], Xr[i2, :])
                gr[ii] = j
                ii += 1
            end
        end
    end

    return X, gr
end

function GeneralizedEstimatingEquations2Model(Xm::AbstractMatrix, Xv::AbstractMatrix, Xr::Union{Nothing,AbstractMatrix},
                                              y::AbstractVector, g::AbstractVector,
                                              make_rcov; link_mean=IdentityLink(), varfunc_mean=ConstantVar(), corstruct_mean=IndependenceCor(),
                                              link_var=LogLink(), varfunc_var=ConstantVar(), corstruct_var=IndependenceCor(),
                                              link_cor=SigmoidLink(-1, 1), varfunc_cor=ConstantVar(), corstruct_cor=IndependenceCor())

    if length(y) != length(g)
        error("'y' and 'g' must have the same length")
    end

    if length(y) != size(Xm, 1)
        error("The length of 'y' must be equal to the number of rows of 'Xm'")
    end

    if length(y) != size(Xv, 1)
        error("The length of 'y' must be equal to the number of rows of 'Xv'")
    end

    (gi, mg) = groupix(g)

    mean_model = fit(GeneralizedEstimatingEquationsModel, Xm, y, g; l=link_mean, v=varfunc_mean,
                     c=corstruct_mean, d=NoDistribution(), dofit=false)
    var_model = fit(GeneralizedEstimatingEquationsModel, Xv, zeros(length(y)), g; l=link_var, v=varfunc_var,
                    c=corstruct_var, d=NoDistribution(), dofit=false)

    if isnothing(Xr)
        cor_model = nothing
    else
        Xrm, gr = build_rcov(Xr, gi, make_rcov)
        cor_model = fit(GeneralizedEstimatingEquationsModel, Xrm, zeros(size(Xrm, 1)), gr; l=link_cor,
                        v=varfunc_cor, c=corstruct_cor, d=NoDistribution(), dofit=false)
    end

    return GeneralizedEstimatingEquations2Model(mean_model, var_model, cor_model, false, false)
end

function fit(::Type{GeneralizedEstimatingEquations2Model}, Xm::AbstractMatrix, Xv::AbstractMatrix, Xr::Union{Nothing,AbstractMatrix},
                                              y::AbstractVector, g::AbstractVector, make_rcov;
                                              link_mean=IdentityLink(), varfunc_mean=ConstantVar(), corstruct_mean=IndependenceCor(),
                                              link_var=LogLink(), varfunc_var=ConstantVar(), corstruct_var=IndependenceCor(),
                                              link_cor=SigmoidLink(-1, 1), varfunc_cor=ConstantVar(), corstruct_cor=IndependenceCor(), dofit=true,
                                              verbosity=0, maxiter=10)

    gee = GeneralizedEstimatingEquations2Model(Xm, Xv, Xr, y, g, make_rcov;
                                               link_mean=link_mean, varfunc_mean=varfunc_mean, corstruct_mean=corstruct_mean,
                                               link_var=link_var, varfunc_var=varfunc_var, corstruct_var=corstruct_var,
                                               link_cor=link_cor, varfunc_cor=varfunc_cor, corstruct_cor=corstruct_cor)

    if dofit
        fit!(gee, verbosity=verbosity, maxiter=maxiter)
    end

    return gee
end

function update_gee2!(gee::GeneralizedEstimatingEquations2Model; verbosity=0)

    (; mean_model, var_model, cor_model) = gee
    gi = mean_model.rr.grpix

    m = predict(mean_model; type=:response)

    resid = mean_model.rr.y - m
    var_model.rr.y .= resid.^2

    if !isnothing(cor_model)

        v = predict(var_model; type=:response)
        b = sum(x->x<0.01, v)
        if b > 0
            if verbosity > 1
                println("Clamping $(b) small variances")
            end
            v = clamp.(v, 0.01, Inf)
        end

        ii = 1
        for j in 1:size(gi, 2)
            for i1 in gi[1, j]:gi[2, j]
                for i2 in i1+1:gi[2, j]
                    cor_model.rr.y[ii] = resid[i1] * resid[i2] ./ sqrt(v[i1] * v[i2])
                    ii += 1
                end
            end
        end
    end
end

function fit!(gee::GeneralizedEstimatingEquations2Model; maxiter=10, verbosity=0)

    (; mean_model, var_model, cor_model) = gee

    for iter in 1:maxiter
        verbosity == 0 || println("Iteration $(iter)")
        verbosity <= 1 || println("  Fitting mean model...")
        fit!(mean_model; verbosity=verbosity, bccor=false)
        verbosity <= 1 || println("  Done fitting mean model")
        update_gee2!(gee; verbosity=verbosity)
        verbosity <= 1 || println("  Fitting variance model...")
        fit!(var_model; verbosity=verbosity, bccor=false)
        verbosity <= 1 || println("  Done fitting variance model")
        if !isnothing(cor_model)
            verbosity <= 1 || println("  Fitting correlation model...")
            update_gee2!(gee; verbosity=verbosity)
            fit!(cor_model; verbosity=verbosity, bccor=false)
            verbosity <= 1 || println("  Done fitting correlation model")
        end
    end

    gee.fit = true
    gee.converged = true
end

function show(io::IO, m::GeneralizedEstimatingEquations2Model)
    println(io, "Mean model:\n")
    show(io, m.mean_model)
    println(io, "Variance model:\n")
    show(io, m.var_model)
    if !isnothing(m.cor_model)
        println(io, "Correlation model:\n")
        show(io, m.cor_model)
    end
end
