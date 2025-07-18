mutable struct GeneralizedEstimatingEquations2Model <: AbstractGEE

    # GEE model for the mean structure
    mean_model::GeneralizedEstimatingEquationsModel

    # GEE model for the scale parameter
    scale_model::GeneralizedEstimatingEquationsModel

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
                                              link_scale=LogLink(), varfunc_scale=IdentityVar(), corstruct_scale=IndependenceCor(),
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
    scale_model = fit(GeneralizedEstimatingEquationsModel, Xv, zeros(length(y)), g; l=link_scale, v=varfunc_scale,
                    c=corstruct_scale, d=NoDistribution(), dofit=false)

    if isnothing(Xr)
        cor_model = nothing
    else
        Xrm, gr = build_rcov(Xr, gi, make_rcov)
        cor_model = fit(GeneralizedEstimatingEquationsModel, Xrm, zeros(size(Xrm, 1)), gr; l=link_cor,
                        v=varfunc_cor, c=corstruct_cor, d=NoDistribution(), dofit=false)
    end

    return GeneralizedEstimatingEquations2Model(mean_model, scale_model, cor_model, false, false)
end

function fit(::Type{GeneralizedEstimatingEquations2Model}, Xm::AbstractMatrix, Xv::AbstractMatrix, Xr::Union{Nothing,AbstractMatrix},
                                              y::AbstractVector, g::AbstractVector, make_rcov;
                                              link_mean=IdentityLink(), varfunc_mean=ConstantVar(), corstruct_mean=IndependenceCor(),
                                              link_scale=LogLink(), varfunc_scale=IdentityVar(), corstruct_scale=IndependenceCor(),
                                              link_cor=SigmoidLink(-1, 1), varfunc_cor=ConstantVar(), corstruct_cor=IndependenceCor(), dofit=true,
                                              verbosity=0, maxiter=10)

    gee = GeneralizedEstimatingEquations2Model(Xm, Xv, Xr, y, g, make_rcov;
                                               link_mean=link_mean, varfunc_mean=varfunc_mean, corstruct_mean=corstruct_mean,
                                               link_scale=link_scale, varfunc_scale=varfunc_scale, corstruct_scale=corstruct_scale,
                                               link_cor=link_cor, varfunc_cor=varfunc_cor, corstruct_cor=corstruct_cor)

    if dofit
        fit!(gee, verbosity=verbosity, maxiter=maxiter)
    end

    return gee
end

function update_gee2!(gee::GeneralizedEstimatingEquations2Model; verbosity=0)

    (; mean_model, scale_model, cor_model) = gee
    gi = mean_model.rr.grpix

    # Fitted mean
    mn = predict(mean_model; type=:response)

    # Raw residuals
    resid = mean_model.rr.y - mn

    # Variance contribution from the mean/variance relationship
    vm = geevar.(varfunc(mean_model), mn)

    # Scale parameter
    scl = predict(scale_model; type=:response)

    # Fitted variance
    va = vm .* scl

    # Update the response for the scale parameter estimation
    scale_model.rr.y .= resid.^2 ./ vm

    # Update the response for correlation estimation
    if !isnothing(cor_model)

        b = sum(x->x<0.01, va)
        if b > 0
            if verbosity > 1
                println("Clamping $(b) small variances")
            end
            va = clamp.(va, 0.01, Inf)
        end

        ii = 1
        for j in 1:size(gi, 2)
            for i1 in gi[1, j]:gi[2, j]
                for i2 in i1+1:gi[2, j]
                    cor_model.rr.y[ii] = resid[i1] * resid[i2] ./ sqrt(va[i1] * va[i2])
                    ii += 1
                end
            end
        end
    end
end

function fit!(gee::GeneralizedEstimatingEquations2Model; maxiter=20, verbosity=0, eps=1e-5)

    (; mean_model, scale_model, cor_model) = gee

    converged = false

    cm = zeros(size(modelmatrix(mean_model), 2))
    cv = zeros(size(modelmatrix(scale_model), 2))
    cr = isnothing(cor_model) ? [] : zeros(size(modelmatrix(cor_model), 2))

    for iter in 1:maxiter
        verbosity == 0 || println("Iteration $(iter)")
        verbosity <= 1 || println("  Fitting mean model...")
        fit!(mean_model; verbosity=verbosity, bccor=false)
        verbosity <= 1 || println("  Done fitting mean model")
        update_gee2!(gee; verbosity=verbosity)
        verbosity <= 1 || println("  Fitting variance model...")
        fit!(scale_model; verbosity=verbosity, bccor=false)
        verbosity <= 1 || println("  Done fitting variance model")
        if !isnothing(cor_model)
            verbosity <= 1 || println("  Fitting correlation model...")
            update_gee2!(gee; verbosity=verbosity)
            fit!(cor_model; verbosity=verbosity, bccor=false)
            verbosity <= 1 || println("  Done fitting correlation model")
        end

        # Check convergence
        if iter > 1
            fm = norm(coef(mean_model) - cm)
            fv = norm(coef(scale_model) - cv)
            fr = isnothing(cor_model) ? 0.0 : norm(coef(cor_model) - cr)
            if max(fm, fv, fr) < eps
                converged = true
                break
            end
        end
        cm = coef(mean_model)
        cv = coef(scale_model)
        cr = isnothing(cor_model) ? [] : coef(cor_model)
    end

    if !converged && verbosity > 0
        println("GEE2 did not converge")
    end

    gee.fit = true
    gee.converged = converged
end

# Return the triangular indices betgween j1 and j2, must match the order used in update_gee2!.
function tri_indices(j1, j2)
    p = j2 - j1 + 1
    m = Int(p*(p-1)/2)
    I1 = zeros(Int, m, 2)
    I2 = zeros(Int, m, 2)
    ii = 1

    for i1 in j1:j2
        for i2 in i1+1:j2
            I1[ii, :] = [i1, i2]
            I2[ii, :] = [i1-j1+1, i2-j1+1]
            ii += 1
        end
    end
    return I1, I2
end

function vcov(gee::GeneralizedEstimatingEquations2Model; cov_type::String="")

    (; mean_model, scale_model, cor_model) = gee

    Xm = modelmatrix(mean_model)
    Xv = modelmatrix(scale_model)
    Xr = isnothing(cor_model) ? zeros(0, 0) : modelmatrix(cor_model)

    # The fitted mean
    mn = predict(mean_model; type=:response)

    # The fitted scale parameter
    scale = predict(scale_model; type=:response)

    v = geevar.(varfunc(mean_model), mn)
    vd = geevarderiv.(varfunc(mean_model), mn)
    va = v .* scale

    p, q = size(Xm, 2), size(Xv, 2)
    r = isnothing(cor_model) ? 0 : size(Xr, 2)
    t = p + q + r

    # The bread and meat of the sandwich covariance
    B = zeros(t, t)
    M = zeros(t, t)

    # We already have the diagonal blocks of B.
    B[1:p, 1:p] .= mean_model.cc.DtViD
    B[p+1:p+q, p+1:p+q] = scale_model.cc.DtViD
    if !isnothing(cor_model)
        B[p+q+1:p+q+r, p+q+1:p+q+r] = cor_model.cc.DtViD
    end

    _iterprep(mean_model)
    _iterprep(scale_model)
    isnothing(cor_model) && _iterprep(cor_model)

    for j in 1:size(mean_model.rr.grpix, 2)
        i1, i2 = mean_model.rr.grpix[:, j]
        _update_group(mean_model, j, false)
        _update_group(scale_model, j, false)
        w = length(mean_model.rr.wts) > 0 ? mean_model.rr.wts[i1:i2] : zeros(0)

        # Update the meat for the mean and variance parameters
        M[1:p, 1:p] .+= mean_model.pp.score_obs * mean_model.pp.score_obs'
        M[p+1:p+q, 1:p] .+= scale_model.pp.score_obs * mean_model.pp.score_obs'
        M[p+1:p+q,p+1:p+q] .+= scale_model.pp.score_obs * scale_model.pp.score_obs'

        # Update the bread for the mean and variance parameters
        # This is the B matrix of Yan and Fine
        # U is partial s / partial beta
        u = -2 * mean_model.rr.resid[i1:i2] .* v[i1:i2] .- mean_model.rr.resid[i1:i2].^2 .* vd[i1:i2]
        U = (Diagonal(u) * mean_model.pp.D) ./ v[i1:i2].^2
        C = covsolve(scale_model.qq.cor, scale_model.rr.mu[i1:i2], scale_model.rr.sd[i1:i2], w, U)
        B[p+1:p+q, 1:p] .-= scale_model.pp.D' * C

        if isnothing(cor_model)
            continue
        end

        j1, j2 = cor_model.rr.grpix[:, j]
        _update_group(cor_model, j, false)

        # Update the meat for the correlation marameters
        M[p+q+1:p+q+r, 1:p] .+= cor_model.pp.score_obs * mean_model.pp.score_obs'
        M[p+q+1:p+q+r, p+1:p+q] .+= cor_model.pp.score_obs * scale_model.pp.score_obs'
        M[p+q+1:p+q+r, p+q+1:p+q+r] .+= cor_model.pp.score_obs * cor_model.pp.score_obs'

        # Update the bread for the correlation parameters
        ix1, ix2 = tri_indices(i1, i2)

        wr = length(cor_model.rr.wts) > 0 ? cor_model.rr.wts[i1:i2] : zeros(0)

        # Bread matrix D in Yan and Fine
        qp = mean_model.rr.resid[ix1[:, 1]] .* mean_model.rr.resid[ix1[:, 2]]
        u = -mean_model.rr.resid[ix1[:, 2]] - 0.5 * qp .* vd[ix1[:, 1]] ./ v[ix1[:, 1]]
        U = Diagonal(u) * mean_model.pp.D[ix2[:, 1]]
        u = -mean_model.rr.resid[ix1[:, 1]] - 0.5 * qp .* vd[ix1[:, 2]] ./ v[ix1[:, 2]]
        U .+= Diagonal(u) * mean_model.pp.D[ix2[:, 2]]
        qf = sqrt.(va[ix1[:, 1]] .* va[ix1[:, 2]])
        U = Diagonal(1 ./ qf) * U
        C = covsolve(cor_model.qq.cor, cor_model.rr.mu[j1:j2], cor_model.rr.sd[j1:j2], wr, U)
        B[p+q+1:p+q+r, 1:p] .-= cor_model.pp.D' * C

        # Bread matrix E in Yan and Fine
        U = Diagonal(1 ./ scale[ix1[:, 1]]) * scale_model.pp.D[ix2[:, 1], :]
        U .+= Diagonal(1 ./ scale[ix1[:, 2]]) * scale_model.pp.D[ix2[:, 2], :]
        U = Diagonal(-0.5 * qp ./ qf) * U
        C = covsolve(cor_model.qq.cor, cor_model.rr.mu[j1:j2], cor_model.rr.sd[j1:j2], w, U)
        B[p+q+1:p+q+r, p+1:p+q] .-= cor_model.pp.D' * C
    end

    # Fill in the other blocks by transposing
    M[1:p, p+1:p+q] .= M[p+1:p+q, 1:p]'
    if !isnothing(cor_model)
        M[1:p, p+q+1:p+q+r] .= M[p+q+1:p+q+r, 1:p]'
        M[p+1:p+q, p+q+1:p+q+r] .= M[p+q+1:p+q+r, p+1:p+q]'
    end

    return B \ M / B'
end

function coef(mm::GeneralizedEstimatingEquations2Model)
    (; mean_model, scale_model, cor_model) = mm
    c = vcat(coef(mean_model), coef(scale_model))
    if !isnothing(cor_model)
        c = vcat(c, coef(cor_model))
    end
    return c
end

function stderror(mm::GeneralizedEstimatingEquations2Model; cov_type::String="robust")
    V = vcov(mm)
    return sqrt.(diag(V))
end

function coeftable(mm::GeneralizedEstimatingEquations2Model; level::Real = 0.95, cov_type::String="robust")

    (; mean_model, scale_model, cor_model) = mm

    p, q = size(modelmatrix(mean_model), 2), size(modelmatrix(scale_model), 2)
    r = isnothing(cor_model) ? 0 : size(modelmatrix(cor_model), 2)

    # Generic covariate names
    vnames1 = ["Mean X$(i)" for i in 1:p]
    vnames2 = ["Scale X$(i)" for i in 1:q]
    vnames3 = ["Cor X$(i)" for i in 1:r]
    vnames = vcat(vnames1, vnames2, vnames3)

    cc = coef(mm)
    se = stderror(mm; cov_type=cov_type)
    zz = cc ./ se
    p = 2 * ccdf.(Ref(Normal()), abs.(zz))
    ci = se * quantile(Normal(), (1 - level) / 2)
    levstr = isinteger(level * 100) ? string(Integer(level * 100)) : string(level * 100)
    CoefTable(
        hcat(cc, se, zz, p, cc + ci, cc - ci),
        ["Coef.", "Std. Error", "z", "Pr(>|z|)", "Lower $levstr%", "Upper $levstr%"],
        vnames,
        4,
        3,
    )
end

function nobs(mm::GeneralizedEstimatingEquations2Model)
    return nobs(mm.mean_model)
end

function show(io::IO, mm::GeneralizedEstimatingEquations2Model)

    (; mean_model, scale_model, cor_model) = mm

    nr = isnothing(cor_model) ? 0 : nobs(cor_model)

    grpix = mean_model.rr.grpix
    gsize = 1 .+ grpix[2, :] - grpix[1, :]
    ngrp = size(grpix, 2)

    ta = [@sprintf("Num obs: %d", nobs(mm)) "Num obs (cor): $(nr)";
          "Num groups: $(ngrp)" @sprintf("Avg grp size: %.1f", mean(gsize));
          @sprintf("Min grp size: %d", minimum(gsize)) @sprintf("Max grp size: %d", maximum(gsize))]
    pretty_table(io, ta; tf=tf_borderless, show_header=false, alignment=:l)
   show(io, coeftable(mm))
end
