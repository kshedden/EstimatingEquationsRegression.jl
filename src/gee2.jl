mutable struct GEE2VCov{T<:Real}
    vcov::Matrix{T}
    M::Matrix{T}
    B::Matrix{T}
end

mutable struct GeneralizedEstimatingEquations2Model <: AbstractGEE

    # GEE model for the mean structure
    mean_model::GeneralizedEstimatingEquationsModel

    # GEE model for the scale parameter
    scale_model::GeneralizedEstimatingEquationsModel

    # GEE model for the correlation structure
    cor_model::Union{GeneralizedEstimatingEquationsModel,Nothing}

    # Indices of observation pairs for estimating correlation parameters.
    # The indices are offset to the beginning of a group.
    cor_pairs::Vector{Matrix{Int}}

    # True if the model has been fit
    fit::Bool

    # True if the model has been fit and the fit was successful
    converged::Bool

    # The underlying TableRegression of the mean model, if it was
    # specified using a formula.
    mean_tablemodel::Union{StatsModels.TableRegressionModel,Nothing}

    # The underlying TableRegression of the scale model, if it was
    # specified using a formula.
    scale_tablemodel::Union{StatsModels.TableRegressionModel,Nothing}

    # The variance/covariance matrix of the parameters, or an empty
    # matrix if it has not yet been constructed.
    vcov::GEE2VCov
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

function build_rcov(Xr, fwts, gi, cp, make_rcov)

    m = sum(x->size(x, 2), cp)

    # Make 1 row to get the length
    u = make_rcov(Xr[1, :], Xr[2, :])
    p = length(u)

    X = zeros(m, p)
    fwtsr = zeros(m)
    gr = zeros(Int, m)

    ii = 1
    for j in eachindex(cp)
        for k in 1:size(cp[j], 2)
            i1, i2 = cp[j][:, k]

            # Convert from offsets to array positions
            i1 += first(gi[j]) - 1
            i2 += first(gi[j]) - 1
            X[ii, :] = make_rcov(Xr[i1, :], Xr[i2, :])
            gr[ii] = j
            fwtsr[ii] = sqrt(fwts[i1] * fwts[i2])
            ii += 1
        end
    end

    return X, fwtsr, gr
end

function gee2_check_arrays(y, g, Xm, Xv, Xr)
    if length(y) != length(g)
        error("'y' and 'g' must have the same length, but 'y' has length $(length(y)) and 'g' has length $(length(g))")
    end

    if length(y) != size(Xm, 1)
        error("The length of 'y' must be equal to the number of rows of 'Xm'")
    end

    if length(y) != size(Xv, 1)
        error("The length of 'y' must be equal to the number of rows of 'Xv'")
    end

    if !isnothing(Xr) && length(y) != size(Xr, 1)
        error("The length of 'y' must be equal to the number of rows of 'Xr'")
    end
end

function gee2_check_arrays(g, Xm, Xv, Xr)
    if length(g) != size(Xm, 1)
        error("The length of 'g' must be equal to the number of rows of 'Xm'")
    end

    if length(g) != size(Xv, 1)
        error("The length of 'g' must be equal to the number of rows of 'Xv'")
    end

    if !isnothing(Xr) && length(g) != size(Xr, 1)
        error("The length of 'g' must be equal to the number of rows of 'Xr'")
    end
end

function GeneralizedEstimatingEquations2Model(Xm::Union{AbstractMatrix,DataFrame}, Xv::Union{AbstractMatrix,DataFrame},
                                              Xr::Union{Nothing,AbstractMatrix}, y::AbstractVector, g::AbstractVector,
                                              make_rcov; fml_mean=nothing, link_mean=IdentityLink(), varfunc_mean=ConstantVar(),
                                              corstruct_mean=IndependenceCor(), offset_mean=Float64[], fml_scale=nothing,
                                              link_scale=LogLink(), varfunc_scale=IdentityVar(), corstruct_scale=IndependenceCor(),
                                              link_cor=SigmoidLink(-1, 1), varfunc_cor=ConstantVar(),
                                              corstruct_cor=IndependenceCor(), cor_pair_factor::Float64=0, fwts=nothing)

    mean_tablemodel = nothing
    scale_tablemodel = nothing

    fwtsv = if isnothing(fwts)
        n = size(Xm, 1)
        ones(n)
    else
        fwts
    end

    if isnothing(fml_mean)
        gee2_check_arrays(y, g, Xm, Xv, Xr)
        mean_model = fit(GeneralizedEstimatingEquationsModel, Xm, y, g; l=link_mean, v=varfunc_mean,
                         c=corstruct_mean, offset=offset_mean, d=NoDistribution(), dofit=false, fwts=fwtsv)
    elseif typeof(fml_mean) <: AbstractTerm
        gee2_check_arrays(g, Xm, Xv, Xr)
        mean_tablemodel = fit(GeneralizedEstimatingEquationsModel, fml_mean, Xm, g; l=link_mean, v=varfunc_mean,
                              c=corstruct_mean, offset=offset_mean, d=NoDistribution(), dofit=false, fwts=fwtsv)
        mean_model = mean_tablemodel.model
    else
        error("Invalid type $(typeof(fml_mean)) for 'fml_mean'")
    end

    n = length(response(mean_model))
    if isnothing(fml_scale)
        gee2_check_arrays(g, Xm, Xv, Xr)
        scale_model = fit(GeneralizedEstimatingEquationsModel, Xv, zeros(n), g; l=link_scale, v=varfunc_scale,
                          c=corstruct_scale, d=NoDistribution(), dofit=false, fwts=fwtsv)
    elseif typeof(fml_scale) <: AbstractTerm
        gee2_check_arrays(g, Xm, Xv, Xr)
        Xv = copy(Xv) # don't add columns to the caller's dataframe
        Xv[:, :squared_residual] = zeros(n)
        fml_scale = term(:squared_residual) ~ fml_scale.rhs
        scale_tablemodel = fit(GeneralizedEstimatingEquationsModel, fml_scale, Xv, g; l=link_scale, v=varfunc_scale,
                              c=corstruct_scale, d=NoDistribution(), dofit=false, fwts=fwtsv)
        scale_model = scale_tablemodel.model
    else
        error("Invalid type $(typeof(fml_scale)) for 'fml_scale'")
    end

    if isnothing(Xr)
        cp = Vector{Matrix{Int}}(undef, 0)
        cor_model = nothing
    else
        gi, mg = groupix(g)
        cp = make_cor_pairs(gi, cor_pair_factor)
        Xrm, fwtsr, gr = build_rcov(Xr, fwtsv, gi, cp, make_rcov)
        cor_model = fit(GeneralizedEstimatingEquationsModel, Xrm, zeros(size(Xrm, 1)), gr; l=link_cor,
                        v=varfunc_cor, c=corstruct_cor, d=NoDistribution(), dofit=false, fwts=fwtsr)
    end

    vcov = GEE2VCov(zeros(0, 0), zeros(0, 0), zeros(0, 0))
    return GeneralizedEstimatingEquations2Model(mean_model, scale_model, cor_model, cp, false, false,
                mean_tablemodel, scale_tablemodel, vcov)
end

function fit(::Type{GeneralizedEstimatingEquations2Model}, Xm::Union{AbstractMatrix,AbstractDataFrame},
             Xv::Union{AbstractMatrix,AbstractDataFrame}, Xr::Union{Nothing,AbstractMatrix}, y::AbstractVector, g::AbstractVector, make_rcov;
             fml_mean=nothing, link_mean=IdentityLink(), offset_mean=Float64[], varfunc_mean=ConstantVar(), corstruct_mean=IndependenceCor(),
             fml_scale=nothing, link_scale=LogLink(), varfunc_scale=PowerVar(2), corstruct_scale=IndependenceCor(),
             link_cor=SigmoidLink(-1, 1), varfunc_cor=ConstantVar(), corstruct_cor=IndependenceCor(), dofit=true,
             verbosity=0, maxiter=10, cor_pair_factor::Float64=0.0, fwts=nothing)

    gee = GeneralizedEstimatingEquations2Model(Xm, Xv, Xr, y, g, make_rcov; fml_mean=fml_mean,
                                               link_mean=link_mean, varfunc_mean=varfunc_mean, corstruct_mean=corstruct_mean,
                                               offset_mean=offset_mean, link_scale=link_scale, varfunc_scale=varfunc_scale,
                                               fml_scale=fml_scale, corstruct_scale=corstruct_scale,
                                               link_cor=link_cor, varfunc_cor=varfunc_cor, corstruct_cor=corstruct_cor,
                                               cor_pair_factor=cor_pair_factor, fwts=fwts)

    if dofit
        fit!(gee, verbosity=verbosity, maxiter=maxiter)
    end

    return gee
end

function update_gee2!(gee::GeneralizedEstimatingEquations2Model, stage; verbosity=0)

    (; mean_model, scale_model, cor_model, cor_pairs) = gee
    gix = mean_model.rr.grpix

    # Fitted mean
    mn = predict(mean_model; type=:response)

    # Raw residuals
    resid = mean_model.rr.y - mn

    # Variance contribution from the mean/variance relationship
    awts1 = ones(length(mn))
    vm = geevar.(varfunc(mean_model), mn, awts1)

    b = sum(x->x<0.01, vm)
    if b > 0
        if verbosity > 1
            println("Clamping $(b) small variances")
        end
        vm = clamp.(vm, 0.01, Inf)
    end

    # Update the response for the scale parameter estimation
    scale_model.rr.y .= resid.^2 ./ vm

    if stage == 1
        return
    end

    # Scale parameter
    scl = predict(scale_model; type=:response)

    # Fitted variance
    va = vm .* scl

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
        for (j,gr) in enumerate(gix)
            ix = cor_pairs[j]
            resid1 = resid[gr]
            va1 = va[gr]
            for j in 1:size(ix, 2)
                i1, i2 = ix[:, j]
                cor_model.rr.y[ii] = resid1[i1] * resid1[i2] ./ sqrt(va1[i1] * va1[i2])
                ii += 1
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

        mean_model.isfitted = false
        scale_model.isfitted = false
        if !isnothing(cor_model)
            cor_model.isfitted = false
        end

        if iter > 1
            va = predict(scale_model; type=:response)
            va .= clamp.(va, 1e-4, Inf)
            mean_model.rr.awts .= 1 ./ va
        end

        verbosity == 0 || println("Iteration $(iter)")
        verbosity <= 1 || println("  Fitting mean model...")
        fit!(mean_model; verbosity=verbosity, inference=false)
        verbosity <= 1 || println("  Done fitting mean model")
        verbosity <= 2 || println("Mean model:\n", mean_model)

        update_gee2!(gee, 1; verbosity=verbosity)
        verbosity <= 1 || println("  Fitting scale model...")
        fit!(scale_model; verbosity=verbosity, inference=false)
        verbosity <= 1 || println("  Done fitting scale model")
        verbosity <= 2 || println("Scale model:\n", scale_model)

        if !isnothing(cor_model)
            verbosity <= 1 || println("  Fitting correlation model...")
            update_gee2!(gee, 2; verbosity=verbosity)
            fit!(cor_model; verbosity=verbosity, inference=false)
            verbosity <= 2 || println("Correlation model:\n", cor_model)
            verbosity <= 1 || println("  Done fitting correlation model")
        end

        # Check convergence
        if iter > 1
            fm = norm(coef(mean_model) - cm)
            fv = norm(coef(scale_model) - cv)
            fr = isnothing(cor_model) ? 0.0 : norm(coef(cor_model) - cr)
            if max(fm, fv, fr) < eps
                converged = true
                if verbosity > 0
                    println("GEE2 converged in $(iter) iterations")
                end
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

# For each group, determine the indices of observation pairs that are used to estimate
# correlation parameters.
function make_cor_pairs(gi::Vector{UnitRange{Int}}, cor_pair_factor::Float64)
    cp = Vector{Matrix{Int}}(undef, length(gi))
    for (j,gr) in enumerate(gi)
        if cor_pair_factor == 0
            cp[j] = tri_indices(gr)
        else
            cp[j] = tri_indices_sample(gr, cor_pair_factor)
        end
    end
    return cp
end

# Return offsets for all lower triangular indices between j1 and j2.
# The columns of the returned array contain offsets relative
# to (j1, j2) that cover all distinct pairs between j1 and j2,
# inclusive.
function tri_indices(gr)
    p = first(size(gr))
    m = div(p*(p-1), 2)
    I = zeros(Int, 2, m)
    ii = 1

    for i1 in gr
        for i2 in i1+1:last(gr)
            I[:, ii] = [i1, i2]
            ii += 1
        end
    end

    # Make it an offset relative to the beginning of the group.
    I .-= first(gr) - 1

    return I
end

# Return a sample of lower triangular indices between j1 and j2, with size
# equal to the group size times cor_pair_factor.
function tri_indices_sample(gr, cor_pair_factor)
    p = first(size(gr))
    m0 = div(p*(p-1), 2)

    # Number of pairs to sample
    m = Int(floor(cor_pair_factor * p))

    if m >= m0
        # Use direct enumeration
        return tri_indices(gr)
    end

    # Random positions in the lower triangle
    J = sample(1:m0, m; replace=false)
    I = zeros(Int, 2, m)
    for (ii,jj) in enumerate(J)
        # Get the row and column for each random position
        p = Int(floor((sqrt(1 + 8*(jj - 1)) - 1) / 2))
        i1 = p + 2
        i2 = Int(jj - 1 - p * (p + 1) / 2 + 1)
        I[:, ii] = [i1, i2]
    end
    return I
end

function vcov(gee::GeneralizedEstimatingEquations2Model; cov_type::String="")

    if size(gee.vcov.vcov, 1) > 0
        # The vcov has already been computed
        return gee.vcov.vcov
    end

    (; mean_model, scale_model, cor_model) = gee

    Xm = modelmatrix(mean_model)
    Xv = modelmatrix(scale_model)
    Xr = isnothing(cor_model) ? zeros(0, 0) : modelmatrix(cor_model)

    # The fitted mean
    mn = predict(mean_model; type=:response)

    # The fitted scale parameter
    scale = predict(scale_model; type=:response)

    # Frequency weights
    fwtsm = weights(mean_model; type=:frequency)
    fwtsr = isnothing(cor_model) ? Float64[] : weights(cor_model; type=:frequency)

    # The variance function and its derivative, do not include
    # the scale function via analytic weights here.
    awts1 = ones(length(mn))
    v = geevar.(varfunc(mean_model), mn, awts1)
    vd = geevarderiv.(varfunc(mean_model), mn, awts1)

    # The variance of the data, accounting for both the
    # variance function and the scale function.
    va = v .* scale

    p, q = size(Xm, 2), size(Xv, 2)
    r = isnothing(cor_model) ? 0 : size(Xr, 2)
    t = p + q + r

    # The bread and meat of the sandwich covariance
    B = zeros(t, t)
    M = zeros(t, t)

    # We already have the diagonal blocks of B.
    B[1:p, 1:p] .= mean_model.cc.DtViD_sum
    B[p+1:p+q, p+1:p+q] = scale_model.cc.DtViD_sum
    if !isnothing(cor_model)
        B[p+q+1:p+q+r, p+q+1:p+q+r] = cor_model.cc.DtViD_sum
    end

    _iterprep(mean_model)
    _iterprep(scale_model)
    if !isnothing(cor_model)
        _iterprep(cor_model)
    end

    # Index the correlation model groups, which may not be aligned with the mean model groups.
    jc = 0

    for j in eachindex(mean_model.rr.grpix)
        gr1 = mean_model.rr.grpix[j]
        _update_group(mean_model, j, false)
        _update_group(scale_model, j, false)

        v1 = v[gr1]
        vd1 = vd[gr1]
        va1 = va[gr1]
        scale1 = scale[gr1]
        resid1 = mean_model.rr.resid[gr1]

        # Update the meat for the mean and variance parameters
        M[1:p, 1:p] .+= mean_model.pp.score_grp * mean_model.pp.score_grp'
        M[p+1:p+q, 1:p] .+= scale_model.pp.score_grp * mean_model.pp.score_grp'
        M[p+1:p+q,p+1:p+q] .+= scale_model.pp.score_grp * scale_model.pp.score_grp'

        # Update the bread for the mean and variance parameters
        # This is the B matrix of Yan and Fine
        # U is partial s / partial beta
        u = -2 * resid1 .* v1 .- resid1.^2 .* vd1
        U = (Diagonal(u) * mean_model.pp.D) ./ v1.^2
        C = covsolve(scale_model.qq.cor, scale_model.rr.mu[gr1], scale_model.rr.sd[gr1], U)
        B[p+1:p+q, 1:p] .-= scale_model.pp.D' * Diagonal(fwtsm[gr1]) * C

        if isnothing(cor_model) || size(gee.cor_pairs[j], 2) == 0
            # Nothing more to do if there is no correlation model or if this group
            # has no pairs.
            continue
        end

        jc += 1
        gr2 = cor_model.rr.grpix[jc]
        _update_group(cor_model, jc, false)

        cor_mu1 = cor_model.rr.mu[gr2]
        cor_sd1 = cor_model.rr.sd[gr2]

        # Update the meat for the correlation marameters
        M[p+q+1:p+q+r, 1:p] .+= cor_model.pp.score_grp * mean_model.pp.score_grp'
        M[p+q+1:p+q+r, p+1:p+q] .+= cor_model.pp.score_grp * scale_model.pp.score_grp'
        M[p+q+1:p+q+r, p+q+1:p+q+r] .+= cor_model.pp.score_grp * cor_model.pp.score_grp'

        # Update the bread for the correlation parameters
        ix = gee.cor_pairs[j]

        # Bread matrix D in Yan and Fine
        qp = resid1[ix[1, :]] .* resid1[ix[2, :]]
        u = -resid1[ix[2, :]] - 0.5 * qp .* vd1[ix[1, :]] ./ v1[ix[1, :]]
        U = Diagonal(u) * mean_model.pp.D[ix[1, :], :]
        u = -resid1[ix[1, :]] - 0.5 * qp .* vd1[ix[2, :]] ./ v1[ix[2, :]]
        U .+= Diagonal(u) * mean_model.pp.D[ix[2, :], :]
        qf = sqrt.(va1[ix[1, :]] .* va1[ix[2, :]])
        U = Diagonal(1 ./ qf) * U
        C = covsolve(cor_model.qq.cor, cor_mu1, cor_sd1, U)
        B[p+q+1:p+q+r, 1:p] .-= cor_model.pp.D' * Diagonal(fwtsr[gr2]) * C

        # Bread matrix E in Yan and Fine
        U = Diagonal(1 ./ scale1[ix[1, :]]) * scale_model.pp.D[ix[1, :], :]
        U .+= Diagonal(1 ./ scale1[ix[2, :]]) * scale_model.pp.D[ix[2, :], :]
        U = Diagonal(-0.5 * qp ./ qf) * U
        C = covsolve(cor_model.qq.cor, cor_mu1, cor_sd1, U)
        B[p+q+1:p+q+r, p+1:p+q] .-= cor_model.pp.D' * Diagonal(fwtsr[gr2]) * C
    end

    # Fill in the other blocks by transposing
    M[1:p, p+1:p+q] .= M[p+1:p+q, 1:p]'
    if !isnothing(cor_model)
        M[1:p, p+q+1:p+q+r] .= M[p+q+1:p+q+r, 1:p]'
        M[p+1:p+q, p+q+1:p+q+r] .= M[p+q+1:p+q+r, p+1:p+q]'
    end

    try
        gee.vcov.vcov = B \ M / B'
        gee.vcov.B = B
        gee.vcov.M = M
    catch e
        println("Using pseudo-inverse to construct parameter covariance matrix")
        BI = pinv(B)
        gee.vcov.vcov = BI * M * BI'
        gee.vcov.B = B
        gee.vcov.M = M
    end

    return gee.vcov.vcov
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

    vnames = coefnames(mm)
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
    gsize = [first(size(x)) for x in grpix]
    ngrp = length(grpix)

    ta = [@sprintf("Num obs: %d", nobs(mm)) "Num obs (cor): $(nr)";
          "Num groups: $(ngrp)" @sprintf("Avg grp size: %.1f", mean(gsize));
          @sprintf("Min grp size: %d", minimum(gsize)) @sprintf("Max grp size: %d", maximum(gsize))]

    tf = TextTableFormat(borders=text_table_borders__compact)
    pretty_table(io, ta; table_format=tf, show_column_labels=false)
    show(io, coeftable(mm))
end

function coefnames(mm::GeneralizedEstimatingEquations2Model)
    p = size(modelmatrix(mm.mean_model), 2)
    q = size(modelmatrix(mm.scale_model), 2)
    r = isnothing(mm.cor_model) ? 0 : size(modelmatrix(mm.cor_model), 2)
    vnames_mean = isnothing(mm.mean_tablemodel) ? ["Mean X$(i)" for i in 1:p] : ["Mean $(x)" for x in coefnames(mm.mean_tablemodel)]
    vnames_scale = isnothing(mm.scale_tablemodel) ? ["Scale X$(i)" for i in 1:p] : ["Scale $(x)" for x in coefnames(mm.scale_tablemodel)]
    vnames_cor = ["Cor X$(i)" for i in 1:r]
    vnames = vcat(vnames_mean, vnames_scale, vnames_cor)
    return vnames
end

function isfitted(gee::GeneralizedEstimatingEquations2Model)
    f = isfitted(gee.mean_model) & isfitted(gee.scale_model)
    if !isnothing(gee.cor_model)
        f = f & isfitted(gee.cor_model)
    end
    return f
end

function residuals(m::GeneralizedEstimatingEquations2Model)
    return residuals(m.mean_model)
end
