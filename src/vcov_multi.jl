function vcov(gee::GeneralizedEstimatingEquations2Model...)

    if !all(isfitted, gee)
        error("All models must be fitted before calling 'vcov'")
    end

    # The bread/meat of the individual models are computed lazily, so make sure that the
    # are present.
    for g in gee
        vcov(g)
    end

    B = vcov_bread(gee...)
    M = vcov_meat(gee...)

    vc = zeros(0, 0)
    try
        vc = B \ M / B'
    catch e
        @warn("Using pseudo-inverse to construct joint parameter covariance matrix")
        BI = pinv(B)
        vc = BI * M * BI'
    end

    return vc, B, M
end

"""
Compute the bread term for the joint variance/covariance matrix of several fitted GEE2 models.

# Arguments
- An arbitrary number of fitted GEE2 models.

# Returns
The bread matrix B.

The variance/covariance matrix is B \\ M / B', where B is the bread matrix and M is the
meat matrix.
"""
function vcov_bread(gee::GeneralizedEstimatingEquations2Model...)

    # The bread/meat of the individual models are computed lazily, so make sure that the
    # are present.
    for g in gee
        vcov(g)
    end

    # The number of models
    m = length(gee)

    # These are only used to get dimension information
    Xm1 = modelmatrix(gee[1].mean_model)
    Xv1 = modelmatrix(gee[1].scale_model)
    Xr1 = isnothing(gee[1].cor_model) ? zeros(0, 0) : modelmatrix(gee[1].cor_model)

    # These should be the same for all models.
    p, q = size(Xm1, 2), size(Xv1, 2)
    r = isnothing(gee[1].cor_model) ? 0 : size(Xr1, 2)
    t = p + q + r

    B = zeros(m*t, m*t)

    # The bread matrix is the block diagonal matrix for the bread matrices of the individual
    # models.
    for j in 1:m
        B[(j-1)*t+1:j*t, (j-1)*t+1:j*t] = gee[j].vcov.B
    end

    return B
end

"""
Compute the meat term for the joint variance/covariance matrix of several fitted GEE2 models.

# Arguments
- An arbitrary number of fitted GEE2 models.

# Returns
The meat matrix M.

The variance/covariance matrix is B \\ M / B, where B is the bread matrix and M is the
meat matrix.
"""
function vcov_meat(gee::GeneralizedEstimatingEquations2Model...)

    # The bread/meat of the individual models are computed lazily, so make sure that the
    # are present.
    for g in gee
        vcov(g)
    end

    # The number of models
    m = length(gee)

    # These are only used to get dimension information
    Xm1 = modelmatrix(gee[1].mean_model)
    Xv1 = modelmatrix(gee[1].scale_model)
    Xr1 = isnothing(gee[1].cor_model) ? zeros(0, 0) : modelmatrix(gee[1].cor_model)

    # These should be the same for all models
    p, q = size(Xm1, 2), size(Xv1, 2)
    r = isnothing(gee[1].cor_model) ? 0 : size(Xr1, 2)
    t = p + q + r

    M = zeros(m*t, m*t)

    # Fill in the matri block-wise.
    for j1 in 1:m

        # The diagonal blocks are the meat matrices for the individual
        # models, which have already been computed.
        M[(j1-1)*t+1:j1*t, (j1-1)*t+1:j1*t] = gee[j1].vcov.M

        # The between-model meat matrix.
        for j2 in j1+1:m
            M0 = vcov_meat_pair(gee[j1], gee[j2])
            M[(j1-1)*t+1:j1*t, (j2-1)*t+1:j2*t] = M0
            M[(j2-1)*t+1:j2*t, (j1-1)*t+1:j1*t] = M0'
        end
    end

    return M
end

"""
Compute the cross-model meat term for two fitted GEE2 models.

# Arguments
- An arbitrary number of fitted GEE2 models.

# Returns
The meat matrix M.

The variance/covariance matrix is B \\ M / B, where B is the bread matrix and M is the
meat matrix.
"""
function vcov_meat_pair(gee1::GeneralizedEstimatingEquations2Model, gee2::GeneralizedEstimatingEquations2Model)

    # The bread/meat of the individual models are computed lazily, so make sure that the
    # are present.
    vcov(gee1)
    vcov(gee2)

    # Only used to get dimension information
    Xm1 = modelmatrix(gee1.mean_model)
    Xv1 = modelmatrix(gee1.scale_model)
    Xr1 = isnothing(gee1.cor_model) ? zeros(0, 0) : modelmatrix(gee1.cor_model)

    # These should be the same between gee1 and gee2
    p, q = size(Xm1, 2), size(Xv1, 2)
    r = isnothing(gee1.cor_model) ? 0 : size(Xr1, 2)
    t = p + q + r

    # The meat of the cross-model sandwich covariance
    M = zeros(t, t)

    _iterprep(gee1.mean_model)
    _iterprep(gee2.mean_model)
    _iterprep(gee1.scale_model)
    _iterprep(gee2.scale_model)
    if !isnothing(gee1.cor_model)
        _iterprep(gee1.cor_model)
        _iterprep(gee2.cor_model)
    end

    # Index the correlation model groups, which may not be aligned with the mean model groups.
    jc = 0

    for j in 1:size(gee1.mean_model.rr.grpix, 2)

        # This should be the same for gee1 and gee2
        i1, i2 = gee1.mean_model.rr.grpix[:, j]

        _update_group(gee1.mean_model, j, false)
        _update_group(gee1.scale_model, j, false)
        _update_group(gee2.mean_model, j, false)
        _update_group(gee2.scale_model, j, false)

        # Update the meat for the mean and variance parameters
        M[1:p, 1:p] .+= gee1.mean_model.pp.score_grp * gee2.mean_model.pp.score_grp'
        M[p+1:p+q, 1:p] .+= gee1.scale_model.pp.score_grp * gee2.mean_model.pp.score_grp'
        M[p+1:p+q,p+1:p+q] .+= gee1.scale_model.pp.score_grp * gee2.scale_model.pp.score_grp'

        if isnothing(gee1.cor_model) || size(gee1.cor_pairs[j], 1) == 0
            continue
        end

        jc += 1
        _update_group(gee1.cor_model, jc, false)
        _update_group(gee2.cor_model, jc, false)

        # Update the meat for the correlation marameters
        M[p+q+1:p+q+r, 1:p] .+= gee1.cor_model.pp.score_grp * gee2. mean_model.pp.score_grp'
        M[p+q+1:p+q+r, p+1:p+q] .+= gee1.cor_model.pp.score_grp * gee2.scale_model.pp.score_grp'
        M[p+q+1:p+q+r, p+q+1:p+q+r] .+= gee1.cor_model.pp.score_grp * gee2.cor_model.pp.score_grp'
    end

    # Fill in the other blocks by transposing
    M[1:p, p+1:p+q] .= M[p+1:p+q, 1:p]'
    if !isnothing(gee1.cor_model)
        M[1:p, p+q+1:p+q+r] .= M[p+q+1:p+q+r, 1:p]'
        M[p+1:p+q, p+q+1:p+q+r] .= M[p+q+1:p+q+r, p+1:p+q]'
    end

    return M
end
