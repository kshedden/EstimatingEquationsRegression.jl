using LinearAlgebra: svd
using Distributions: Chisq, cdf

function _score_transforms(model::AbstractGEE, submodel::AbstractGEE)

    xm = modelmatrix(model)
    xs = modelmatrix(submodel)

    u, s, v = svd(xm)

    # xm * qm = xs
    si = Diagonal(1 ./ s)
    qm = v * si * u' * xs

    # Check that submodel is actually a submodel
    e = norm(xs - xm * qm)
    e < 1e-8 || throw(error("scoretest submodel is not a submodel"))

    # Get the orthogonal complement of xs in xm.
    a, _, _ = svd(xs)
    a = u - a * (a' * u)
    xsc, sb, _ = svd(a)
    xsc = xsc[:, sb.>1e-12]

    qc = v * si * u' * xsc

    (qm, qc)
end

struct ScoreTestResult <: HypothesisTest
    dof::Integer
    stat::Float64
    pvalue::Float64
end

function pvalue(st::ScoreTestResult)
    return st.pvalue
end

function dof(st::ScoreTestResult)
    return st.dof
end

"""
    scoretest(model::AbstractGEE, submodel::AbstractGEE)

GEE score test comparing submodel to model.  model need not have
been fit before calling scoretest.
"""
function scoretest(model::AbstractGEE, submodel::AbstractGEE)

    # Contents of model will be altered so copy it.
    model = deepcopy(model)

    xm = modelmatrix(model)
    xs = modelmatrix(submodel)

    # Checks for whether test is appropriate
    size(xm, 1) == size(xs, 1) ||
        throw(error("scoretest models must have same number of rows"))
    size(xs, 2) < size(xm, 2) ||
        throw(error("scoretest submodel must have smaller rank than parent model"))
    typeof(Distribution(model)) == typeof(Distribution(submodel)) ||
        throw(error("scoretest models must have same distributions"))
    typeof(Corstruct(model)) == typeof(Corstruct(submodel)) ||
        throw(error("scoretest models must have same correlation structures"))
    typeof(Link(model)) == typeof(Link(submodel)) ||
        throw(error("scoretest models must have same link functions"))
    typeof(Varfunc(model)) == typeof(Varfunc(submodel)) ||
        throw(error("scoretest models must have same link functions"))

    qm, qc = _score_transforms(model, submodel)

    # Submodel coefficients embedded into parent model coordinates.
    coef_ex = qm * coef(submodel)

    # The score vector of the parent model, evaluated at the fitted
    # coefficients of the submodel
    pp, rr, qq, cc = model.pp, model.rr, model.qq, model.cc
    pp.beta0 = coef_ex
    _iterprep(pp, rr, qq)
    _iterate(pp, rr, qq, cc, true)
    score = model.pp.score
    score2 = qc' * score

    amat = cc.nacov
    scrcov = cc.scrcov

    bmat11 = qm' * scrcov * qm
    bmat22 = qc' * scrcov * qc
    bmat12 = qm' * scrcov * qc

    amat11 = qm' * amat * qm
    amat12 = qm' * amat * qc

    scov = bmat22 - amat12' * (amat11 \ bmat12)
    scov = scov .- bmat12' * (amat11 \ amat12)
    scov = scov .+ amat12' * (amat11 \ bmat11) * (amat11 \ amat12)

    stat = score2' * (pinv(scov) * score2)
    dof = length(score2)
    pvalue = 1 - cdf(Chisq(dof), stat)

    return ScoreTestResult(dof, stat, pvalue)
end
