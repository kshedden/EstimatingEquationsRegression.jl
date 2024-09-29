
function gendat(ngroup, gsize, p, r, rng, dist)

    # Sample size per group
    n = 1 .+ rand(Poisson(gsize), ngroup)
    N = sum(n)

    # Group labels
    id = vcat([fill(i, n[i]) for i in eachindex(n)]...)

    # Random intercepts
    ri = randn(ngroup)
    ri = ri[id]

    X = randn(N, p)
    for j in 2:p
        X[:, j] = r*X[:, j-1] + sqrt(1-r^2)*X[:, j]
    end

    lp = X[:, 1] - 2*X[:, 2]

    if dist == :Gaussian
        ey = lp
        y = ey + ri + randn(N)
    elseif dist == :Poisson
        ey = exp.(0.2*lp)
        e = (ri + randn(N)) / sqrt(2)
        u = map(Base.Fix1(cdf, Normal()), e)
        y = quantile.(Poisson.(ey), u)
    else
        error("Invalid distribution")
    end

    return (id=id, X=X, y=y, ey=ey)
end

@testset "Check linear model versus R" begin

    rng = StableRNG(123)

    d = gendat(100, 10, 10, 0.4, rng, :Gaussian)

    (; id, y, X, ey) = d

    @rput y
    @rput X
    @rput id

    R"
    library(geepack)
    da = data.frame(y=y, x1=X[,1], x2=X[,2], x3=X[,3], x4=X[,4], x5=X[,5], id=id)
    m0 = geeglm(y ~ x1 + x2 + x3 + x4 + x5, corstr='independence', id=id, data=da)
    rc0 = coef(m0)
    rv0 = vcov(m0)
    m1 = geeglm(y ~ x1 + x2 + x3 + x4 + x5, corstr='exchangeable', id=id, data=da)
    rc1 = coef(m1)
    rv1 = vcov(m1)
    "

    @rget rc0
    @rget rv0
    @rget rc1
    @rget rv1

    da = DataFrame(y=y, x1=X[:,1], x2=X[:,2], x3=X[:,3], x4=X[:,4], x5=X[:,5], id=id)

    f = @formula(y ~ x1 + x2 + x3 + x4 + x5)

    m0 = gee(f, da, da[:, :id], IdentityLink(), ConstantVar(), IndependenceCor(), atol=1e-12, rtol=1e-12)
    m1 = gee(f, da, da[:, :id], IdentityLink(), ConstantVar(), ExchangeableCor(), atol=1e-12, rtol=1e-12)

    jc0 = coef(m0)
    jc1 = coef(m1)
    jv0 = vcov(m0)
    jv1 = vcov(m1)

    @test isapprox(rc0, jc0)
    @test isapprox(rc1, jc1, rtol=1e-4, atol=1e-6)

    @test isapprox(rv0, jv0)
    @test isapprox(rv1, jv1, rtol=1e-3, atol=1e-6)
end

@testset "Check Poisson model versus R" begin

    rng = StableRNG(123)

    d = gendat(100, 10, 10, 0.4, rng, :Poisson)

    (; id, y, X, ey) = d

    @rput y
    @rput X
    @rput id

    R"
    library(geepack)
    da = data.frame(y=y, x1=X[,1], x2=X[,2], x3=X[,3], x4=X[,4], x5=X[,5], id=id)

    # Fit the model with independence correlation
    m0 = geeglm(y ~ x1 + x2 + x3 + x4 + x5, family=poisson, corstr='independence', id=id, data=da)
    rc0 = coef(m0)
    rv0 = vcov(m0)

    # Fit the model with exchangeable correlation
    m1 = geeglm(y ~ x1 + x2 + x3 + x4 + x5, family=poisson, corstr='exchangeable', id=id, data=da)
    rc1 = coef(m1)
    rv1 = vcov(m1)
    "

    @rget rc0
    @rget rv0
    @rget rc1
    @rget rv1

    da = DataFrame(y=y, x1=X[:,1], x2=X[:,2], x3=X[:,3], x4=X[:,4], x5=X[:,5], id=id)

    f = @formula(y ~ x1 + x2 + x3 + x4 + x5)

    m0 = fit(GeneralizedEstimatingEquationsModel, f, da, da[:, :id], LogLink(), IdentityVar(), IndependenceCor();
             atol=1e-12, rtol=1e-12)
    m1 = fit(GeneralizedEstimatingEquationsModel, f, da, da[:, :id], LogLink(), IdentityVar(), ExchangeableCor();
             atol=1e-12, rtol=1e-12)

    c0 = coef(m0)
    c1 = coef(m1)
    v0 = vcov(m0)
    v1 = vcov(m1)

    @test isapprox(rc0, c0)
    @test isapprox(rc1, c1, rtol=1e-3, atol=1e-6)

    @test isapprox(rv0, v0)
    @test isapprox(rv1, v1, rtol=1e-3, atol=1e-6)
end
