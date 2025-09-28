@testitem "Check linear model versus R" setup=[Gendat] begin

    using StableRNGs
    using DataFrames
    using StatsModels
    using RCall

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

    m0 = fit(GeneralizedEstimatingEquationsModel, f, da, da[:, :id]; l=IdentityLink(), v=ConstantVar(), c=IndependenceCor(), atol=1e-12, rtol=1e-12)
    jc0 = coef(m0)
    jv0 = vcov(m0)

    @test isapprox(rc0, jc0, rtol=1e-4, atol=1e-6)
    @test isapprox(rv0, jv0; rtol=1e-3, atol=1e-3)

    m1 = fit(GeneralizedEstimatingEquationsModel, f, da, da[:, :id]; l=IdentityLink(), v=ConstantVar(), c=ExchangeableCor(), atol=1e-12, rtol=1e-12)
    jc1 = coef(m1)
    jv1 = vcov(m1)

    @test isapprox(rc1, jc1, rtol=1e-4, atol=1e-6)
    @test isapprox(rv1, jv1, rtol=1e-3, atol=1e-3)
end

@testitem "Check Poisson model versus R" setup=[Gendat] begin

    using StableRNGs
    using DataFrames
    using StatsModels
    using RCall

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

    m0 = fit(GeneralizedEstimatingEquationsModel, f, da, da[:, :id]; d=NoDistribution(), l=LogLink(), v=IdentityVar(), c=IndependenceCor(),
             atol=1e-12, rtol=1e-12)
    c0 = coef(m0)
    v0 = vcov(m0)
    @test isapprox(rc0, c0, rtol=1e-3, atol=1e-3)
    @test isapprox(rv0, v0, rtol=1e-3, atol=1e-3)

    m1 = fit(GeneralizedEstimatingEquationsModel, f, da, da[:, :id]; d=NoDistribution(), l=LogLink(), v=IdentityVar(), c=ExchangeableCor(),
             atol=1e-12, rtol=1e-12)
    c1 = coef(m1)
    v1 = vcov(m1)
    @test isapprox(rc1, c1, rtol=1e-3, atol=1e-3)
    @test isapprox(rv1, v1, rtol=1e-3, atol=1e-3)
end
