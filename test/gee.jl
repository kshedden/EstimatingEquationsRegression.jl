@testset "Check offset" begin

    y, _, X, g, df = data1()
    rng = StableRNG(123)
    offset = rand(rng, length(y))

    # In a Gaussian linear model, using an offset is the same as shifting the response.
    m0 = fit(GeneralizedEstimatingEquationsModel, X, y, g; c=ExchangeableCor())
    m1 = fit(GeneralizedEstimatingEquationsModel, X, y+offset, g; c=ExchangeableCor(), offset=offset)
    @test isapprox(coef(m0), coef(m1))
    @test isapprox(vcov(m0), vcov(m1))

    # A constant offset only changes the intercept and does not change the
    # standard errors.
    X[:, 1] .= 1
    offset = ones(length(y))
    for fam in [Normal, Poisson]
        m0 = fit(GeneralizedEstimatingEquationsModel, X, y, g; d=fam(), c=IndependenceCor())
        m1 = fit(GeneralizedEstimatingEquationsModel, X, y, g; d=fam(), c=IndependenceCor(), offset=offset)
        @test isapprox(coef(m0)[1], coef(m1)[1] + 1)
        @test isapprox(coef(m0)[2:end], coef(m1)[2:end])
        @test isapprox(vcov(m0), vcov(m1))
    end
end

@testset "Equivalence of distribution-based and variance function-based interfaces (Gaussian/linear)" begin

    y, _, X, g, df = data1()

    # Without formulas
    m1 = fit(GeneralizedEstimatingEquationsModel, X, y, g; c=ExchangeableCor())
    m2 = fit(GeneralizedEstimatingEquationsModel, X, y, g; d=NoDistribution(), v=ConstantVar(), c=ExchangeableCor())
    @test isapprox(coef(m1), coef(m2))
    @test isapprox(vcov(m1), vcov(m2))
    @test isapprox(corparams(m1), corparams(m2))

    # With formulas
    f = @formula(y ~ x1 + x2 + x3)
    m1 = gee(f, df, g; c=ExchangeableCor())
    m2 = gee(f, df, g; d=NoDistribution(), v=ConstantVar(), c=ExchangeableCor())
    @test isapprox(coef(m1), coef(m2))
    @test isapprox(vcov(m1), vcov(m2))
    @test isapprox(corparams(m1), corparams(m2))
end

@testset "Equivalence of distribution-based and variance function-based interfaces (Binomial/logit)" begin

    _, y, X, g, df = data1()
    cs = ExchangeableCor()

    # Without formulas
    m1 = fit(GeneralizedEstimatingEquationsModel, X, y, g; d=Binomial(), c=cs)
    m2 = fit(GeneralizedEstimatingEquationsModel, X, y, g; l=LogitLink(),
             d=NoDistribution(), v=BinomialVar(), c=cs)
    @test isapprox(coef(m1), coef(m2))
    @test isapprox(vcov(m1), vcov(m2))
    @test isapprox(corparams(m1), corparams(m2))

    # With formulas
    f = @formula(z ~ x1 + x2 + x3)
    m1 = gee(f, df, g; d=Binomial(), c=cs)
    m2 = gee(f, df, g; d=NoDistribution(), c=cs, l=LogitLink(), v=BinomialVar())
    @test isapprox(coef(m1), coef(m2))
    @test isapprox(vcov(m1), vcov(m2))
    @test isapprox(corparams(m1), corparams(m2))
end

@testset "Equivalence of distribution-based and variance function-based interfaces (Poisson/log)" begin

    y, _, X, g, df = data1()

    # Without formulas
    m1 = fit(GeneralizedEstimatingEquationsModel, X, y, g; d=Poisson(), c=ExchangeableCor())
    m2 = fit(GeneralizedEstimatingEquationsModel, X, y, g; d=NoDistribution(), l=LogLink(), v=IdentityVar(), c=ExchangeableCor())
    @test isapprox(coef(m1), coef(m2))
    @test isapprox(vcov(m1), vcov(m2), atol=1e-5, rtol=1e-5)
    @test isapprox(corparams(m1), corparams(m2))

    # With formulas
    f = @formula(y ~ x1 + x2 + x3)
    m1 = gee(f, df, g; d=Poisson(), c=ExchangeableCor())
    m2 = gee(f, df, g; d=NoDistribution(), l=LogLink(), v=IdentityVar(), c=ExchangeableCor())
    @test isapprox(coef(m1), coef(m2))
    @test isapprox(vcov(m1), vcov(m2), atol = 1e-5, rtol = 1e-5)
    @test isapprox(corparams(m1), corparams(m2))
end

@testset "linear/normal autoregressive model" begin

    y, _, X, g, _ = data1()
    m = fit(GeneralizedEstimatingEquationsModel, X, y, g; d=Normal(), c=AR1Cor())
    se = sqrt.(diag(vcov(m)))

    @test isapprox(coef(m), [-0.0049, 0.7456, 0.2844], atol=1e-4)
    @test isapprox(se, [0.015, 0.023, 0.002], rtol=1e-3, atol=1e-3)
    @test isapprox(dispersion(m), 0.699, atol=1e-3)
    @test isapprox(corparams(m), -0.696, atol=1e-3)
end

@testset "logit/binomial autoregressive model" begin

    _, z, X, g, _ = data1()
    m = fit(GeneralizedEstimatingEquationsModel, X, z, g; d=Binomial(), c=AR1Cor())
    se = sqrt.(diag(vcov(m)))

    @test isapprox(coef(m), [0.5693, -0.1835, -0.9295], atol=1e-4)
    @test isapprox(se, [0.101, 0.125, 0.153], rtol=1e-3, atol=1e-3)
    @test isapprox(dispersion(m), 1, atol=1e-3)
    @test isapprox(corparams(m), -0.163, atol=1e-3)
end

@testset "log/Poisson autoregressive model" begin

    y, _, X, g, _ = data1()
    m = fit(GeneralizedEstimatingEquationsModel, X, y, g; d=Poisson(), c=AR1Cor())
    se = sqrt.(diag(vcov(m)))

    @test isapprox(coef(m), [-0.0135, 0.3025, 0.0413], atol=1e-4)
    @test isapprox(se, [0.002, 0.025, 0.029], rtol=1e-3, atol=1e-3)
    @test isapprox(dispersion(m), 1, atol=1e-3)
    @test isapprox(corparams(m), -0.722, atol=1e-3)
end

@testset "log/Gamma autoregressive model" begin

    y, _, X, g, _ = data1()
    m = fit(GeneralizedEstimatingEquationsModel, X, y, g; d=Gamma(), c=AR1Cor(), l=LogLink())
    se = sqrt.(diag(vcov(m)))

    @test isapprox(coef(m), [-0.0221, 0.3091, 0.0516], atol=1e-4)
    @test isapprox(se, [0.006, 0.022, 0.026], rtol=1e-3, atol=1e-3)
    @test isapprox(dispersion(m), 0.118, atol=1e-3)
    @test isapprox(corparams(m), -0.7132, atol=1e-3)
end

@testset "AR1 covsolve" begin

    Random.seed!(123)
    makeAR = (r, d) -> [r^abs(i - j) for i = 1:d, j = 1:d]

    for d in [1, 2, 4]
        for q in [1, 3]

            c = AR1Cor(0.4)
            v = q == 1 ? randn(d) : randn(d, q)
            sd = rand(d)
            mu = rand(d)
            sm = Diagonal(sd)

            mat = makeAR(0.4, d)
            vi = (sm \ (mat \ (sm \ v)))
            vi2 = EstimatingEquationsRegression.covsolve(c, mu, sd, v)
            @test isapprox(vi, vi2)

        end
    end
end

@testset "Exchangeable covsolve" begin

    Random.seed!(123)
    makeEx = (r, d) -> [i == j ? 1 : r for i = 1:d, j = 1:d]

    for d in [1, 2, 4]
        for q in [1, 3]

            c = ExchangeableCor(0.4)
            v = q == 1 ? randn(d) : randn(d, q)
            mu = rand(d)
            sd = rand(d)
            sm = Diagonal(sd)

            mat = makeEx(0.4, d)
            vi = (sm \ (mat \ (sm \ v)))
            vi2 = EstimatingEquationsRegression.covsolve(c, mu, sd, v)
            @test isapprox(vi, vi2)
        end
    end
end

@testset "OrdinalIndependence covsolve" begin

    Random.seed!(123)

    c = OrdinalIndependenceCor(2)

    mu = [0.2, 0.3, 0.4, 0.5]
    sd = mu .* (1 .- mu)
    rhs = Array{Float64}(I(4))

    rslt = EstimatingEquationsRegression.covsolve(c, mu, sd, rhs)
    rslt = inv(rslt)
    @test isapprox(rslt[1:2, 3:4], zeros(2, 2))
    @test isapprox(rslt, rslt')
    @test isapprox(diag(rslt), mu .* (1 .- mu))
end

@testset "linear/normal independence model" begin

    y, _, X, g, _ = data1()
    m = fit(GeneralizedEstimatingEquationsModel, X, y, g; d=Normal(), c=IndependenceCor())
    se = sqrt.(diag(vcov(m)))
    @test isapprox(coef(m), [0.0397, 0.7499, 0.2147], atol = 1e-4)
    @test isapprox(se, [0.089, 0.089, 0.021], atol = 1e-3)
    @test isapprox(dispersion(m), 0.673, atol = 1e-4)

    # Check fitting using formula/dataframe
    df = DataFrame(X, [@sprintf("x%d", j) for j = 1:size(X, 2)])
    df[!, :g] = g
    df[!, :y] = y
    f = @formula(y ~ 0 + x1 + x2 + x3)
    m1 = fit(GeneralizedEstimatingEquationsModel, f, df, g; d=Normal(), c=IndependenceCor(), l=IdentityLink(), bccor=true)
    m2 = gee(f, df, g; d=Normal(), c=IndependenceCor(), l=IdentityLink())
    se1 = sqrt.(diag(vcov(m1)))
    se2 = sqrt.(diag(vcov(m2)))
    @test isapprox(coef(m), coef(m1), atol = 1e-8)
    @test isapprox(se, se1, atol = 1e-8)
    @test isapprox(coef(m), coef(m2), atol = 1e-8)
    @test isapprox(se, se2, atol = 1e-8)

    # With independence correlation, GLM and GEE have the same parameter estimates
    m0 = glm(X, y, Normal(), IdentityLink())
    @test isapprox(coef(m), coef(m0), atol = 1e-5)

    # Test Mancl-DeRouen bias-corrected covariance
    md = [
        0.109898 -0.107598 -0.031721
        -0.107598 0.128045 0.043794
        -0.031721 0.043794 0.016414
    ]
    @test isapprox(vcov(m1, cov_type = "md"), md, atol = 1e-5)

    # Score test
    m = fit(GeneralizedEstimatingEquationsModel, X, y, g; d=Normal(), c=IndependenceCor(), l=IdentityLink(), dofit=false)
    subm = fit(GeneralizedEstimatingEquationsModel, X[:, [1, 2]], y, g; d=Normal(), c=IndependenceCor(), l=IdentityLink())
    cst = scoretest(m, subm)
    @test isapprox(cst.stat, 2.01858, atol = 1e-4)
    @test isapprox(dof(cst), 1)
    @test isapprox(pvalue(cst), 0.155385, atol = 1e-5)

    # Score test
    m = fit(GeneralizedEstimatingEquationsModel, X, y, g; d=Normal(), c=IndependenceCor(), l=IdentityLink(), dofit=false)
    subm = fit(GeneralizedEstimatingEquationsModel, X[:, [2]], y, g; d=Normal(), c=IndependenceCor(), l=IdentityLink())
    cst = scoretest(m, subm)
    @test isapprox(cst.stat, 2.80908, atol = 1e-4)
    @test isapprox(dof(cst), 2)
    @test isapprox(pvalue(cst), 0.24548, atol = 1e-4)
end

@testset "logit/Binomial independence model" begin
    _, y, X, g, _ = data1()
    m = fit(GeneralizedEstimatingEquationsModel, X, y, g; d=Binomial(), c=IndependenceCor(), l=LogitLink())
    se = sqrt.(diag(vcov(m)))
    @test isapprox(coef(m), [0.5440, -0.2293, -0.8340], atol = 1e-4)
    @test isapprox(se, [0.121, 0.144, 0.178], atol = 1e-3)
    @test isapprox(dispersion(m), 1)

    # With independence correlation, GLM and GEE have the same parameter estimates
    m0 = glm(X, y, Binomial(), LogitLink())
    @test isapprox(coef(m), coef(m0), atol = 1e-5)

    # Score test
    m = fit(GeneralizedEstimatingEquationsModel, X, y, g; d=Binomial(), c=IndependenceCor(), l=LogitLink(), dofit=false)
    subm = fit(GeneralizedEstimatingEquationsModel, X[:, [1, 2]], y, g; d=Binomial(), c=IndependenceCor(), l=LogitLink())
    cst = scoretest(m, subm)
    @test isapprox(cst.stat, 2.53019, atol = 1e-4)
    @test isapprox(dof(cst), 1)
    @test isapprox(pvalue(cst), 0.11169, atol = 1e-5)

    # Score test
    m = fit(GeneralizedEstimatingEquationsModel, X, y, g; d=Binomial(), c=IndependenceCor(), l=LogitLink(), dofit=false)
    subm = fit(GeneralizedEstimatingEquationsModel, X[:, [2]], y, g; d=Binomial(), c=IndependenceCor(), l=LogitLink())
    cst = scoretest(m, subm)
    @test isapprox(cst.stat, 2.77068, atol = 1e-4)
    @test isapprox(dof(cst), 2)
    @test isapprox(pvalue(cst), 0.25024, atol = 1e-4)
end

@testset "log/Poisson independence model" begin
    y, _, X, g, _ = data1()
    m = fit(GeneralizedEstimatingEquationsModel, X, y, g; d=Poisson(), c=IndependenceCor(), l=LogLink())
    se = sqrt.(diag(vcov(m)))
    @test isapprox(coef(m), [0.0051, 0.2777, 0.0580], atol = 1e-4)
    @test isapprox(se, [0.020, 0.033, 0.014], atol = 1e-3)
    @test isapprox(dispersion(m), 1)

    # With independence correlation, GLM and GEE have the same parameter estimates
    m0 = glm(X, y, Poisson(), LogLink())
    @test isapprox(coef(m), coef(m0), atol = 1e-5)

    # Score test
    m = fit(GeneralizedEstimatingEquationsModel, X, y, g; d=Poisson(), c=IndependenceCor(), l=LogLink(), dofit=false)
    subm = fit(GeneralizedEstimatingEquationsModel, X[:, [1, 2]], y, g; d=Poisson(), c=IndependenceCor(), l=LogLink())
    cst = scoretest(m, subm)
    @test isapprox(cst.stat, 2.600191, atol = 1e-4)
    @test isapprox(dof(cst), 1)
    @test isapprox(pvalue(cst), 0.106851, atol = 1e-5)

    # Score test
    m = fit(GeneralizedEstimatingEquationsModel, X, y, g; d=Poisson(), c=IndependenceCor(), l=LogLink(), dofit=false)
    subm = fit(GeneralizedEstimatingEquationsModel, X[:, [2]], y, g; d=Poisson(), c=IndependenceCor(), l=LogLink())
    cst = scoretest(m, subm)
    @test isapprox(cst.stat, 2.94147, atol = 1e-4)
    @test isapprox(dof(cst), 2)
    @test isapprox(pvalue(cst), 0.229757, atol = 1e-5)
end

@testset "log/Gamma independence model" begin
    y, _, X, g, _ = data1()
    m = fit(GeneralizedEstimatingEquationsModel, X, y, g; d=Gamma(), c=IndependenceCor(), l=LogLink())
    se = sqrt.(diag(vcov(m)))
    @test isapprox(coef(m), [-0.0075, 0.2875, 0.0725], atol = 1e-4)
    @test isapprox(se, [0.019, 0.034, 0.006], atol = 1e-3)
    @test isapprox(dispersion(m), 0.104, atol = 1e-3)

    # With independence correlation, GLM and GEE have the same parameter estimates
    m0 = glm(X, y, Gamma(), LogLink())
    @test isapprox(coef(m), coef(m0), atol = 1e-5)

    # Score test
    m = fit(GeneralizedEstimatingEquationsModel, X, y, g; d=Gamma(), c=IndependenceCor(), l=LogLink(), dofit=false)
    subm = fit(GeneralizedEstimatingEquationsModel, X[:, [1, 2]], y, g; d=Gamma(), c=IndependenceCor(), l=LogLink())
    cst = scoretest(m, subm)
    @test isapprox(cst.stat, 2.471939, atol = 1e-4)
    @test isapprox(dof(cst), 1)
    @test isapprox(pvalue(cst), 0.115895, atol = 1e-5)

    # Score test
    m = fit(GeneralizedEstimatingEquationsModel, X, y, g; d=Gamma(), c=IndependenceCor(), l=LogLink(), dofit=false)
    subm = fit(GeneralizedEstimatingEquationsModel, X[:, [2]], y, g; d=Gamma(), c=IndependenceCor(), l=LogLink())
    cst = scoretest(m, subm)
    @test isapprox(cst.stat, 2.99726, atol = 1e-4)
    @test isapprox(dof(cst), 2)
    @test isapprox(pvalue(cst), 0.223437, atol = 1e-5)
end

@testset "frequency weights" begin

    Random.seed!(432)
    X = randn(6, 2)
    X[:, 1] .= 1
    y = X[:, 1] + randn(6)
    y[6] = y[5]
    X[6, :] = X[5, :]

    g = [1.0, 1, 1, 2, 2, 2]
    m1 = fit(GeneralizedEstimatingEquationsModel, X, y, g; d=Normal())
    se1 = sqrt.(diag(vcov(m1)))

    fwts = [1.0, 1, 1, 1, 1, 1]
    m2 = fit(GeneralizedEstimatingEquationsModel, X, y, g; d=Normal(), fwts=fwts)
    se2 = sqrt.(diag(vcov(m2)))

    fwts = [1.0, 1, 1, 1, 2]
    X1 = X[1:5, :]
    y1 = y[1:5]
    g1 = g[1:5]
    m3 = fit(GeneralizedEstimatingEquationsModel, X1, y1, g1; d=Normal(), fwts=fwts)
    se3 = sqrt.(diag(vcov(m3)))

    @test isapprox(coef(m1), coef(m2))
    @test isapprox(coef(m1), coef(m3))
    @test isapprox(se1, se2)
    @test isapprox(se1, se3)
    @test isapprox(dispersion(m1), dispersion(m2))
    @test isapprox(dispersion(m1), dispersion(m3))
end

@testset "linear/normal exchangeable model" begin

    y, _, X, g, _ = data1()
    m = fit(GeneralizedEstimatingEquationsModel, X[:, 1:1], y, g; d=Normal(), c=ExchangeableCor())
    se = sqrt.(diag(vcov(m)))
    @test isapprox(coef(m), [0.2718], atol = 1e-4)
    @test isapprox(se, [0.037], atol = 1e-3)
    @test isapprox(dispersion(m), 3.915, atol = 1e-3)
    @test isapprox(corparams(m), 0.428, atol = 1e-3)

    # Holding the exchangeable correlation parameter fixed at zero should give the same
    # result as fitting with the independence correlation model.
    m1 = fit(GeneralizedEstimatingEquationsModel, X, y, g; d=Normal(), c=IndependenceCor())
    se1 = sqrt.(diag(vcov(m1)))
    m2 = fit(GeneralizedEstimatingEquationsModel, X, y, g; d=Normal(), c=ExchangeableCor(0), fitcor=false)
    se2 = sqrt.(diag(vcov(m2)))

    @test isapprox(coef(m1), coef(m2), atol = 1e-7)
    @test isapprox(se1, se2, atol = 1e-7)

    # Hold the parameters fixed at the GLM estimates, then estimate the exchangeable
    # correlation parameter.
    m3 = fit(GeneralizedEstimatingEquationsModel, X, y, g; d=Normal(), c=ExchangeableCor(), fitcoef=false)

    @test isapprox(coef(m1), coef(m3), atol = 1e-6)
    @test isapprox(corparams(m3), 0, atol = 1e-4)

    # Hold the parameters fixed at zero, then estimate the exchangeable correlation parameter.
    m4 = fit(GeneralizedEstimatingEquationsModel, X, y, g; d=Normal(), c=ExchangeableCor(), start=[0.0, 0, 0], fitcoef=false)

    @test isapprox(coef(m4), [0, 0, 0], atol = 1e-6)
    @test isapprox(corparams(m4), 0.6409037, atol = 1e-4)
end

@testset "log/Poisson exchangeable model" begin
    y, _, X, g, _ = data1()
    m = fit(GeneralizedEstimatingEquationsModel, X[:, 1:1], y, g; d=Poisson(), c=ExchangeableCor(0), l=LogLink(), fit_cor=false)
    se = sqrt.(diag(vcov(m)))
    @test isapprox(coef(m), [0.1423], atol = 1e-4)
    @test isapprox(se, [0.021], atol = 1e-3)
    @test isapprox(dispersion(m), 1)
    @test isapprox(corparams(m), 0.130, atol = 1e-3)
end

@testset "log/Gamma exchangeable model" begin
    y, _, X, g, _ = data1()
    m = fit(GeneralizedEstimatingEquationsModel, X, y, g; d=Gamma(), c=ExchangeableCor(), l=LogLink())
    se = sqrt.(diag(vcov(m)))
    @test isapprox(coef(m), [-0.0075, 0.2875, 0.0725], atol = 1e-4)
    @test isapprox(se, [0.019, 0.034, 0.006], atol = 1e-3)
    @test isapprox(dispersion(m), 0.104, atol = 1e-3)
    @test isapprox(corparams(m), 0, atol = 1e-3)
end

@testset "logit/Binomial exchangeable model" begin
    _, z, X, g, _ = data1()
    m = fit(GeneralizedEstimatingEquationsModel, X, z, g; d=Binomial(), c=ExchangeableCor(), l=LogitLink())
    se = sqrt.(diag(vcov(m)))
    @test isapprox(coef(m), [0.5440, -0.2293, -0.8340], atol = 1e-4)
    @test isapprox(se, [0.121, 0.144, 0.178], atol = 1e-3)
    @test isapprox(dispersion(m), 1)
    @test isapprox(corparams(m), 0, atol = 1e-3)
end
