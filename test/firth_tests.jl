@testitem "Basic Firth check" begin

    using StableRNGs
    using DataFrames
    using StatsModels

    rng = StableRNG(123)

    y = [1, 0, 0, 1, 1, 1, 0, 0]
    X = [1 3; 1 2; 1 4; 1 6; 1 9; 1 4; 1 6; 1 6]

    # Test with arrays
    m = fit(FirthModel, X, y; l=LogitLink(), v=BinomialVar())
    b = coef(m)
    br = [-0.8610757, 0.1697547]
    @test isapprox(b, br, atol=1e-5, rtol=1e-5)
    rhat = [0.2450868, 0.3830251, 0.1578064, 0.1607244,
            0.5741021, 0.1578064, 0.1607244, 0.1607244]
    @test isapprox(m.resp.hat, rhat, atol=1e-5, rtol=1e-5)
    vc = [2.6948019 -0.4514064; -0.4514064  0.0893284]
    @test isapprox(vcov(m), vc, atol=1e-5, rtol=1e-5)

    # Test with formula
    dd = DataFrame(y=y, x=X[:, 2])
    fml = @formula(y ~ x)
    m1 = fit(FirthModel, fml, dd; l=LogitLink(), v=BinomialVar())
    @test isapprox(coef(m), coef(m1))
    @test isapprox(vcov(m), vcov(m1))

    # Crude test of displaying the results
    s = IOBuffer()
    show(s, m1)
    s = String(take!(s))
    @test length(s) > 20
end

@testitem "Check Firth with weights" begin

    using StableRNGs
    using StatsModels

    rng = StableRNG(123)

    y = [1, 0, 0, 1, 1, 1, 0, 0]
    X = [1 3; 1 2; 1 4; 1 6; 1 9; 1 4; 1 6; 1 6]
    w = [1, 1, 1, 1, 2, 2, 2, 2]

    m1 = fit(FirthModel, X, y; fwts=w, l=LogitLink(), v=BinomialVar())

    yy = vcat(y, y[5:8])
    XX = vcat(X, X[5:8, :])
    m2 = fit(FirthModel, XX, yy; l=LogitLink(), v=BinomialVar())

    @test isapprox(coef(m1), coef(m2), atol=1e-5, rtol=1e-5)
    @test isapprox(vcov(m1), vcov(m2), atol=1e-5, rtol=1e-5)
end

@testitem "Test Firth logistic versus R" begin

    using StableRNGs
    using RCall

    rng = StableRNG(123)

    p = 3

    for n in [20, 50]
        for k in 1:10
            X = randn(rng, n, p)
            X[:, 1] .= 1
            beta = rand(rng, p)
            lp = X * beta
            mn = 1 ./ (1 .+ exp.(-lp))
            y = [rand(rng) < p ? 1 : 0 for p in mn]

            m = fit(FirthModel, X, y; l=LogitLink(), v=BinomialVar())
            b = coef(m)
            hat = m.resp.hat
            vc = vcov(m)

            @rput y
            @rput X

            R"
            library(logistf)
            dd = data.frame(y=y, x1=X[,2], x2=X[,3])
            m = logistf(y ~ x1 + x2, data=dd)
            br = coef(m)
            hatr = m$hat.diag
            vcr = vcov(m)
            "

            @rget br
            @rget hatr
            @rget vcr
            @test isapprox(b, br, atol=1e-5, rtol=1e-5)
            @test isapprox(hat, hatr, atol=1e-5, rtol=1e-5)
            @test isapprox(vc, vcr, atol=1e-5, rtol=1e-4)
        end
    end
end
