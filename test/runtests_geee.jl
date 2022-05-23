using GLM

@testset "GEEE at tau=0.5 match to OLS/GEE" begin

    rng = StableRNG(123)
    n = 1000
    X = randn(rng, n, 3)
    X[:, 1] .= 1
    y = X[:, 2] + (1 .+ X[:, 3]) .* randn(rng, n)
    g = kron(1:200, ones(5))

    m1 = fit(GEEE, X, y, g, [0.5])
    m2 = lm(X, y)
    @test isapprox(coef(m1), coef(m2), atol = 1e-4, rtol = 1e-4)
    m2 = fit(GeneralizedEstimatingEquationsModel, X, y, g, Normal(), IndependenceCor())
    @test isapprox(coef(m1), coef(m2), atol = 1e-4, rtol = 1e-4)
    @test isapprox(vcov(m1), vcov(m2), atol = 1e-4, rtol = 1e-4)

    m1 = fit(GEEE, X, y, g, [0.5], ExchangeableCor())
    m2 = fit(GeneralizedEstimatingEquationsModel, X, y, g, Normal(), ExchangeableCor())
    @test isapprox(coef(m1), coef(m2), atol = 1e-4, rtol = 1e-4)
    @test isapprox(vcov(m1), vcov(m2), atol = 1e-4, rtol = 1e-4)
end

@testset "GEEE simulation" begin

    rng = StableRNG(123)
    nrep = 1000
    n = 1000
    betas = zeros(nrep, 9)
    X = randn(rng, n, 3)
    X[:, 1] .= 1

    for i = 1:nrep
        y = X[:, 2] + (1 .+ X[:, 3]) .* randn(rng, n)
        g = kron(1:200, ones(5))
        m = fit(GEEE, X, y, g, [0.2, 0.5, 0.8])
        betas[i, :] = vec(m.beta)
    end

    m = mean(betas, dims = 1)[:]
    t = [-0.66, 1, -0.36, 0, 1, 0, 0.66, 1, 0.36]
    @test isapprox(m, t, atol = 1e-2, rtol = 1e-3)
end
