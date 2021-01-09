using Test, DataFrames, CSV, StatsBase, StatsModels, LinearAlgebra

using GEE
using GLM: Normal, IdentityLink, LogLink, LogitLink, glm
using Distributions: Poisson, Normal, Binomial, cdf, quantile
using Random

function genx(n, p)
    X = randn(n, p)
    if p > 2
        r = 0.8
        X[:, 2] = r*X[:, 1] + sqrt(1-r^2)*X[:, 2]
    end
    X
end

function gennormal(n=2000, p=5, m=200)
    Random.seed!(123)
    X = genx(n, p)
    lp = X[:, 2] - X[:, 3]
    y = lp
    u = 2*randn(n) # Independent unexplained variation
    g = rand(1:m, n)
    sort!(g)
    e = randn(m)
    f = [e[i] for i in g] # Exchangeable group effects
    y .= y + f + u
    return y, X, g
end

function genpoisson(n=2000, p=5, m=200)
    Random.seed!(123)
    X = genx(n, p)
    lp = X[:, 2] - X[:, 3]
    ey = exp.(lp) # Marginal mean
    g = rand(1:m, n)
    sort!(g)
    h = randn(m)/sqrt(2)
    f = [h[i] for i in g]
    e = randn(n)/sqrt(2) + f
    u = cdf(Normal(), e) # Gaussian copula
    y = [Float64(floor(quantile(Poisson(ey[i]), u[i]))) for i in 1:n]
    return y, X, g
end

function genbinomial(n=2000, p=5, m=200)
    Random.seed!(123)
    X = genx(n, p)
    lp = X[:, 2] - X[:, 3]
    ey = 1 ./ (1 .+ exp.(-lp)) # Marginal mean
    g = rand(1:m, n)
    sort!(g)
    h = randn(m)/sqrt(2)
    f = [h[i] for i in g]
    e = randn(n)/sqrt(2) + f
    u = cdf(Normal(), e) # Gaussian copula
    y = [u[i] < ey[i] ? 1.0 : 0.0 for i in 1:n]
    return y, X, g
end

function save(y, X, g)
    da = DataFrame(X)
    da[:, :y] = y
    da[:, :g] = g
    CSV.write("tmp.csv", da)
end

@testset "AR1 covsolve" begin

    Random.seed!(123)

    makeAR = (r, d) -> [r^abs(i-j) for i in 1:d, j in 1:d]

    for d in [1, 2, 4]
       for q in [1, 3]

            c = AR1Cor(0.4)
            v = q == 1 ? randn(d) : randn(d, q)
            sd = rand(d)
	    sm = Diagonal(sd)

            mat = makeAR(0.4, d)
            vi = (sm \ (mat \ (sm \ v)))
            vi2 = GEE.covsolve(c, sd, v)
            @test isapprox(vi, vi2)

        end
    end
end


@testset "Exchangeable covsolve" begin

    Random.seed!(123)

    makeEx = (r, d) -> [i==j ? 1 : r for i in 1:d, j in 1:d]

    for d in [1, 2, 4]
       for q in [1, 3]

            c = ExchangeableCor(0.4)
            v = q == 1 ? randn(d) : randn(d, q)
            sd = rand(d)
	    sm = Diagonal(sd)

            mat = makeEx(0.4, d)
            vi = (sm \ (mat \ (sm \ v)))
            vi2 = GEE.covsolve(c, sd, v)
            @test isapprox(vi, vi2)

        end
    end
end


@testset "linear/normal independence model" begin
    y, X, g = gennormal()
    m = fit(GeneralizedEstimatingEquationsModel, X, y, g, Normal(), IndependenceCor(), IdentityLink())
    @test isapprox(coef(m), [-0.054514, 0.962038, -1.011813, 0.032347, 0.052044], atol=1e-4)
    @test isapprox(stderr(m), [0.086152, 0.087786, 0.047687, 0.044884,0.051413], atol=1e-4)
    @test isapprox(dispersion(m), 5.03324, atol=1e-4)

    # Check fitting using formula/dataframe
    df = DataFrame(X)
    df[!, :g] = g
    df[!, :y] = y
    f = @formula(y ~ 0 + x1 + x2 + x3 + x4 + x5)
    m1 = fit(GeneralizedEstimatingEquationsModel, f, df, g, Normal(), IndependenceCor(), IdentityLink())
    m2 = gee(f, df, g, Normal(), IndependenceCor(), IdentityLink())
    @test isapprox(coef(m), coef(m1), atol=1e-8)
    @test isapprox(stderr(m), stderr(m1), atol=1e-8)
    @test isapprox(coef(m), coef(m2), atol=1e-8)
    @test isapprox(stderr(m), stderr(m2), atol=1e-8)

    # With independence correlation, GLM and GEE have the same parameter estimates
    m0 = glm(X, y, Normal(), IdentityLink())
    @test isapprox(coef(m), coef(m0), atol=1e-5)

    # score test where the null hypothesis is false
    m = fit(GeneralizedEstimatingEquationsModel, X, y, g, Normal(), IndependenceCor(), IdentityLink(); dofit=false)
    subm = fit(GeneralizedEstimatingEquationsModel, X[:, 1:2], y, g, Normal(), IndependenceCor(), IdentityLink())
    cst = scoretest(m, subm)
    @test isapprox(cst.stat, 113.08148, atol=1e-4)
    @test isapprox(cst.dof, 3)
    @test isapprox(cst.pvalue, 0, atol=1e-5)

    # score test where the null hypothesis is true
    m = fit(GeneralizedEstimatingEquationsModel, X, y, g, Normal(), IndependenceCor(), IdentityLink(); dofit=false)
    subm = fit(GeneralizedEstimatingEquationsModel, X[:, [2,3]], y, g, Normal(), IndependenceCor(), IdentityLink())
    cst = scoretest(m, subm)
    @test isapprox(cst.stat, 1.63509, atol=1e-4)
    @test isapprox(cst.dof, 3)
    @test isapprox(cst.pvalue, 0.65146, atol=1e-4)
end

@testset "linear/normal exchangeable model" begin
    y, X, g = gennormal()
    m = fit(GeneralizedEstimatingEquationsModel, X, y, g, Normal(), ExchangeableCor(), IdentityLink())
    @test isapprox(coef(m), [-0.065818, 1.012311, -1.014096, 0.043734, 0.067308], atol=1e-4)
    @test isapprox(stderr(m), [0.080425, 0.078654, 0.045197, 0.041166, 0.047853], atol=1e-4)
    @test isapprox(dispersion(m), 5.03536, atol=1e-4)
    @test isapprox(corparams(m), 0.203567, atol=1e-3)
end

@testset "log/Poisson independence model" begin
    y, X, g = genpoisson()
    m = fit(GeneralizedEstimatingEquationsModel, X, y, g, Poisson(), IndependenceCor(), LogLink())
    @test isapprox(coef(m), [-0.026783, 1.021046, -1.000394, 0.007432, 0.013475], atol=1e-4)
    @test isapprox(stderr(m), [0.022133, 0.021290, 0.010624, 0.013264, 0.014284], atol=1e-4)
    @test isapprox(dispersion(m), 1)

    # With independence correlation, GLM and GEE have the same parameter estimates
    m0 = glm(X, y, Poisson(), LogLink())
    @test isapprox(coef(m), coef(m0), atol=1e-5)

    # score test where the null hypothesis is false
    m = fit(GeneralizedEstimatingEquationsModel, X, y, g, Poisson(), IndependenceCor(), LogLink(); dofit=false)
    subm = fit(GeneralizedEstimatingEquationsModel, X[:, 1:2], y, g, Poisson(), IndependenceCor(), LogLink())
    cst = scoretest(m, subm)
    @test isapprox(cst.stat, 82.59548, atol=1e-4)
    @test isapprox(cst.dof, 3)
    @test isapprox(cst.pvalue, 0, atol=1e-5)

    # score test where the null hypothesis is true
    m = fit(GeneralizedEstimatingEquationsModel, X, y, g, Poisson(), IndependenceCor(), LogLink(); dofit=false)
    subm = fit(GeneralizedEstimatingEquationsModel, X[:, [2,3]], y, g, Poisson(), IndependenceCor(), LogLink())
    cst = scoretest(m, subm)
    @test isapprox(cst.stat, 2.53648, atol=1e-4)
    @test isapprox(cst.dof, 3)
    @test isapprox(cst.pvalue, 0.468735, atol=1e-5)
end

@testset "log/Poisson exchangeable model" begin
    y, X, g = genpoisson()
    m = fit(GeneralizedEstimatingEquationsModel, X, y, g, Poisson(), ExchangeableCor(), LogLink())
    @test isapprox(coef(m), [-0.001435, 0.985413, -1.007042, -0.004655, -0.000079], atol=1e-4)
    @test isapprox(stderr(m), [0.017777, 0.016603, 0.007687, 0.009982, 0.011137], atol=1e-4)
    @test isapprox(dispersion(m), 1)
    @test isapprox(corparams(m), 0.39790, atol=1e-3)
end

@testset "logit/Binomial independence model" begin
    y, X, g = genbinomial()
    m = fit(GeneralizedEstimatingEquationsModel, X, y, g, Binomial(), IndependenceCor(), LogitLink())
    @test isapprox(coef(m), [0.163437, 0.890216, -0.985922, -0.000825, -0.062478], atol=1e-4)
    @test isapprox(stderr(m), [0.093111, 0.094662, 0.063312, 0.053646, 0.047485],  atol=1e-4)
    @test isapprox(dispersion(m), 1)

    # With independence correlation, GLM and GEE have the same parameter estimates
    m0 = glm(X, y, Binomial(), LogitLink())
    @test isapprox(coef(m), coef(m0), atol=1e-5)
end

@testset "logit/Binomial exchangeable model" begin
    y, X, g = genbinomial()
    m = fit(GeneralizedEstimatingEquationsModel, X, y, g, Binomial(), ExchangeableCor(), LogitLink())
    @test isapprox(coef(m), [0.056464, 0.991988, -0.991286, 0.032371, -0.035211], atol=1e-4)
    @test isapprox(stderr(m), [0.081861, 0.082812, 0.060770, 0.050310, 0.043223], atol=1e-4)
    @test isapprox(dispersion(m), 1)
    @test isapprox(corparams(m), 0.255286, atol=1e-3)
end
