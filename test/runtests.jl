using Test, DataFrames, CSV, StatsBase, StatsModels, LinearAlgebra, Distributions

using GEE
using GLM: IdentityLink, LogLink, LogitLink, glm
using Random

function data1()
	X = [3. 3  2; 
	     1  2  1; 
	     2  3  3; 
	     1  5  4; 
	     4  0  0; 
	     3  2  2;
	     4  2  0;
	     6  1  4;
	     0  0  0;
	     2  0  4;
	     8  3  3;
	     4  4  0;
	     1  2  1;
	     2  1  5]
	y = [3., 1, 4, 4, 1, 3, 1, 2, 1, 1, 2, 4, 2, 2]
	z = [0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0]
	g = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3]
    return y, z, X, g
end

function save()
	y, z, X, g = data1()
    da = DataFrame(X, :auto)
    da[:, :y] = y
    da[:, :z] = z
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
            vi2 = GEE.covsolve(c, sd, zeros(0), v)
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
            vi2 = GEE.covsolve(c, sd, zeros(0), v)
            @test isapprox(vi, vi2)

        end
    end
end


@testset "linear/normal independence model" begin

    y, _, X, g = data1()
    m = fit(GeneralizedEstimatingEquationsModel, X, y, g, Normal(), IndependenceCor(), IdentityLink())
	se = sqrt.(diag(vcov(m)))
    @test isapprox(coef(m), [0.0397, 0.7499, 0.2147], atol=1e-4)
    @test isapprox(se, [0.089, 0.089, 0.021], atol=1e-3)
    @test isapprox(dispersion(m), 0.673, atol=1e-4)

    # Check fitting using formula/dataframe
    df = DataFrame(X, :auto)
    df[!, :g] = g
    df[!, :y] = y
    f = @formula(y ~ 0 + x1 + x2 + x3)
    m1 = fit(GeneralizedEstimatingEquationsModel, f, df, g, Normal(), IndependenceCor(), IdentityLink())
    m2 = gee(f, df, g, Normal(), IndependenceCor(), IdentityLink())
	se1 = sqrt.(diag(vcov(m1)))
	se2 = sqrt.(diag(vcov(m2)))
    @test isapprox(coef(m), coef(m1), atol=1e-8)
    @test isapprox(se, se1, atol=1e-8)
    @test isapprox(coef(m), coef(m2), atol=1e-8)
    @test isapprox(se, se2, atol=1e-8)

    # With independence correlation, GLM and GEE have the same parameter estimates
    m0 = glm(X, y, Normal(), IdentityLink())
    @test isapprox(coef(m), coef(m0), atol=1e-5)

    # Test Mancl-DeRouen bias-corrected covariance
    md = [ 0.109898 -0.107598 -0.031721;
          -0.107598  0.128045  0.043794;
          -0.031721  0.043794  0.016414]
    @test isapprox(vcov(m1, cov_type="md"), md, atol=1e-5)

    # Score test
    m = fit(GeneralizedEstimatingEquationsModel, X, y, g, Normal(), IndependenceCor(), IdentityLink(); dofit=false)
    subm = fit(GeneralizedEstimatingEquationsModel, X[:, [1,2]], y, g, Normal(), IndependenceCor(), IdentityLink())
    cst = scoretest(m, subm)
    @test isapprox(cst.stat, 2.01858, atol=1e-4)
    @test isapprox(cst.dof, 1)
    @test isapprox(cst.pvalue, 0.155385, atol=1e-5)

    # Score test
    m = fit(GeneralizedEstimatingEquationsModel, X, y, g, Normal(), IndependenceCor(), IdentityLink(); dofit=false)
    subm = fit(GeneralizedEstimatingEquationsModel, X[:, [2]], y, g, Normal(), IndependenceCor(), IdentityLink())
    cst = scoretest(m, subm)
    @test isapprox(cst.stat, 2.80908, atol=1e-4)
    @test isapprox(cst.dof, 2)
    @test isapprox(cst.pvalue, 0.24548, atol=1e-4)
end

@testset "logit/Binomial independence model" begin
    _, y, X, g = data1()
    m = fit(GeneralizedEstimatingEquationsModel, X, y, g, Binomial(), IndependenceCor(), LogitLink())
	se = sqrt.(diag(vcov(m)))
    @test isapprox(coef(m), [0.5440, -0.2293, -0.8340], atol=1e-4)
    @test isapprox(se, [0.121, 0.144, 0.178],  atol=1e-3)
    @test isapprox(dispersion(m), 1)

    # With independence correlation, GLM and GEE have the same parameter estimates
    m0 = glm(X, y, Binomial(), LogitLink())
    @test isapprox(coef(m), coef(m0), atol=1e-5)

    # Score test
    m = fit(GeneralizedEstimatingEquationsModel, X, y, g, Binomial(), IndependenceCor(), LogitLink(); dofit=false)
    subm = fit(GeneralizedEstimatingEquationsModel, X[:, [1,2]], y, g, Binomial(), IndependenceCor(), LogitLink())
    cst = scoretest(m, subm)
    @test isapprox(cst.stat, 2.53019, atol=1e-4)
    @test isapprox(cst.dof, 1)
    @test isapprox(cst.pvalue, 0.11169, atol=1e-5)

    # Score test
    m = fit(GeneralizedEstimatingEquationsModel, X, y, g, Binomial(), IndependenceCor(), LogitLink(); dofit=false)
    subm = fit(GeneralizedEstimatingEquationsModel, X[:, [2]], y, g, Binomial(), IndependenceCor(), LogitLink())
    cst = scoretest(m, subm)
    @test isapprox(cst.stat, 2.77068, atol=1e-4)
    @test isapprox(cst.dof, 2)
    @test isapprox(cst.pvalue, 0.25024, atol=1e-4)
end

@testset "log/Poisson independence model" begin
    y, _, X, g = data1()
    m = fit(GeneralizedEstimatingEquationsModel, X, y, g, Poisson(), IndependenceCor(), LogLink())
    se = sqrt.(diag(vcov(m)))
    @test isapprox(coef(m), [0.0051, 0.2777, 0.0580], atol=1e-4)
    @test isapprox(se, [0.020, 0.033, 0.014], atol=1e-3)
    @test isapprox(dispersion(m), 1)

    # With independence correlation, GLM and GEE have the same parameter estimates
    m0 = glm(X, y, Poisson(), LogLink())
    @test isapprox(coef(m), coef(m0), atol=1e-5)

    # Score test
    m = fit(GeneralizedEstimatingEquationsModel, X, y, g, Poisson(), IndependenceCor(), LogLink(); dofit=false)
    subm = fit(GeneralizedEstimatingEquationsModel, X[:, [1,2]], y, g, Poisson(), IndependenceCor(), LogLink())
    cst = scoretest(m, subm)
    @test isapprox(cst.stat, 2.600191, atol=1e-4)
    @test isapprox(cst.dof, 1)
    @test isapprox(cst.pvalue, 0.106851, atol=1e-5)

    # Score test
    m = fit(GeneralizedEstimatingEquationsModel, X, y, g, Poisson(), IndependenceCor(), LogLink(); dofit=false)
    subm = fit(GeneralizedEstimatingEquationsModel, X[:, [2]], y, g, Poisson(), IndependenceCor(), LogLink())
    cst = scoretest(m, subm)
    @test isapprox(cst.stat, 2.94147, atol=1e-4)
    @test isapprox(cst.dof, 2)
    @test isapprox(cst.pvalue, 0.229757, atol=1e-5)
end

@testset "log/Gamma independence model" begin
    y, _, X, g = data1()
    m = fit(GeneralizedEstimatingEquationsModel, X, y, g, Gamma(), IndependenceCor(), LogLink())
	se = sqrt.(diag(vcov(m)))
    @test isapprox(coef(m), [-0.0075, 0.2875, 0.0725], atol=1e-4)
    @test isapprox(se, [0.019, 0.034, 0.006], atol=1e-3)
    @test isapprox(dispersion(m), 0.104, atol=1e-3)

    # With independence correlation, GLM and GEE have the same parameter estimates
    m0 = glm(X, y, Gamma(), LogLink())
    @test isapprox(coef(m), coef(m0), atol=1e-5)

    # Acore test
    m = fit(GeneralizedEstimatingEquationsModel, X, y, g, Gamma(), IndependenceCor(), LogLink(); dofit=false)
    subm = fit(GeneralizedEstimatingEquationsModel, X[:, [1,2]], y, g, Gamma(), IndependenceCor(), LogLink())
    cst = scoretest(m, subm)
    @test isapprox(cst.stat, 2.471939, atol=1e-4)
    @test isapprox(cst.dof, 1)
    @test isapprox(cst.pvalue, 0.115895, atol=1e-5)

    # Acore test
    m = fit(GeneralizedEstimatingEquationsModel, X, y, g, Gamma(), IndependenceCor(), LogLink(); dofit=false)
    subm = fit(GeneralizedEstimatingEquationsModel, X[:, [2]], y, g, Gamma(), IndependenceCor(), LogLink())
    cst = scoretest(m, subm)
    @test isapprox(cst.stat, 2.99726, atol=1e-4)
    @test isapprox(cst.dof, 2)
    @test isapprox(cst.pvalue, 0.223437, atol=1e-5)
end

@testset "weights" begin

	Random.seed!(432)
    X = randn(6, 2)
    X[:, 1] .= 1
    y = X[:, 1] + randn(6)
    y[6] = y[5]
    X[6, :] = X[5, :]

    g = [1., 1, 1, 2, 2, 2]
    m1 = fit(GeneralizedEstimatingEquationsModel, X, y, g, Normal())
	se1 = sqrt.(diag(vcov(m1)))

    wts = [1., 1, 1, 1, 1, 1]
    m2 = fit(GeneralizedEstimatingEquationsModel, X, y, g, Normal(), wts=wts)
	se2 = sqrt.(diag(vcov(m2)))

    wts = [1., 1, 1, 1, 2]
    X1 = X[1:5, :]
    y1 = y[1:5]
    g1 = g[1:5]
    m3 = fit(GeneralizedEstimatingEquationsModel, X1, y1, g1, Normal(), wts=wts)
	se3 = sqrt.(diag(vcov(m3)))

    @test isapprox(coef(m1), coef(m2))
    @test isapprox(coef(m1), coef(m3))
    @test isapprox(se1, se2)
    @test isapprox(se1, se3)
    @test isapprox(dispersion(m1), dispersion(m2))
    #@test isapprox(dispersion(m1), dispersion(m3))

end

#@testset "linear/normal exchangeable model" begin
#    y, X, g = gennormal()
#    m = fit(GeneralizedEstimatingEquationsModel, X, y, g, Normal(), ExchangeableCor(), IdentityLink())
#    @test isapprox(coef(m), [-0.065818, 1.012311, -1.014096, 0.043734, 0.067308], atol=1e-4)
#    @test isapprox(stderr(m), [0.080425, 0.078654, 0.045197, 0.041166, 0.047853], atol=1e-4)
#    @test isapprox(dispersion(m), 5.03536, atol=1e-4)
#    @test isapprox(corparams(m), 0.203567, atol=1e-3)
#end

#@testset "log/Poisson exchangeable model" begin
#    y, X, g = genpoisson()
#    m = fit(GeneralizedEstimatingEquationsModel, X, y, g, Poisson(), ExchangeableCor(), LogLink())
#    @test isapprox(coef(m), [-0.001435, 0.985413, -1.007042, -0.004655, -0.000079], atol=1e-4)
#    @test isapprox(stderr(m), [0.017777, 0.016603, 0.007687, 0.009982, 0.011137], atol=1e-4)
#    @test isapprox(dispersion(m), 1)
#    @test isapprox(corparams(m), 0.39790, atol=1e-3)
#end

#@testset "logit/Binomial exchangeable model" begin
#    y, X, g = genbinomial()
#    m = fit(GeneralizedEstimatingEquationsModel, X, y, g, Binomial(), ExchangeableCor(), LogitLink())
#    @test isapprox(coef(m), [0.056464, 0.991988, -0.991286, 0.032371, -0.035211], atol=1e-4)
#    @test isapprox(stderr(m), [0.081861, 0.082812, 0.060770, 0.050310, 0.043223], atol=1e-4)
#    @test isapprox(dispersion(m), 1)
#    @test isapprox(corparams(m), 0.255286, atol=1e-3)
#end
