@testitem "GEE2 no correlation with and without formula" begin

    using StableRNGs
    using StatsBase
    using DataFrames
    using StatsModels
    using LinearAlgebra

    rng = StableRNG(123)

    n = 400
    N = 1000
    g = sample(rng, 1:50, N; replace=true)
    sort!(g)
    Xm = randn(rng, N, 3)
    Xm[:, 1] .= 1
    beta_m = [1, 0, -1]
    Ey = Xm * beta_m
    Xv = randn(rng, N, 3)
    Xv[:, 1] .= 1
    beta_v = [1, 0.2, 0]
    Vy = exp.(Xv * beta_v)
    y = Ey + sqrt.(Vy) .* randn(rng, N)
    par = vcat(beta_m, beta_v)

    df = DataFrame(y=y, x1=Xm[:, 2], x2=Xm[:, 3])
    fml = @formula(y ~ x1 + x2)

    c_save, v_save = [], []

    for use_fml in [false, true]
        if use_fml
            mm = fit(GeneralizedEstimatingEquations2Model, df, Xv, nothing, [], g, nothing; fml_mean=fml)
        else
            mm = fit(GeneralizedEstimatingEquations2Model, Xm, Xv, nothing, y, g, nothing)
        end

        c = coef(mm)
        push!(c_save, c)
        @test length(c) == 6
        s = IOBuffer()
        show(s, mm)
        s = stderror(mm)
        z = (c - par) ./ s
        @test all(abs.(z) .< 2)
        v = vcov(mm)
        push!(v_save, v)
        @test all(size(v) .== (6, 6))
        @test all(stderror(mm) .== sqrt.(diag(v)))
    end

    @test isapprox(c_save[1], c_save[2], rtol=1e-5, atol=1e-5)
    @test isapprox(v_save[1], v_save[2], rtol=1e-5, atol=1e-5)
end

@testitem "GEE2 with correlation, with and without formula" begin

    using StableRNGs
    using StatsBase
    using DataFrames
    using StatsModels
    using LinearAlgebra

    rng = StableRNG(321)

    n = 400
    N = 1000
    ngrp = 50
    g = sample(rng, 1:ngrp, N; replace=true)
    sort!(g)
    Xm = randn(rng, N, 3)
    Xm[:, 1] .= 1
    beta_m = [1, 0, -1]
    Ey = Xm * beta_m
    Xv = randn(rng, N, 3)
    Xv[:, 1] .= 1
    beta_v = [1, 0.2, 0]
    Vy = exp.(Xv * beta_v)
    Xr = randn(rng, N, 1)
    beta_r = [0.5, 0]

    r = 0.5
    u = randn(rng, ngrp)
    e = randn(rng, N)
    y = Ey + sqrt.(Vy) .* (sqrt(r)*u[g] + sqrt(1-r)*e)

    br = [EstimatingEquationsRegression.linkfun(SigmoidLink(-1, 1), r), 0]
    par = vcat(beta_m, beta_v, br)

    make_rcov = function(x1, x2)
        return [1, abs(x1[1] - x2[1])]
    end

    df = DataFrame(y=y, x1=Xm[:, 2], x2=Xm[:, 3])
    fml = @formula(y ~ x1 + x2)

    c_save, v_save = [], []

    for use_fml in [false, true]
        if use_fml
            mm = fit(GeneralizedEstimatingEquations2Model, df, Xv, Xr, [], g, make_rcov; fml_mean=fml)
        else
            mm = fit(GeneralizedEstimatingEquations2Model, Xm, Xv, Xr, y, g, make_rcov)
        end

        c = coef(mm)
        push!(c_save, c)
        @test length(c) == 8
        s = IOBuffer()
        show(s, mm)
        s = stderror(mm)
        z = (c - par) ./ s
        @test all(abs.(z) .< 2)
        v = vcov(mm)
        push!(v_save, v)
        @test all(size(v) .== (8, 8))
        @test all(stderror(mm) .== sqrt.(diag(v)))
    end

    @test isapprox(c_save[1], c_save[2], rtol=1e-5, atol=1e-5)
    @test isapprox(v_save[1], v_save[2], rtol=1e-5, atol=1e-5)
end
