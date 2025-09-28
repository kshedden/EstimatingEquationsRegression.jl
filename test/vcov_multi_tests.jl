@testitem "Test some properties of the joint multivariate vcov" begin

    using StableRNGs
    using LinearAlgebra

    rng = StableRNG(1)
    n = 500
    m = 5
    N = n*m

    # Residual correlation for each outcome
    rr = [0.6, 0, 0.4]

    beta_m = hcat([1, 0, -1, 0, 0], [0, 1, 1, 0, -1], [0, 0, 1, -1, 0])
    beta_v = hcat([-0.4, 0.5, -0.5, 0.0], [-0.2, 0.2, 0.5, 0], [-0.3, 0, 0, 0.5])

    function make_rcov(x1, x2)
        return [1., x1[2]+x2[2], abs(x1[2]-x2[2])]
    end

    g = repeat(1:n, inner=m)
    Xm = randn(rng, N, 5)
    Xv = randn(rng, N, 4)
    Xv[:, 1] .= 1
    Xr = randn(rng, N, 2)

    Ey = Xm * beta_m
    Vy = exp.(Xv * beta_v)

    y = zeros(N, 3)
    for j in 1:3
        u = repeat(randn(n), inner=m)
        y[:, j] = Ey[:, j] + sqrt.(Vy[:, j]) .* (sqrt(rr[j])*u + sqrt(1-rr[j])*randn(N))
    end

    mm1 = fit(GeneralizedEstimatingEquations2Model, Xm, Xv, Xr, y[:, 1], g, make_rcov;
              verbosity=0, cor_pair_factor=0.0)

    mm2 = fit(GeneralizedEstimatingEquations2Model, Xm, Xv, Xr, y[:, 2], g, make_rcov;
              verbosity=0, cor_pair_factor=0.0)

    mm3 = fit(GeneralizedEstimatingEquations2Model, Xm, Xv, Xr, y[:, 3], g, make_rcov;
              verbosity=0, cor_pair_factor=0.0)

    vc = vcov(mm1, mm2, mm3) |> first

    t = size(Xm, 2) + size(Xv, 2) + 3
    @test isapprox(vc[1:t, 1:t], vcov(mm1))
    @test isapprox(vc[t+1:2*t, t+1:2*t], vcov(mm2))
    @test isapprox(vc[2*t+1:3*t, 2*t+1:3*t], vcov(mm3))

    @test maximum(abs.(vc - vc')) < 1e-10
    a, _ = eigen(vc)
    @test all(a .> 0)
end
