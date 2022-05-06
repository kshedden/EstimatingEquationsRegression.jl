
@testset "GEEE simulation" begin

    rng = StableRNG(123)
    nrep = 1000
    betas = zeros(nrep, 9)
    X = randn(rng, 1000, 3)
    X[:, 1] .= 1

    for i = 1:nrep
        y = X[:, 2] + (1 .+ X[:, 3]) .* randn(rng, 1000)
        g = kron(1:200, [1, 1, 1, 1, 1])
        m = fit(GEEE, X, y, g, [0.2, 0.5, 0.8])
        betas[i, :] = vec(m.beta)
    end

    m = mean(betas, dims = 1)[:]
    t = [-0.66, 1, -0.36, 0, 1, 0, 0.66, 1, 0.36]
    @test isapprox(m, t, atol = 1e-2, rtol = 1e-3)
end
