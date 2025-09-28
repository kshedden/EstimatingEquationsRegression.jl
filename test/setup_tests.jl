@testsetup module Gendat

    export data1, gendat

    using Distributions
    using DataFrames

    function data1()
        X = [
            3.0 3 2
            1 2 1
            2 3 3
            1 5 4
            4 0 0
            3 2 2
            4 2 0
            6 1 4
            0 0 0
            2 0 4
            8 3 3
            4 4 0
            1 2 1
            2 1 5
        ]
        y = [3.0, 1, 4, 4, 1, 3, 1, 2, 1, 1, 2, 4, 2, 2]
        z = [0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0]
        g = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3]
        df = DataFrame(y = y, z = z, x1 = X[:, 1], x2 = X[:, 2], x3 = X[:, 3], g = g)

        return y, z, X, g, df
    end

    function gendat(ngroup, gsize, p, r, rng, dist)

        # Sample size per group
        n = 1 .+ rand(rng, Poisson(gsize), ngroup)
        N = sum(n)

        # Group labels
        id = vcat([fill(i, n[i]) for i in eachindex(n)]...)

        # Random intercepts
        ri = randn(rng, ngroup)
        ri = ri[id]

        X = randn(rng, N, p)
        for j in 2:p
            X[:, j] = r*X[:, j-1] + sqrt(1-r^2)*X[:, j]
        end

        lp = X[:, 1] - 2*X[:, 2]

        if dist == :Gaussian
            ey = lp
            y = ey + ri + randn(rng, N)
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
end

