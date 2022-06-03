@testset "QIF score Jacobian" begin

    rng = StableRNG(123)
    gs = 10
    ngrp = 100
    n = ngrp * gs
    xm = randn(rng, n, 3)
    grp = kron(1:ngrp, ones(gs))
    grp = Int.(grp)
    lp = (xm[:, 1] - xm[:, 3]) ./ 2
    basis = [QIFIdentityBasis(), QIFHollowBasis()]
    q = length(basis)
    y = rand.(rng, Poisson.(exp.(lp)))
    m = qif(
        xm,
        y,
        grp;
        basis = basis,
        link = GLM.LogLink(),
        varfunc = IdentityVar(),
        dofit = false,
    )

    score = function (beta)
        m.beta .= beta
        GEE.iterprep!(m, beta)
        scr = zeros(q * length(beta))
        GEE.score!(m, scr)
        return scr
    end

    jac = function (beta)
        m.beta = beta
        GEE.iterprep!(m, beta)
        scd = zeros(q * length(beta), length(beta))
        GEE.scorederiv!(m, scd)
        return scd
    end

    beta = Float64[0.5, 0, -0.5]
    scd = jac(beta)
    scd1 = jacobian(central_fdm(10, 1), score, beta)[1]
    @test isapprox(scd, scd1, atol = 0.1, rtol = 0.2)
end

@testset "QIF fun/grad" begin

    rng = StableRNG(123)
    gs = 10
    ngrp = 100
    n = ngrp * gs
    xm = randn(rng, n, 3)
    grp = kron(1:ngrp, ones(gs))
    grp = Int.(grp)
    lp = (xm[:, 1] - xm[:, 3]) ./ 2
    basis = [QIFIdentityBasis(), QIFHollowBasis()]
    q = length(basis)
    y = rand.(rng, Poisson.(exp.(lp)))
    m = qif(
        xm,
        y,
        grp;
        basis = basis,
        link = GLM.LogLink(),
        varfunc = IdentityVar(),
        dofit = false,
    )

    for i = 1:10
        scov = randn(rng, 3 * q, 3 * q)
        scov = scov' * scov
        fun, grad! = GEE.get_fungrad(m, scov)
        beta = randn(rng, 3)
        gra = zeros(3)
        grad!(gra, beta)
        grn = grad(central_fdm(10, 1), fun, beta)[1]
        @test isapprox(gra, grn, atol = 1e-3, rtol = 1e-3)
    end
end

@testset "QIF independent observations Poisson" begin

    rng = StableRNG(123)
    gs = 10
    ngrp = 100
    n = ngrp * gs
    xm = randn(rng, n, 3)
    grp = kron(1:ngrp, ones(gs))
    grp = Int.(grp)
    lp = (xm[:, 1] - xm[:, 3]) ./ 2

    # When using only the identity basis, GLM and QIF should give the same result.
    b = [QIFIdentityBasis()]
    y = rand.(rng, Poisson.(exp.(lp)))
    m0 = glm(xm, y, Poisson())
    m1 = qif(
        xm,
        y,
        grp;
        basis = b,
        link = GLM.LogLink(),
        varfunc = IdentityVar(),
        start = coef(m0),
    )
    m2 = qif(xm, y, grp; basis = b, link = GLM.LogLink(), varfunc = IdentityVar())
    @test isapprox(coef(m0), coef(m1), atol = 1e-4, rtol = 1e-4)
    @test isapprox(coef(m0), coef(m2), atol = 1e-4, rtol = 1e-4)

    basis = [
        [QIFIdentityBasis()],
        [QIFIdentityBasis(), QIFHollowBasis()],
        [QIFIdentityBasis(), QIFHollowBasis(), QIFSubdiagonalBasis(1)],
    ]

    for b in basis
        y = rand.(Poisson.(exp.(lp)))
        m = qif(xm, y, grp; basis = b, link = GLM.LogLink(), varfunc = IdentityVar())
        @test isapprox(coef(m), [0.5, 0, -0.5], atol = 0.05, rtol = 0.1)
    end
end

@testset "QIF independent observations linear" begin

    rng = StableRNG(123)
    gs = 10
    ngrp = 100
    n = ngrp * gs
    xm = randn(rng, n, 3)
    grp = kron(1:ngrp, ones(gs))
    grp = Int.(grp)
    ey = xm[:, 1] - xm[:, 3]
    nrep = 5

    basis = [
        [QIFIdentityBasis()],
        [QIFIdentityBasis(), QIFHollowBasis()],
        [QIFIdentityBasis(), QIFHollowBasis(), QIFSubdiagonalBasis(1)],
    ]

    for f in [0.5, 1, 2, 3]
        for b in basis
            cf = zeros(3)
            for i = 1:nrep
                y = ey + f * randn(rng, n)
                m = qif(xm, y, grp; basis = b)
                cf .+= coef(m)
                if !m.converged
                    println("f=", f, " b=", b, " i=", i)
                end
                @test isapprox(diag(vcov(m)), f^2 * ones(3) / n, atol = 1e-2, rtol = 1e-2)
            end
            cf ./= nrep
            @test isapprox(cf, [1, 0, -1], atol = 1e-2, rtol = 2 * f / sqrt(n))
        end
    end
end

@testset "QIF dependent observations linear" begin

    rng = StableRNG(123)
    gs = 10
    ngrp = 100
    n = ngrp * gs
    xm = randn(rng, n, 3)
    grp = kron(1:ngrp, ones(gs))
    grp = Int.(grp)
    ey = xm[:, 1] - xm[:, 3]
    nrep = 5

    # The variance for GLS and OLS
    s = 0.5 * I(gs) .+ 0.5
    v_ols = zeros(3, 3)
    v_gls = zeros(3, 3)
    for i = 1:100
        x = randn(rng, gs, 3)
        v_gls .+= x' * (s \ x)
        xtx = x' * x
        v_ols .+= x' * s * x
    end
    v_ols /= 100
    v_gls /= 100
    v_gls = diag(inv(v_gls)) / ngrp
    v_ols = diag(v_ols) / (gs^2 * ngrp)

    basis = [
        [QIFIdentityBasis()],
        [QIFIdentityBasis(), QIFHollowBasis()],
        [QIFIdentityBasis(), QIFHollowBasis(), QIFSubdiagonalBasis(1)],
    ]

    for f in [0.5, 1, 2, 3]
        for (j, b) in enumerate(basis)
            cf = zeros(3)
            va = zeros(3)
            for i = 1:nrep
                y = ey + f * (randn(rng, ngrp)[grp] + randn(rng, n)) / sqrt(2)
                m = qif(xm, y, grp; basis = b)
                cf .+= coef(m)
                va .+= diag(vcov(m))
            end
            cf ./= nrep
            va ./= nrep
            @test isapprox(cf, [1, 0, -1]; atol = 1e-2, rtol = 2 * f / sqrt(n))

            # Including the hollow basis is needed to get GLS-like performance
            @test isapprox(
                va,
                j == 1 ? f^2 * v_ols : f^2 * v_gls,
                atol = 1e-3,
                rtol = 2 * f / sqrt(n),
            )
        end
    end
end
