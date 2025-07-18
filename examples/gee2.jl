using StableRNGs
using EstimatingEquationsRegression
using Statistics

rng = StableRNG(1)
n = 1000
m = 5
N = n*m
r = 0.5

g = repeat(1:n, inner=m)
Xm = randn(rng, N, 5)
beta_m = [1, 0, -1, 0, 0]
Ey = exp.(Xm * beta_m)
Xv = randn(rng, N, 4)
Xv[:, 1] .= 1
beta_v = [-0.4, 0.5, -0.5, 0.0]
Vy = Ey .* exp.(Xv * beta_v)
Xr = randn(rng, N, 2)
Xr[:, 1] .= 1

beta_r = [EstimatingEquationsRegression.linkfun(SigmoidLink(-1, 1), r), 0, 0]
beta = vcat(beta_m, beta_v, beta_r)

function make_rcov(x1, x2)
    return [1., x1[2]+x2[2], abs(x1[2]-x2[2])]
end

nrep = 100

function main()
    B = zeros(nrep, length(beta))
    Z = zeros(nrep, length(beta))
    for j in 1:nrep
        println(j)
        u = repeat(randn(n), inner=m)
        y = Ey + sqrt.(Vy) .* (sqrt(r)*u + sqrt(1-r)*randn(N))
        mm = fit(GeneralizedEstimatingEquations2Model, Xm, Xv, Xr, y, g, make_rcov; link_mean=LogLink(),
                 varfunc_mean=IdentityVar(), verbosity=0)
        B[j, :] = coef(mm)
        Z[j, :] = (coef(mm) - beta) ./ stderror(mm)
    end
    return B, Z
end

B, Z = main()

# use std(Z; dims=1) to assess the results

