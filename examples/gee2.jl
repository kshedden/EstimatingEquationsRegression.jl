using StableRNGs
using EstimatingEquationsRegression
using Statistics

rng = StableRNG(1)

n = 1000
m = 5
N = n*m
g = repeat(1:n, inner=m)
Xm = randn(rng, N, 3)
beta_m = [1, 0, -1]
Ey = Xm * beta_m
Xv = randn(rng, N, 3)
Xv[:, 1] .= 1
beta_v = [-0.4, 0.5, -0.5]
Vy = exp.(Xv * beta_v)

r = 0.4
u = repeat(randn(n), inner=m)
y = Ey + sqrt.(Vy) .* (sqrt(r)*u + sqrt(1-r)*randn(N))

Xr = randn(rng, N, 2)
Xr[:, 1] .= 1

function make_rcov(x1, x2)
    return [1., x1[2]+x2[2], abs(x1[2]-x2[2])]
end

m = fit(GeneralizedEstimatingEquations2Model, Xm, Xv, Xr, y, g, make_rcov; verbosity=2)
