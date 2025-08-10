using StableRNGs
using EstimatingEquationsRegression
using Statistics
using LinearAlgebra

rng = StableRNG(1)
n = 1000
m = 5
N = n*m

# Residual correlation for each outcome
rr = [0.6, 0, 0.4]

# Coefficients for mean and scale models
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

# Generate additive errors with both row and column correlations.  The strength of the
# correlations is controlled by the parameter 'f'.
f = 0.5
E = zeros(N, 3)
v = repeat(randn(n), inner=m)
ee = randn(N)
for j in 1:3
    u = repeat(randn(n), inner=m)
    u = sqrt(f)*v + sqrt(1-f)*u
    e = sqrt(f)*ee + sqrt(1-f)*randn(N)
    E[:, j] = sqrt.(Vy[:, j]) .* (sqrt(rr[j])*u + sqrt(1-rr[j])*e)
end

y = Ey + E

mm1 = fit(GeneralizedEstimatingEquations2Model, Xm, Xv, Xr, y[:, 1], g, make_rcov;
          verbosity=0, cor_pair_factor=0.0)

mm2 = fit(GeneralizedEstimatingEquations2Model, Xm, Xv, Xr, y[:, 2], g, make_rcov;
          verbosity=0, cor_pair_factor=0.0)

mm3 = fit(GeneralizedEstimatingEquations2Model, Xm, Xv, Xr, y[:, 3], g, make_rcov;
          verbosity=0, cor_pair_factor=0.0)

vc = vcov(mm1, mm2, mm3)

# Print the correlations between parameter estimates for two responses.
m = vc[1]
Diagonal(sqrt.(diag(m[1:12, 1:12]))) \ m[1:12, 13:24] / Diagonal(sqrt.(diag(m[13:24, 13:24]))) |> diag |> display
