using StableRNGs
using EstimatingEquationsRegression
using Statistics
using LinearAlgebra

rng = StableRNG(1)

# Number of groups
n = 1000

# Sample size per group
m = 5

# Total number of observations
N = n*m

# Intraclass correlation for each outcome
rr = [0.6, 0, 0.4]

# Coefficients for mean and scale models.  There are five covariates
# in the mean model and four covariates in the scale model.
beta_m = hcat([1, 0, -1, 0, 0], [0, 1, 1, 0, -1], [0, 0, 1, -1, 0])
beta_v = hcat([-0.4, 0.5, -0.5, 0.0], [-0.2, 0.2, 0.5, 0], [-0.3, 0, 0, 0.5])

# There are three parameters in the correlation model
function make_rcov(x1, x2)
    return [1., x1[2]+x2[2], abs(x1[2]-x2[2])]
end

# Group indicators
g = repeat(1:n, inner=m)

# Design matrix for the mean structure
Xm = randn(rng, N, 5)

# Design matrix for the variance structure
Xv = randn(rng, N, 4)
Xv[:, 1] .= 1

# Design matrix for the correlation structure
Xr = randn(rng, N, 2)

# True means and variances
Ey = Xm * beta_m
Vy = exp.(Xv * beta_v)

# The correlation between errors for different outcomes is controlled by the parameter 'f'.
f = 0.5

# Storage for total errors
E = zeros(N, 3)

# Shared errors within groups that are also shared among outcomes
v = repeat(randn(n), inner=m)

# Unique errors that are shared across outcomes
ee = randn(N)

for j in 1:3
    # Shared errors within groups that are unique to each outcome
    u = repeat(randn(n), inner=m)

    # Overall shared errors, with unit variance
    u = sqrt(f)*v + sqrt(1-f)*u

    # Unique errors with unit variance
    e = sqrt(f)*ee + sqrt(1-f)*randn(N)

    # Overall errors
    E[:, j] = sqrt.(Vy[:, j]) .* (sqrt(rr[j])*u + sqrt(1-rr[j])*e)
end

# The observed data
y = Ey + E

# Fit a model for each outcome
mm = Any[nothing, nothing, nothing]
for j in 1:3
    mm[j] = fit(GeneralizedEstimatingEquations2Model, Xm, Xv, Xr, y[:, j], g, make_rcov;
              verbosity=0, cor_pair_factor=0.0)
end

vc, B, M = vcov(mm...)

# The cross-covariance between mean structure parameters for the first and second outcomes.
A = vc[1:5, 13:17]

# Convert A into a cross-correlation matrix
# If f is large, we expect strong correlations in R and in A
B = Diagonal(sqrt.(diag(vc[1:5, 1:5])))
C = Diagonal(sqrt.(diag(vc[13:17, 13:17])))
R = B \ A / C

# The correlation of rr should closely match the correlation of E.
rs = hcat([residuals(m) for m in mm]...)
