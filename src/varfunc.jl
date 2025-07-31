abstract type Varfunc end

# Make varfuncs broadcast like a scalar
Base.Broadcast.broadcastable(vf::Varfunc) = Ref(vf)

struct ConstantVar <: Varfunc end

struct IdentityVar <: Varfunc end

struct BinomialVar <: Varfunc end

# Use the default variance function for the family
struct DefaultVar <: Varfunc end

struct PowerVar <: Varfunc
    p::Float64
end

geevar(d::Normal, v::DefaultVar, mu::T, awt::T) where {T<:Real} = 1 / awt
geevar(d::Poisson, v::DefaultVar, mu::T, awt::T) where {T<:Real} = mu / awt
geevar(d::Binomial, v::DefaultVar, mu::T, awt::T) where {T<:Real} = mu * (1 - mu) / awt
geevar(d::Gamma, v::DefaultVar, mu::T, awt::T) where {T<:Real} = mu^2 / awt

geevar(::Distribution, v::ConstantVar, mu::T, awt::T) where {T<:Real} = 1 / awt
geevar(::Distribution, v::IdentityVar, mu::T, awt::T) where {T<:Real} = mu / awt
geevar(::Distribution, v::PowerVar, mu::T, awt::T) where {T<:Real} = mu^v.p / awt
geevar(::Distribution, v::BinomialVar, mu::T, awt::T) where {T<:Real} = mu * (1 - mu) / awt

geevarderiv(d::Normal, v::DefaultVar, mu::T, awt::T) where {T<:Real} = zero(T)
geevarderiv(d::Poisson, v::DefaultVar, mu::T, awt::T) where {T<:Real} = one(T) / awt
geevarderiv(d::Binomial, v::DefaultVar, mu::T, awt::T) where {T<:Real} = (1 - 2*mu) / awt
geevarderiv(d::Gamma, v::DefaultVar, mu::T, awt::T) where {T<:Real} = 2*mu / awt

geevarderiv(::Distribution, v::ConstantVar, mu::T, awt::T) where {T<:Real} = zero(T)
geevarderiv(::Distribution, v::IdentityVar, mu::T, awt::T) where {T<:Real} = one(T) / awt
geevarderiv(::Distribution, v::PowerVar, mu::T, awt::T) where {T<:Real} = v.p * mu^(v.p - 1) / awt
geevarderiv(::Distribution, v::BinomialVar, mu::T, awt::T) where {T<:Real} = (1 - 2*mu) / awt

geevar(v::Varfunc, mu::T, awt::T) where {T<:Real} = geevar(NoDistribution(), v, mu, awt)
geevarderiv(v::Varfunc, mu::T, awt::T) where {T<:Real} = geevarderiv(NoDistribution(), v, mu, awt)
