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

geevar(d::Normal, v::DefaultVar, mu::T) where {T<:Real} = 1
geevar(d::Poisson, v::DefaultVar, mu::T) where {T<:Real} = mu
geevar(d::Binomial, v::DefaultVar, mu::T) where {T<:Real} = mu * (1 - mu)
geevar(d::Gamma, v::DefaultVar, mu::T) where {T<:Real} = mu^2

geevar(::Distribution, v::ConstantVar, mu::T) where {T<:Real} = 1
geevar(::Distribution, v::IdentityVar, mu::T) where {T<:Real} = mu
geevar(::Distribution, v::PowerVar, mu::T) where {T<:Real} = mu^v.p
geevar(::Distribution, v::BinomialVar, mu::T) where {T<:Real} = mu*(1-mu)

geevarderiv(d::Normal, v::DefaultVar, mu::T) where {T<:Real} = zero(T)
geevarderiv(d::Poisson, v::DefaultVar, mu::T) where {T<:Real} = one(T)
geevarderiv(d::Binomial, v::DefaultVar, mu::T) where {T<:Real} = 1 - 2*mu
geevarderiv(d::Gamma, v::DefaultVar, mu::T) where {T<:Real} = 2*mu

geevarderiv(::Distribution, v::ConstantVar, mu::T) where {T<:Real} = zero(T)
geevarderiv(::Distribution, v::IdentityVar, mu::T) where {T<:Real} = one(T)
geevarderiv(::Distribution, v::PowerVar, mu::T) where {T<:Real} = v.p * mu^(v.p - 1)
geevarderiv(::Distribution, v::BinomialVar, mu::T) where {T<:Real} = 1 - 2*mu
