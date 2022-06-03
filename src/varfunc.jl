abstract type Varfunc end

# Make varfuncs broadcast like a scalar
Base.Broadcast.broadcastable(vf::Varfunc) = Ref(vf)

struct ConstantVar <: Varfunc end

struct IdentityVar <: Varfunc end

# Used when the variance is specified through a distribution/family
# rather than an explicit variance function.
struct NullVar <: Varfunc end

struct PowerVar <: Varfunc
    p::Float64
end

geevar(::ConstantVar, mu::T) where {T<:Real} = 1
geevar(::IdentityVar, mu::T) where {T<:Real} = mu
geevar(v::PowerVar, mu::T) where {T<:Real} = mu^v.p

geevarderiv(::ConstantVar, mu::T) where {T<:Real} = zero(T)
geevarderiv(::IdentityVar, mu::T) where {T<:Real} = one(T)
geevarderiv(v::PowerVar, mu::T) where {T<:Real} = v.p * mu^(v.p - 1)
