"""
Gaussian distributions in precision parameterization (τ = μ/σ², ρ = 1/σ²), which makes
multiplication/division (the operations belief propagation needs) simple additions/subtractions
of natural parameters instead of the usual mean/variance combination formulas.
"""
module Gaussian

using Distributions
using Random
using StatsFuns: normcdf
import Base: iterate, length, size, eltype, IteratorSize

"""
    GaussianDist

Abstract parent of `Gaussian1D` (normalized) and `NonNormalizedGaussian1D` (carries an
explicit log-normalization constant, needed when tracking evidence/ELBO terms).
"""
abstract type GaussianDist end

"""
    Gaussian1D(τ::Float64, ρ::Float64)

Normalized 1D Gaussian in precision form: `τ = μ/σ²`, `ρ = 1/σ²` (`ρ ≥ 0`; `ρ = 0` is the
uniform/improper distribution).
"""
struct Gaussian1D <: GaussianDist
    τ::Float64            # precision-weighted mean, τ = μ·ρ = μ/σ²
    ρ::Float64            # precision, ρ = 1/σ²

    # inner constructor: rejects a negative precision
    Gaussian1D(τ, ρ) =
        (ρ < 0) ? error("precision of a Gaussian must be non-negative (ρ = $ρ)") :
        new(promote(τ, ρ)...)
end

"""
    NonNormalizedGaussian1D(τ::Float64, ρ::Float64, log_norm::Float64)

Like `Gaussian1D` but with an explicit `log_norm` (log of the normalization constant `Z`),
needed by message-passing updates that track evidence rather than just shape.
"""
struct NonNormalizedGaussian1D <: GaussianDist
    τ::Float64            # precision-weighted mean, τ = μ·ρ = μ/σ²
    ρ::Float64            # precision, ρ = 1/σ²
    log_norm::Float64     # log Z, Z the normalization constant

    # inner constructor: rejects a negative precision
    NonNormalizedGaussian1D(τ, ρ, log_norm) =
        (ρ < 0) ? error("precision of a Gaussian must be non-negative") :
        new(promote(τ, ρ, log_norm)...)
end


"""
    Gaussian1D() -> Gaussian1D

Standard normal N(0,1).
"""
Gaussian1D() = Gaussian1D(0.0, 1.0)

"""
    Gaussian1DFromMeanVariance(μ::Real, σ²::Real) -> Gaussian1D

Construct from mean/variance: `τ = μ/σ²`, `ρ = 1/σ²`. Requires `σ² > 0`.
"""
@inline function Gaussian1DFromMeanVariance(μ, σ2)
    inv_σ2 = inv(σ2)
    Gaussian1D(μ * inv_σ2, inv_σ2)
end

"""
    Gaussian1DUniform() -> Gaussian1D

Uniform (improper, `ρ=0`) distribution -- the non-informative prior / identity for `*`.
"""
Gaussian1DUniform() = Gaussian1D(0.0, 0.0)

"""
    Gaussian1D(g::NonNormalizedGaussian1D) -> Gaussian1D

Drop the normalization constant, keeping `(τ, ρ)`.
"""
Gaussian1D(g::NonNormalizedGaussian1D) = Gaussian1D(g.τ, g.ρ)

"""
    NonNormalizedGaussian1D(g::Gaussian1D) -> NonNormalizedGaussian1D

Lift to non-normalized form with `log_norm = 0.0`.
"""
NonNormalizedGaussian1D(g::Gaussian1D) = NonNormalizedGaussian1D(g.τ, g.ρ, 0.0)


"""
    is_uniform(g::GaussianDist) -> Bool

`true` iff `g.ρ == 0` (improper/non-informative).
"""
function is_uniform(g::T) where T<:GaussianDist
    return g.ρ == 0.0
end

"""
    mean(g::GaussianDist) -> Float64

`τ/ρ`, or `0.0` for a uniform distribution.
"""
function mean(g::T) where T<:GaussianDist
    if is_uniform(g)
        return 0.0
    else
        return g.τ / g.ρ
    end
end

"""
    variance(g::GaussianDist) -> Float64

`1/ρ`, or `+Inf` for a uniform distribution.
"""
function variance(g::T) where T<:GaussianDist
    if is_uniform(g)
        return +Inf
    else
        return 1.0 / g.ρ
    end
end

"""
    *(g1::Gaussian1D, g2::Gaussian1D) -> Gaussian1D

Product of two Gaussian densities: adds natural parameters, `(τ₁+τ₂, ρ₁+ρ₂)`.
"""
@inline function Base.:*(g1::Gaussian1D, g2::Gaussian1D)
    return (Gaussian1D(g1.τ + g2.τ, g1.ρ + g2.ρ))
end

"""
    mult_fast(g1, g2) -> Gaussian1D

`*` without any extra checks.
"""
@inline function mult_fast(g1::Gaussian1D, g2::Gaussian1D)
    return Gaussian1D(g1.τ + g2.τ, g1.ρ + g2.ρ)
end

"""
    *(g1::NonNormalizedGaussian1D, g2::Gaussian1D) -> NonNormalizedGaussian1D

Product tracking the normalization constant via the two distributions' overlap integral.
"""
function Base.:*(g1::NonNormalizedGaussian1D, g2::Gaussian1D)
    log_norm =
        if is_uniform(g1) || is_uniform(g2)
            0.0
        else
            σ2_sum = variance(g1) + variance(g2)
            μ_diff = mean(g1) - mean(g2)
            log_norm = -0.5 * (log(2.0 * π * σ2_sum) + μ_diff * μ_diff / σ2_sum)
        end

    return (NonNormalizedGaussian1D(g1.τ + g2.τ, g1.ρ + g2.ρ, g1.log_norm + log_norm))
end

"""
    *(g1::Gaussian1D, g2::NonNormalizedGaussian1D) -> NonNormalizedGaussian1D

Commutative counterpart of `*(::NonNormalizedGaussian1D, ::Gaussian1D)`.
"""
Base.:*(g1::Gaussian1D, g2::NonNormalizedGaussian1D) = g2 * g1

"""
    /(g1::Gaussian1D, g2::Gaussian1D) -> Gaussian1D

Quotient of two Gaussian densities: `(τ₁-τ₂, ρ₁-ρ₂)`. Errors if `g1` is uniform and `g2` is
not (a uniform can only be divided by another uniform); `ρ₁ ≥ ρ₂` is required.
"""
@inline function Base.:/(g1::Gaussian1D, g2::Gaussian1D)
    if (g1.ρ - g2.ρ < -1e-4) && (variance(g1) - variance(g2) < -1e-8)
        error("The precision of the divisor is too large (", g1.ρ - g2.ρ, ") given g1 = ", g1, " and g2 = ", g2, ")")
    end
    if is_uniform(g1)
        if is_uniform(g2)
            return Gaussian1DUniform()
        else
            error("A uniform cannot be divided by anything else than a Gaussian uniform")
        end
    else
        if g2.ρ > g1.ρ
            @warn "Gaussian division: the divisor's precision exceeds the dividend's; the quotient's precision is clamped to 0."
        end
        return (Gaussian1D(g1.τ - g2.τ, max(g1.ρ - g2.ρ, 0.0)))
    end
end

"""
    div_fast(g1, g2) -> Gaussian1D

`/` without any extra checks (caller must ensure validity).
"""
@inline function div_fast(g1::Gaussian1D, g2::Gaussian1D)
    # assumes caller ensures validity and non-uniform constraints
    return Gaussian1D(g1.τ - g2.τ, g1.ρ - g2.ρ)
end

"""
    /(g1::NonNormalizedGaussian1D, g2::Gaussian1D) -> NonNormalizedGaussian1D

Quotient tracking the normalization constant. Same constraints as `/(::Gaussian1D, ::Gaussian1D)`.
"""
function Base.:/(g1::NonNormalizedGaussian1D, g2::Gaussian1D)
    if (g1.ρ - g2.ρ < -1e-4) && (variance(g1) - variance(g2) < -1e-8)
        error("The precision of the divisor is too large (", g1.ρ - g2.ρ, ") given g1 = ", g1, " and g2 = ", g2, ")")
    end
    if is_uniform(g1)
        if is_uniform(g2)
            return NonNormalizedGaussian1D(Gaussian1DUniform())
        else
            error("A uniform cannot be divided by anything else than a Gaussian uniform")
        end
    else
        log_norm =
            if is_uniform(g2)
                0.0
            else
                g2_var = variance(g2)
                σ2_diff = max(g2_var - variance(g1), 0.0)
                if (σ2_diff == 0.0)
                    0.0
                else
                    μ_diff = mean(g1) - mean(g2)
                    log(g2_var) + 0.5 * (log(2 * π / σ2_diff) + μ_diff * μ_diff / σ2_diff)
                end
            end
        return NonNormalizedGaussian1D(g1.τ - g2.τ, max(g1.ρ - g2.ρ, 0.0), g1.log_norm + log_norm)
    end
end

"""
    +(g1, g2)

Sum of two independent Gaussians. Uniform if either input is.
"""
function Base.:+(g1::Gaussian1D, g2::Gaussian1D)
    if is_uniform(g1) || is_uniform(g2)
        return Gaussian1DUniform()
    end#

    # μ_sum = μ1+μ2, σ²_sum = σ1²+σ2²; in natural parameters ρ_sum = ρ1ρ2/(ρ1+ρ2) and
    # τ_sum = (τ1ρ2 + τ2ρ1)/(ρ1+ρ2).
    denom = g1.ρ + g2.ρ
    return (denom == 0.0) ? Gaussian1DUniform() : Gaussian1D((g1.τ * g2.ρ + g2.τ * g1.ρ) / denom, g1.ρ * g2.ρ / denom)
end

"""
    +(g, c)

Shift the mean of `g` by constant `c`; variance unchanged. Uniform stays uniform.
"""
function Base.:+(g::Gaussian1D, c::Real)
    is_uniform(g) && return Gaussian1DUniform()
    return Gaussian1DFromMeanVariance(mean(g) + c, variance(g))
end

Base.:+(c::Real, g::Gaussian1D) = g + c

"""
    -(g, c)

Shift the mean of `g` by `-c`; variance unchanged.
"""
function Base.:-(g::Gaussian1D, c::Real)
    is_uniform(g) && return Gaussian1DUniform()
    return Gaussian1DFromMeanVariance(mean(g) - c, variance(g))
end

"""
    -(c, g)

`c - X` for `X ~ g`: negates the mean about `c`; variance unchanged.
"""
function Base.:-(c::Real, g::Gaussian1D)
    is_uniform(g) && return Gaussian1DUniform()
    return Gaussian1DFromMeanVariance(c - mean(g), variance(g))
end

"""
    cdf(g, x)

`P(G <= x)`. Undefined (errors) for a uniform Gaussian.
"""
function cdf(g::T, x::Real) where T<:GaussianDist
    if is_uniform(g)
        error("CDF is undefined for a Gaussian uniform distribution")
    end

    return normcdf(mean(g), sqrt(variance(g)), x)
end

"""
    KL_divergence(g1::T, g2::T) where T<:GaussianDist -> Float64

`D_KL(g1 || g2)`. `0.0` if either is uniform (no information to diverge on); values below
`1e-5` are snapped to `0.0`.
"""
function KL_divergence(g1::T, g2::T) where T<:GaussianDist
    if is_uniform(g1) || is_uniform(g2)
        return 0.0
    else
        μ_diff = mean(g1) - mean(g2)
        σ1_2 = variance(g1)
        σ2_2 = variance(g2)
        kl = 0.5 * (log(σ2_2 / σ1_2) + (σ1_2 + μ_diff * μ_diff) / σ2_2 - 1)
        if kl < 1e-5
            kl = 0.0
        end
        return kl
    end
end

"""
    show(io::IO, g::Gaussian1D)

`"μ = .., σ = .."`, or `"uniform"`.
"""
function Base.show(io::IO, g::Gaussian1D)
    if (is_uniform(g))
        print(io, "uniform")
    else
        print(io, "μ = ", mean(g), ", σ = ", sqrt(variance(g)))
    end
end


"""
    show(io::IO, g::NonNormalizedGaussian1D)

`"μ = .., σ = .., Z = .."`, or `"uniform (Z = ..)"`.
"""
function Base.show(io::IO, g::NonNormalizedGaussian1D)
    if (is_uniform(g))
        print(io, "uniform (Z = " , exp(g.log_norm), ")")
    else
        print(io, "μ = ", mean(g), ", σ = ", sqrt(variance(g)), ", Z = ", exp(g.log_norm))
    end
end

"""
    to_distribution(g::GaussianDist) -> Distributions.Normal

Convert to a `Distributions.Normal(μ, σ)` for interop (sampling, quantiles, ...). Throws
`DomainError` for a uniform distribution (infinite variance).
"""
function to_distribution(g::GaussianDist)
    if is_uniform(g)
        throw(DomainError("Uniform distribution cannot be converted to a Distributions.Normal object"))
    else
        return Distributions.Normal(mean(g), sqrt(variance(g)))
    end
end

const LOG2PI_F64 = log(2π)

"""
    logpdf(g::GaussianDist, x::T) where {T<:AbstractFloat} -> T

Log-density of `g` at `x`, computed directly from `(τ, ρ)` for numerical stability. Throws
`DomainError` for a uniform (improper) distribution.
"""
@inline function logpdf(g::GaussianDist, x::T) where {T<:AbstractFloat}
    if is_uniform(g)
        throw(DomainError(
            g.ρ,
            "logpdf is undefined for uniform (improper) Gaussian distributions with ρ == 0. " *
            "Use a proper Gaussian with positive precision (ρ > 0) instead.",
        ))
    end
    quad = muladd(g.ρ, x*x, -2*g.τ*x) + (g.τ*g.τ)/g.ρ
    return T(0.5)*log(g.ρ) - T(0.5*LOG2PI_F64) - T(0.5)*quad
end

 #Treat Gaussian1D as one scalar-like item
iterate(g::Gaussian1D) = (g, nothing)
iterate(::Gaussian1D, ::Nothing) = nothing

length(::Gaussian1D) = 1
size(::Gaussian1D) = (1,)
eltype(::Type{Gaussian1D}) = Gaussian1D
IteratorSize(::Type{Gaussian1D}) = Base.HasLength()

export Gaussian1D, NonNormalizedGaussian1D, Gaussian1DFromMeanVariance, Gaussian1DUniform, is_uniform
export *, /, +, -, KL_divergence, show, to_distribution, cdf, mean, variance, logpdf
export div_fast, mult_fast

end
