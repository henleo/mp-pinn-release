abstract type AbstractScalarProductFactor <: Factor end

save_msg_division(marginal, msg) = Gaussian.is_uniform(marginal) ? Gaussian1DUniform() : marginal / msg

"""
    ScalarProductFactorWithFixedY <: AbstractScalarProductFactor

Dot-product constraint `z = Σᵢ yᵢ·xᵢ` with fixed coefficients `y`. Forward-only: used to read a
projection `z` off existing Gaussian beliefs on `x` (e.g. `predict_slice`'s `z = w·φ`), not to
update `x` itself -- the weights' own beliefs are updated exclusively by the fused data/residual
factors. For `xᵢ ~ N(μᵢ, σᵢ²)`: `z ~ N(Σᵢ yᵢμᵢ, Σᵢ yᵢ²σᵢ²)`.
"""
struct ScalarProductFactorWithFixedY <: AbstractScalarProductFactor
  db_gauss_marginal::DistributionBag{Gaussian1D}
  db_gauss_msg::DistributionBag{Gaussian1D}
  x::Vector{Int64}
  y::Vector{Float64}
  z::Int64
  msg_to_x::Vector{Int64}
  msg_to_z::Int64
end

"""
    ScalarProductFactorWithFixedY(db_gauss_marginal, db_gauss_msg, x, y, z)

Constructs a `ScalarProductFactorWithFixedY` for `z = Σᵢ yᵢ·xᵢ`. Throws `ArgumentError` if
`length(x) != length(y)`.
"""
function ScalarProductFactorWithFixedY(db_gauss_marginal, db_gauss_msg, x, y, z)
  if (length(x) != length(y))
    throw(ArgumentError("Length of x and y must be equal"))
  end

  return ScalarProductFactorWithFixedY(
    db_gauss_marginal,
    db_gauss_msg,
    x,
    y,
    z,
    map(_ -> add!(db_gauss_msg), 1:length(x)),
    add!(db_gauss_msg),
  )
end

"""
    update_msg_to_z!(f::ScalarProductFactorWithFixedY)

Forward direction: `z ~ N(Σᵢ yᵢμᵢ, Σᵢ yᵢ²σᵢ²)` from each xᵢ's incoming message. Returns the KL
divergence between old and new marginal for `z`.
"""
function update_msg_to_z!(f::ScalarProductFactorWithFixedY)
    μ, σ2 = 0.0, 0.0
    @inbounds @simd for k in eachindex(f.x)
        msg_from_x_k_to_f = save_msg_division(f.db_gauss_marginal[f.x[k]], f.db_gauss_msg[f.msg_to_x[k]])
        μ += iszero(f.y[k]) ? 0.0 : f.y[k] * Gaussian.mean(msg_from_x_k_to_f)
        σ2 += iszero(f.y[k]) ? 0.0 : (f.y[k]^2) * Gaussian.variance(msg_from_x_k_to_f)
    end
    new_msg_to_z = if isfinite(μ) && isfinite(σ2)
        Gaussian1DFromMeanVariance(μ, σ2)
    else
        @warn "ScalarProductFactorWithFixedY update_msg_to_z!: using uniform message due to non-finite moments" μ σ2
        Gaussian1DUniform()
    end

    old_marginal_z = f.db_gauss_marginal[f.z]
    old_msg_to_z = f.db_gauss_msg[f.msg_to_z]
    msg_from_z = save_msg_division(old_marginal_z, old_msg_to_z)
    new_marginal_z = msg_from_z * new_msg_to_z
    f.db_gauss_marginal[f.z] = new_marginal_z
    f.db_gauss_msg[f.msg_to_z] = new_msg_to_z
    if msg_debug_on()
        others = [
            var_name(f.x[k]) => save_msg_division(f.db_gauss_marginal[f.x[k]], f.db_gauss_msg[f.msg_to_x[k]])
            for k in eachindex(f.x)
        ]
        record_msg!(;
            factor = factor_label(f), position = "z", target = var_name(f.z),
            messages_from_others = others,
            old_msg = old_msg_to_z, old_marginal = old_marginal_z,
            new_msg = new_msg_to_z, new_marginal = new_marginal_z,
            # NaN μ/σ2 (as opposed to the merely-Inf case _check_and_dump already tolerates for
            # intermediates) would otherwise trip the debug logger's own finiteness assertion,
            # even though new_msg_to_z above has already safely fallen back to uniform.
            intermediates = (isfinite(μ) && isfinite(σ2)) ? ["μ" => μ, "σ2" => σ2] : Pair{String,Float64}[],
        )
    end
    return Gaussian.KL_divergence(old_marginal_z, new_marginal_z)
end
