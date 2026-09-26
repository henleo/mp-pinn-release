"""
Factor graph / EP library for `mp_pinn_model.jl`:
a fixed-coefficient scalar product factor (`z = Σᵢ yᵢ·xᵢ`), the piecewise-exponential
activation factor with its exact moments, and the fused data/residual observation factors.

`db_marginal`/`db_msg` (`DistributionBag`s of `Gaussian1D`) hold beliefs and messages;
`update_msg_to_*!` functions update them and return a KL divergence for convergence checks.
"""
module Factors

include("../collections/distribution_collection.jl")
include("../distributions/gaussian.jl")

using .DistributionCollections
using .Gaussian
using StatsFuns: normcdf, normpdf

# Abstract base type for all factor implementations
abstract type Factor end

# Debug logging (MSG_DEBUG flag, ring buffer); used by the update functions.
include("msg_debug.jl")

include("factors_scalar_product.jl")          # Scalar product factor: z = Σᵢ xᵢ·yᵢ (fixed y)
include("factors_piecewise_activation.jl")    # Piecewise activation exp(βx)-1 for x>0, -α(exp(-βx)-1) for x≤0 (exact closed form)
include("factors_pinn.jl")                    # Data/residual observation factors for u = A*g(w*φ)
include("factors_pinn_wide.jl")               # Wide (H-neuron, fixed readout) versions of the two above

# Public API - Factor types
export Factor
export PiecewiseActivationFunctionFactor
export ScalarProductFactorWithFixedY
export ResidualFactor, DataObsFactor
export WideResidualFactor, WideDataObsFactor

# Public API - Message update functions
export update_msg_to_y!, update_msg_to_z!,
    piecewise_activation, piecewise_activation_derivative,
    piecewise_activation_update_to_y, PIECEWISE_BETA,
    piecewise_tilted_moments, piecewise_tilted_moments_neg, piecewise_activation_moments,
    piecewise_activation_derivative_moments,
    piecewise_derivative_component_moments,
    residual_forward_message, residual_backward_message_k, residual_backward_update!,
    data_forward_message, data_backward_message_k, data_backward_update!,
    wide_data_forward_message, wide_data_backward_update!,
    wide_residual_forward_message, wide_residual_backward_message_k, wide_residual_backward_update!

# Public API - Debug logging
export MSG_DEBUG, var_name, dump_msg_log

end  # module Factors
