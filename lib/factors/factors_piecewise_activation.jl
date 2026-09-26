const PIECEWISE_BETA = 0.1

"""
    PiecewiseActivationFunctionFactor <: Factor

Piecewise exponential activation, mirrored (not scaled) on the negative side:

    Y = g(X) = exp(βX) - 1          for X > 0
             = -α·(exp(-βX) - 1)     for X ≤ 0        (β = PIECEWISE_BETA, 0<α<1)

`g(0)=0`, `g'(0⁺)=β`, `g'(0⁻)=αβ` -- leaky-ReLU-like, exponential tails, strictly increasing.
"""
struct PiecewiseActivationFunctionFactor <: Factor
    db_marginal::DistributionBag{Gaussian1D}
    db_msg::DistributionBag{Gaussian1D}
    x::Int64
    y::Int64
    α::Float64
    msg_to_x::Int64
    msg_to_y::Int64
end

function PiecewiseActivationFunctionFactor(db_marginal, db_msg, x, y, α)
    (0.0 < α < 1.0) || throw(ArgumentError("α must satisfy 0<α<1 (got $α)"))
    PiecewiseActivationFunctionFactor(db_marginal, db_msg, x, y, α, add!(db_msg), add!(db_msg))
end

@inline function piecewise_activation(x::Float64, α::Float64)
    x > 0 ? expm1(PIECEWISE_BETA * x) : -α * expm1(-PIECEWISE_BETA * x)
end

"""
    piecewise_activation_derivative(x, α) -> Float64

`g'(x)`: `β·exp(βx)` for `x>0`, `αβ·exp(-βx)` for `x≤0`. Positive on both sides.
"""
@inline function piecewise_activation_derivative(x::Float64, α::Float64)
    x > 0 ? PIECEWISE_BETA * exp(PIECEWISE_BETA * x) : α * PIECEWISE_BETA * exp(-PIECEWISE_BETA * x)
end

"""
    piecewise_tilted_moments(μ_X, σ_X, t) -> (F0, F1, F2)

Exact `(E[exp(tX)·1_{X>0}], E[X·exp(tX)·1_{X>0}], E[X²·exp(tX)·1_{X>0}])` for `X~N(μ_X,σ_X²)`,
closed form via `∫₀^∞ exp(tx)f_X(x)dx = exp(tμ_X+t²σ_X²/2)·Φ(u)`, `u=μ_X/σ_X+tσ_X`, and its
1st/2nd `t`-derivatives. Shared by every forward moment below (`t=β,2β`).

    F0(t) = exp(tμ_X + t²σ_X²/2)·Φ(u),   u = μ_X/σ_X + tσ_X
    F1(t) = (μ_X+tσ_X²)·F0(t) + σ_X·exp(tμ_X+t²σ_X²/2)·φ(u)
    F2(t) = (μ_X+tσ_X²)·F1(t) + σ_X²·F0(t)
"""
function piecewise_tilted_moments(μ_X::Float64, σ_X::Float64, t::Float64)
    g = t * μ_X + 0.5 * t^2 * σ_X^2
    u = μ_X / σ_X + t * σ_X
    expg = exp(g)
    Φu, φu = normcdf(u), normpdf(u)
    F0 = expg * Φu
    F1 = (μ_X + t * σ_X^2) * F0 + σ_X * expg * φu
    F2 = (μ_X + t * σ_X^2) * F1 + σ_X^2 * F0
    return F0, F1, F2
end

"""
    piecewise_tilted_moments_neg(μ_X, σ_X, t) -> (F0, F1, F2)

Mirror image of `piecewise_tilted_moments` for `X≤0`, via `X'=-X~N(-μ_X,σ_X²)`: `F0`/`F2` come
directly from `piecewise_tilted_moments(-μ_X,σ_X,-t)`, `F1` picks up an extra minus sign.
"""
function piecewise_tilted_moments_neg(μ_X::Float64, σ_X::Float64, t::Float64)
    F0, F1, F2 = piecewise_tilted_moments(-μ_X, σ_X, -t)
    return F0, -F1, F2
end

"""
    piecewise_activation_derivative_moments(μ_X, σ_X, α) -> (Float64, Float64)

Exact `(E[g'(X)], E[g''(X)])` for `X~N(μ_X,σ_X²)` -- the statistically correct linearization
coefficients for `h(B,X)=B·g'(X)`, not the point evaluations `g'(μ_X)`/`g''(μ_X)` (`g'`/`g''`
are exponential, so `E[g'(X)] ≠ g'(E[X])`, a Jensen's-inequality gap growing with `σ_X²`):

    E[g'(X)]  = β·E[exp(βX)·1_{X>0}]  + αβ·E[exp(-βX)·1_{X≤0}]
    E[g''(X)] = β²·E[exp(βX)·1_{X>0}] - αβ²·E[exp(-βX)·1_{X≤0}]
"""
function piecewise_activation_derivative_moments(μ_X::Float64, σ_X::Float64, α::Float64)
    F0_pos, = piecewise_tilted_moments(μ_X, σ_X, PIECEWISE_BETA)
    F0_neg, = piecewise_tilted_moments_neg(μ_X, σ_X, -PIECEWISE_BETA)
    β = PIECEWISE_BETA
    E_gp = β * F0_pos + α * β * F0_neg
    E_gpp = β^2 * F0_pos - α * β^2 * F0_neg
    return E_gp, E_gpp
end

"""
    piecewise_activation_moments(μ_X, σ_X, α) -> (Float64, Float64)

Exact `(E[Y], Var(Y))` for `Y=g(X)`, `X~N(μ_X,σ_X²)`. With `p₊=Φ(μ_X/σ_X)=P(X>0)`, `p₋=1-p₊`:

    E[Y]  = (F0₊(β) - p₊) - α·(F0₋(-β) - p₋)
    E[Y²] = (F0₊(2β) - 2F0₊(β) + p₊) + α²·(F0₋(-2β) - 2F0₋(-β) + p₋)
"""
function piecewise_activation_moments(μ_X::Float64, σ_X::Float64, α::Float64)
    p_pos = normcdf(μ_X / σ_X)
    p_neg = 1.0 - p_pos

    F0_pos_1, = piecewise_tilted_moments(μ_X, σ_X, PIECEWISE_BETA)
    F0_pos_2, = piecewise_tilted_moments(μ_X, σ_X, 2.0PIECEWISE_BETA)
    F0_neg_1, = piecewise_tilted_moments_neg(μ_X, σ_X, -PIECEWISE_BETA)
    F0_neg_2, = piecewise_tilted_moments_neg(μ_X, σ_X, -2.0PIECEWISE_BETA)

    μ_Y = (F0_pos_1 - p_pos) - α * (F0_neg_1 - p_neg)
    E_Y2 = (F0_pos_2 - 2.0F0_pos_1 + p_pos) + α^2 * (F0_neg_2 - 2.0F0_neg_1 + p_neg)
    return μ_Y, E_Y2 - μ_Y^2
end

"""
    piecewise_activation_update_to_y(msg_from_x, α) -> Gaussian1D

Exact forward moment match of `Y=g(X)` (see `piecewise_activation_moments`); falls back to
uniform if `msg_from_x` is uniform or the moments overflow to non-finite/non-positive.
"""
function piecewise_activation_update_to_y(msg_from_x::Gaussian1D, α::Float64)
    if Gaussian.is_uniform(msg_from_x)
        return Gaussian1D(0.0, 0.0)
    end
    μ_X, σ2_X = Gaussian.mean(msg_from_x), Gaussian.variance(msg_from_x)
    μ_Y, σ2_Y = piecewise_activation_moments(μ_X, sqrt(σ2_X), α)
    if !(isfinite(μ_Y) && isfinite(σ2_Y) && σ2_Y >= 0.0)
        @warn "PiecewiseActivationFunctionFactor: non-finite/non-positive moments (exp overflow?), using uniform message" μ_X σ2_X μ_Y σ2_Y
        return Gaussian1DUniform()
    end
    return Gaussian1DFromMeanVariance(μ_Y, σ2_Y)
end

function update_msg_to_y!(f::PiecewiseActivationFunctionFactor)
    msg_from_x = f.db_marginal[f.x] / f.db_msg[f.msg_to_x]
    new_msg_to_y = piecewise_activation_update_to_y(msg_from_x, f.α)

    old_marginal_y = f.db_marginal[f.y]
    old_msg_to_y = f.db_msg[f.msg_to_y]
    msg_from_y = old_marginal_y / old_msg_to_y
    new_marginal_y = msg_from_y * new_msg_to_y
    f.db_marginal[f.y] = new_marginal_y
    f.db_msg[f.msg_to_y] = new_msg_to_y
    if msg_debug_on()
        record_msg!(;
            factor = factor_label(f), position = "y", target = var_name(f.y),
            messages_from_others = ["x " * var_name(f.x) => msg_from_x],
            old_msg = old_msg_to_y, old_marginal = old_marginal_y,
            new_msg = new_msg_to_y, new_marginal = new_marginal_y,
            intermediates = Pair{String,Float64}[],
        )
    end
    return Gaussian.KL_divergence(old_marginal_y, new_marginal_y)
end

"""
    piecewise_activation_times_derivative_moment(μ_X, σ_X, α) -> Float64

Exact `E[g(X)·g'(X)]` for `X~N(μ_X,σ_X²)`:

    g(X)·g'(X) = β·(exp(2βX) - exp(βX))            for X>0
               = α²β·(exp(-βX) - exp(-2βX))          for X≤0

    E[g(X)g'(X)] = β·(F0₊(2β) - F0₊(β)) + α²β·(F0₋(-β) - F0₋(-2β))

Verified against Monte Carlo at several `(μ_X,σ_X,α)` (relative error < 1% at 2e7 samples).
Needed by `ResidualFactor`'s reaction-term sensitivity `H_Z` (`d/dX[g(X)²]=2g(X)g'(X)`).
"""
function piecewise_activation_times_derivative_moment(μ_X::Float64, σ_X::Float64, α::Float64)
    β = PIECEWISE_BETA
    F0_pos_1, = piecewise_tilted_moments(μ_X, σ_X, β)
    F0_pos_2, = piecewise_tilted_moments(μ_X, σ_X, 2.0β)
    F0_neg_1, = piecewise_tilted_moments_neg(μ_X, σ_X, -β)
    F0_neg_2, = piecewise_tilted_moments_neg(μ_X, σ_X, -2.0β)
    return β * (F0_pos_2 - F0_pos_1) + α^2 * β * (F0_neg_1 - F0_neg_2)
end

"""
    piecewise_derivative_component_moments(μ_B, σ_B, Xi, μ_X, σ_X, α) -> (Float64, Float64)

Exact `(E[D], Var(D))` for `D=B·g'(X)` (`B=Wᵢ`), `B~N(μ_B,σ_B²)`, `X~N(μ_X,σ_X²)`, jointly
Gaussian with correlation `ρ=Xi·σ_B/σ_X` (since `X=ΣⱼWⱼΦⱼ` implies `Cov(Wᵢ,X)=Φᵢ·Var(Wᵢ)`).

Writes `B=c₀+kX+ε` (`k=ρσ_B/σ_X`, `c₀=μ_B-kμ_X`, `ε⊥X`, `Var(ε)=σ_B²(1-ρ²)`) and integrates
out `ε` first:

    E[D·1_{X>0}]  = β·(c₀·F0₊(β) + k·F1₊(β))
    E[D²·1_{X>0}] = β²·((σ_B²(1-ρ²)+c₀²)·F0₊(2β) + 2c₀k·F1₊(2β) + k²·F2₊(2β))

The `X≤0` side mirrors this via `piecewise_tilted_moments_neg` with an extra `α`/`α²`; totals
add directly, `Var(D)=E[D²]-E[D]²`.
"""
function piecewise_derivative_component_moments(μ_B::Float64, σ_B::Float64, Xi::Float64, μ_X::Float64, σ_X::Float64, α::Float64)
    ρ = Xi * σ_B / σ_X
    k = ρ * σ_B / σ_X
    c0 = μ_B - k * μ_X
    β = PIECEWISE_BETA

    F0p1, F1p1, = piecewise_tilted_moments(μ_X, σ_X, β)
    F0p2, F1p2, F2p2 = piecewise_tilted_moments(μ_X, σ_X, 2.0β)
    F0n1, F1n1, = piecewise_tilted_moments_neg(μ_X, σ_X, -β)
    F0n2, F1n2, F2n2 = piecewise_tilted_moments_neg(μ_X, σ_X, -2.0β)

    varB_given_X = σ_B^2 * (1.0 - ρ^2)

    E_D_pos = β * (c0 * F0p1 + k * F1p1)
    E_D2_pos = β^2 * ((varB_given_X + c0^2) * F0p2 + 2.0c0 * k * F1p2 + k^2 * F2p2)

    E_D_neg = α * β * (c0 * F0n1 + k * F1n1)
    E_D2_neg = (α * β)^2 * ((varB_given_X + c0^2) * F0n2 + 2.0c0 * k * F1n2 + k^2 * F2n2)

    E_D = E_D_pos + E_D_neg
    E_D2 = E_D2_pos + E_D2_neg
    return E_D, E_D2 - E_D^2
end
