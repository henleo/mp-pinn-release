"""
    msg_debug.jl — buffered, NaN-tripped message-passing debug logger

A single switch (`MSG_DEBUG`) controls everything. When its integer value `N` is `> 0`, every
instrumented `update_msg_to_*!` call records one entry per target variable into a ring buffer that
keeps only the last `N`. The buffer is dumped to a local file `MSG_DEBUG.log` and the run is
interrupted with an `error` the moment a persisted message/marginal τ/ρ becomes non-finite (Inf or
NaN), or an intermediate scratch value becomes NaN (Inf intermediates are allowed — e.g. the
variance of a uniform message).

Turn it on in exactly one place:
- environment: `MSG_DEBUG=200 julia …`, or
- code: `Factors.MSG_DEBUG[] = 200`

When off, the only cost on the hot path is a single integer comparison (`msg_debug_on()`).
"""

# ── Activation flag (the one place) ─────────────────────────────────────────────
# N > 0 enables logging; N is the ring-buffer capacity (number of retained entries).
const MSG_DEBUG = Ref(parse(Int, get(ENV, "MSG_DEBUG", "20")))

@inline msg_debug_on() = MSG_DEBUG[] > 0

# ── Variable-name lookup: marginal index → human name ────────────────────────────
# Optional registry; an index without an entry is shown as "#idx".
const VAR_NAMES = Dict{Int,String}()

var_name(idx::Integer) = get(VAR_NAMES, idx, "#$idx")

# ── Factor-name lookup: factor object → code-variable name ──────────────────────
# Optional registry, like VAR_NAMES; `factor_label` falls back to the type name.
const FACTOR_NAMES = IdDict{Any,String}()

# Factor label: "<code-variable-name>(<StructType>)", or just the type when unregistered.
function factor_label(f)
    t = nameof(typeof(f))
    name = get(FACTOR_NAMES, f, "")
    return isempty(name) ? string(t) : string(name, "(", t, ")")
end

# ── Ring buffer of update snapshots ─────────────────────────────────────────────
struct MsgEntry
    factor::String
    position::String                              # "z", "x[3]", "y[2]", …
    target::String                               # variable name being updated
    # messages arriving from the factor's *other* variables, used to compute the new message to the
    # target; each kept as name => (τ, ρ).
    messages_from_others::Vector{Pair{String,Tuple{Float64,Float64}}}
    old_msg::Tuple{Float64,Float64}              # previous factor→target message (τ, ρ)
    old_marginal::Tuple{Float64,Float64}
    new_msg::Tuple{Float64,Float64}
    new_marginal::Tuple{Float64,Float64}
    intermediates::Vector{Pair{String,Float64}}
end

const MSG_LOG = MsgEntry[]

@inline _tr(g) = (g.τ, g.ρ)

"""
    record_msg!(; factor, position, target, messages_from_others, old_msg, old_marginal,
                new_msg, new_marginal, intermediates)

Append one snapshot of a message update. `messages_from_others` is a vector of `name => Gaussian1D`
(the messages from the factor's other variables); `old_*`/`new_*` are `Gaussian1D`; `intermediates`
is a vector of `name => Float64`. Trims the buffer to the last `MSG_DEBUG[]` entries, then dumps/halts
if any persisted message/marginal τ/ρ is non-finite or any intermediate is NaN.
"""
function record_msg!(;
    factor::AbstractString,
    position::AbstractString,
    target::AbstractString,
    messages_from_others = Pair{String,Any}[],
    old_msg,
    old_marginal,
    new_msg,
    new_marginal,
    intermediates = Pair{String,Float64}[],
)
    entry = MsgEntry(
        String(factor),
        String(position),
        String(target),
        [String(k) => _tr(v) for (k, v) in messages_from_others],
        _tr(old_msg),
        _tr(old_marginal),
        _tr(new_msg),
        _tr(new_marginal),
        [String(k) => Float64(v) for (k, v) in intermediates],
    )
    push!(MSG_LOG, entry)
    while length(MSG_LOG) > MSG_DEBUG[]
        popfirst!(MSG_LOG)
    end
    _check_and_dump(entry)
    return nothing
end

# τ/ρ of the actual messages/marginals — these get persisted into the distribution bags.
function _message_values(e::MsgEntry)
    vals = Float64[]
    for (_, (τ, ρ)) in e.messages_from_others
        push!(vals, τ, ρ)
    end
    for t in (e.old_msg, e.old_marginal, e.new_msg, e.new_marginal)
        push!(vals, t[1], t[2])
    end
    return vals
end

# scratch quantities used to compute the messages (denom, a, c, d, μ, …)
_intermediate_values(e::MsgEntry) = [v for (_, v) in e.intermediates]

function _check_and_dump(e::MsgEntry)
    # Persisted message/marginal τ/ρ must be finite — Inf or NaN there is a real blowup. Intermediate
    # scratch values may legitimately be Inf (e.g. the variance of a uniform message), so only their
    # NaNs are errors; an Inf intermediate that later yields a NaN is caught at that point.
    if any(!isfinite, _message_values(e)) || any(isnan, _intermediate_values(e))
        n = length(MSG_LOG)
        dump_msg_log("MSG_DEBUG.log")
        empty!(MSG_LOG)   # buffer is now on disk; clear it so a later trip starts fresh
        error(
            "MSG_DEBUG: non-finite value while updating $(e.target) (position $(e.position)) of " *
            "$(e.factor); dumped last $n entries to ./MSG_DEBUG.log",
        )
    end
    return nothing
end

"""
    dump_msg_log(path)

Write the buffered entries (oldest → newest) to `path` in a readable form.
"""
function dump_msg_log(path::AbstractString)
    open(path, "w") do io
        for (n, e) in enumerate(MSG_LOG)
            println(io, "── entry $n / $(length(MSG_LOG)) ──")
            println(io, "factor:        $(e.factor)")
            println(io, "target:        $(e.target)  (position $(e.position))")
            for (name, (τ, ρ)) in e.messages_from_others
                println(io, "  msg from $name: τ=$(τ) ρ=$(ρ)")
            end
            println(io, "  old msg→tgt:   τ=$(e.old_msg[1]) ρ=$(e.old_msg[2])")
            println(io, "  old marginal:  τ=$(e.old_marginal[1]) ρ=$(e.old_marginal[2])")
            println(io, "  new msg→tgt:   τ=$(e.new_msg[1]) ρ=$(e.new_msg[2])")
            println(io, "  new marginal:  τ=$(e.new_marginal[1]) ρ=$(e.new_marginal[2])")
            for (name, v) in e.intermediates
                println(io, "  $name = $v")
            end
            println(io)
        end
    end
    return path
end
