"""
`DistributionBag{T}`: a resizable `AbstractArray{T,1}` of distributions (e.g. beliefs/messages
in a factor graph), all starting uniform and addressed by integer index.
"""
module DistributionCollections

"""
    DistributionBag{T} <: AbstractArray{T, 1}

Resizable, type-safe container of distributions sharing type `T`. `uniform` is both the
template for new/reset entries and the type marker.
"""
struct DistributionBag{T} <: AbstractArray{T, 1}
    uniform::T      # initial/reset value of every entry (improper uniform of type T)
    bag::Vector{T}  # storage, grown by `add!`
    # inner constructor: empty storage
    DistributionBag{T}(uniform::T) where {T} = new(uniform, Vector{T}(undef, 0))
end

"""
    DistributionBag(uniform::T) where T -> DistributionBag{T}

Empty bag with type `T` inferred from `uniform`.
"""
DistributionBag(uniform::T) where {T} = DistributionBag{T}(uniform)

"""
    show(io::IO, db::DistributionBag{T}) where T

One line per entry, indexed; `"empty bag"` when empty.
"""
function Base.show(io::IO, db::DistributionBag{T}) where {T}
    println(io, "Uniform: ", db.uniform)
    if (length(db.bag) == 0)
        println(io, "  empty bag")
    else
        for i in eachindex(db.bag)
            println(io, "  [", i, "]: ", db.bag[i])
        end
    end
end

"""
    add!(db::DistributionBag{T}) where T -> Int

Append a uniform-initialized distribution; returns its (1-based) index.
"""
function add!(db::DistributionBag{T}) where {T}
    push!(db.bag, db.uniform)
    return (length(db.bag))
end

"""
    reset!(db::DistributionBag{T}) where T -> Nothing

Reset every entry to uniform in place (size unchanged).
"""
function reset!(db::DistributionBag{T}) where {T}
    @inbounds for i in eachindex(db.bag)
        db.bag[i] = db.uniform
    end
    return
end

"""
    getindex(db::DistributionBag{T}, i::Int64) where T -> T

`db[i]`, 1-based.
"""
Base.getindex(db::DistributionBag{T}, i::Int64) where {T} = db.bag[i]

"""
    setindex!(db::DistributionBag{T}, d::T, i::Int64) where T -> Nothing

`db[i] = d`.
"""
function Base.setindex!(db::DistributionBag{T}, d::T, i::Int64) where {T}
    db.bag[i] = d
end

Base.firstindex(db::DistributionBag{T}) where {T} = return (1)

Base.lastindex(db::DistributionBag{T}) where {T} = return (length(db.bag))

Base.size(db::DistributionBag{T}) where {T} = (length(db.bag),)

Base.IndexStyle(::Type{<:DistributionBag{T}}) where {T} = IndexLinear()

Base.eltype(::Type{<:DistributionBag{T}}) where {T} = T

Base.iterate(db::DistributionBag{T}, i=1) where {T} = (i > length(db.bag)) ? nothing : (db.bag[i], i+1)

export DistributionBag, add!, reset!

end
