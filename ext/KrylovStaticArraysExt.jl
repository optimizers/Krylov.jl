module KrylovStaticArraysExt

using Krylov: Krylov
using StaticArrays: StaticVector

"""
    Krylov.ktypeof(::StaticVector{S,T}) where {S,T}

Return the corresponding `Vector{T}` type.
"""
Krylov.ktypeof(::StaticVector{S,T}) where {S,T} = Vector{T}

end
