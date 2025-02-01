export kstdout

"Default I/O stream for all Krylov methods."
const kstdout = Core.stdout

"""
    FloatOrComplex{T}
Union type of `T` and `Complex{T}` where T is an `AbstractFloat`.
"""
const FloatOrComplex{T} = Union{T, Complex{T}} where T <: AbstractFloat

"""
    (c, s, ρ) = sym_givens(a, b)

Numerically stable symmetric Givens reflection.
Given `a` and `b` reals, return `(c, s, ρ)` such that

    [ c  s ] [ a ] = [ ρ ]
    [ s -c ] [ b ] = [ 0 ].
"""
function sym_givens(a :: T, b :: T) where T <: AbstractFloat
  #
  # Modeled after the corresponding Matlab function by M. A. Saunders and S.-C. Choi.
  # http://www.stanford.edu/group/SOL/dissertations/sou-cheng-choi-thesis.pdf
  # D. Orban, Montreal, May 2015.

  if b == 0
    if a == 0
      c = one(T)
    else
      c = sign(a)  # In Julia, sign(0) = 0.
    end
    s = zero(T)
    ρ = abs(a)

  elseif a == 0
    c = zero(T)
    s = sign(b)
    ρ = abs(b)

  elseif abs(b) > abs(a)
    t = a / b
    s = sign(b) / sqrt(one(T) + t * t)
    c = s * t
    ρ = b / s  # Computationally better than ρ = a / c since |c| ≤ |s|.

  else
    t = b / a
    c = sign(a) / sqrt(one(T) + t * t)
    s = c * t
    ρ = a / c  # Computationally better than ρ = b / s since |s| ≤ |c|
  end

  return (c, s, ρ)
end

"""
Numerically stable symmetric Givens reflection.
Given `a` and `b` complexes, return `(c, s, ρ)` with
c real and (s, ρ) complexes such that

    [ c   s ] [ a ] = [ ρ ]
    [ s̅  -c ] [ b ] = [ 0 ].
"""
function sym_givens(a :: Complex{T}, b :: Complex{T}) where T <: AbstractFloat
  #
  # Modeled after the corresponding Fortran function by M. A. Saunders and S.-C. Choi.
  # A. Montoison, Montreal, March 2020.

  abs_a = abs(a)
  abs_b = abs(b)

  if abs_b == 0
    c = one(T)
    s = zero(Complex{T})
    ρ = a

  elseif abs_a == 0
    c = zero(T)
    s = one(Complex{T})
    ρ = b

  elseif abs_b > abs_a
    t = abs_a / abs_b
    c = one(T) / sqrt(one(T) + t * t)
    s = c * conj((b / abs_b) / (a / abs_a))
    c = c * t
    ρ = b / conj(s)

  else
    t = abs_b / abs_a
    c = one(T) / sqrt(one(T) + t * t)
    s = c * t * conj((b / abs_b) / (a / abs_a))
    ρ = a / c
  end

  return (c, s, ρ)
end

sym_givens(a :: Complex{T}, b :: T) where T <: AbstractFloat = sym_givens(a, Complex{T}(b))
sym_givens(a :: T, b :: Complex{T}) where T <: AbstractFloat = sym_givens(Complex{T}(a), b)

"""
    roots = roots_quadratic(q₂, q₁, q₀; nitref)

Find the real roots of the quadratic

    q(x) = q₂ x² + q₁ x + q₀,

where q₂, q₁ and q₀ are real. Care is taken to avoid numerical
cancellation. Optionally, `nitref` steps of iterative refinement
may be performed to improve accuracy. By default, `nitref=1`.
"""
function roots_quadratic(q₂ :: T, q₁ :: T, q₀ :: T;
                         nitref :: Int=1) where T <: AbstractFloat
  # Case where q(x) is linear.
  if q₂ == zero(T)
    if q₁ == zero(T)
      q₀ == zero(T) || error("The quadratic `q` doesn't have real roots.")
      root = zero(T)
    else
      root = -q₀ / q₁
    end
    return (root, root)
  end

  # Case where q(x) is indeed quadratic.
  rhs = √eps(T) * q₁ * q₁
  if abs(q₀ * q₂) > rhs
    ρ = q₁ * q₁ - 4 * q₂ * q₀
    ρ < 0 && return error("The quadratic `q` doesn't have real roots.")
    d = -(q₁ + copysign(sqrt(ρ), q₁)) / 2
    root1 = d / q₂
    root2 = q₀ / d
  else
    # Ill-conditioned quadratic.
    root1 = -q₁ / q₂
    root2 = zero(T)
  end

  # Perform a few Newton iterations to improve accuracy.
  for it = 1 : nitref
    q = (q₂ * root1 + q₁) * root1 + q₀
    dq = 2 * q₂ * root1 + q₁
    dq == zero(T) && continue
    root1 = root1 - q / dq
  end

  for it = 1 : nitref
    q = (q₂ * root2 + q₁) * root2 + q₀
    dq = 2 * q₂ * root2 + q₁
    dq == zero(T) && continue
    root2 = root2 - q / dq
  end
  return (root1, root2)
end

"""
    s = vec2str(x; ndisp)

Display an array in the form

    [ -3.0e-01 -5.1e-01  1.9e-01 ... -2.3e-01 -4.4e-01  2.4e-01 ]

with (ndisp - 1)/2 elements on each side.
"""
function vec2str(x :: AbstractVector{T}; ndisp :: Int=7) where T <: Union{AbstractFloat, Missing}
  n = length(x)
  if n ≤ ndisp
    ndisp = n
    nside = n
  else
    nside = max(1, div(ndisp - 1, 2))
  end
  s = "["
  i = 1
  while i ≤ nside
    if x[i] !== missing
      s *= @sprintf("%8.1e ", x[i])
    else
      s *= " ✗✗✗✗ "
    end
      i += 1
  end
  if i ≤ div(n, 2)
    s *= "... "
  end
  i = max(i, n - nside + 1)
  while i ≤ n
    if x[i] !== missing
      s *= @sprintf("%8.1e ", x[i])
    else
      s *= " ✗✗✗✗ "
    end
    i += 1
  end
  s *= "]"
  return s
end

"""
    S = ktypeof(v)

Return the most relevant storage type `S` based on the type of `v`.
"""
function ktypeof end

function ktypeof(v::S) where S <: DenseVector
  if S.name.name == :ComponentArray
    T = eltype(S)
    return Vector{T}
  else
    return S
  end
end

function ktypeof(v::S) where S <: DenseMatrix
  return S
end

function ktypeof(v::S) where S <: AbstractVector
  if S.name.name == :Zeros || S.name.name == :Ones || S.name.name == :SArray || S.name.name == :MArray || S.name.name == :SizedArray || S.name.name == :FieldArray
    T = eltype(S)
    return Vector{T}  # FillArrays, StaticArrays
  else
    return S  # BlockArrays, PartitionedArrays, etc...
  end
end

function ktypeof(v::S) where S <: SubArray
  vp = v.parent
  if isa(vp, DenseMatrix)
    M = typeof(vp)
    return matrix_to_vector(M)  # view of a row or a column of a matrix
  else
    return ktypeof(vp)  # view of a vector
  end
end

"""
    M = vector_to_matrix(S)

Return the dense matrix storage type `M` related to the dense vector storage type `S`.
"""
function vector_to_matrix(::Type{S}) where S <: DenseVector
  T = hasproperty(S, :body) ? S.body : S
  par = T.parameters
  npar = length(par)
  (2 ≤ npar ≤ 3) || error("Type $S is not supported.")
  if npar == 2
    M = T.name.wrapper{par[1], 2}
  else
    M = T.name.wrapper{par[1], 2, par[3]}
  end
  return M
end

"""
    S = matrix_to_vector(M)

Return the dense vector storage type `S` related to the dense matrix storage type `M`.
"""
function matrix_to_vector(::Type{M}) where M <: DenseMatrix
  T = hasproperty(M, :body) ? M.body : M
  par = T.parameters
  npar = length(par)
  (2 ≤ npar ≤ 3) || error("Type $M is not supported.")
  if npar == 2
    S = T.name.wrapper{par[1], 1}
  else
    S = T.name.wrapper{par[1], 1, par[3]}
  end
  return S
end

"""
    v = kzeros(S, n)

Create a vector of storage type `S` of length `n` only composed of zero.
"""
kzeros(S, n) = fill!(S(undef, n), zero(eltype(S)))

"""
    v = kones(S, n)

Create a vector of storage type `S` of length `n` only composed of one.
"""
kones(S, n) = fill!(S(undef, n), one(eltype(S)))

allocate_if(bool, solver, v, S, u) = bool && isempty(solver.:($v)::S) && (solver.:($v)::S = similar(u))
# allocate_if(bool, solver, v, S, n::Int) = bool && isempty(solver.:($v)::S) && (solver.:($v)::S = S(undef, n))
allocate_if(bool, solver, v, S, m::Int, n::Int) = bool && isempty(solver.:($v)::S) && (solver.:($v)::S = S(undef, m, n))

kdisplay(iter, verbose) = (verbose > 0) && (mod(iter, verbose) == 0)

ktimer(start_time::UInt64) = (time_ns() - start_time) / 1e9

mulorldiv!(y, P, x, ldiv::Bool) = ldiv ? ldiv!(y, P, x) : mul!(y, P, x)

kdot(n :: Integer, x :: Vector{T}, y :: Vector{T}) where T <: BLAS.BlasReal = BLAS.dot(n, x, 1, y, 1)
kdot(n :: Integer, x :: Vector{T}, y :: Vector{T}) where T <: BLAS.BlasComplex = BLAS.dotc(n, x, 1, y, 1)
kdot(n :: Integer, x :: AbstractVector{T}, y :: AbstractVector{T}) where T <: FloatOrComplex = dot(x, y)

kdotr(n :: Integer, x :: AbstractVector{T}, y :: AbstractVector{T}) where T <: AbstractFloat = kdot(n, x, y)
kdotr(n :: Integer, x :: AbstractVector{Complex{T}}, y :: AbstractVector{Complex{T}}) where T <: AbstractFloat = kdot(n, x, y) |> real

knorm(n :: Integer, x :: Vector{T}) where T <: BLAS.BlasFloat = BLAS.nrm2(n, x, 1)
knorm(n :: Integer, x :: AbstractVector{T}) where T <: FloatOrComplex = norm(x)

knorm_elliptic(n :: Integer, x :: AbstractVector{T}, y :: AbstractVector{T}) where T <: FloatOrComplex = (x === y) ? knorm(n, x) : kdotr(n, x, y) |> sqrt

kscal!(n :: Integer, s :: T, x :: Vector{T}) where T <: BLAS.BlasFloat = BLAS.scal!(n, s, x, 1)
kscal!(n :: Integer, s :: T, x :: AbstractVector{T}) where T <: FloatOrComplex = rmul!(x, s)
kscal!(n :: Integer, s :: T, x :: AbstractVector{Complex{T}}) where T <: AbstractFloat = kscal!(n, Complex{T}(s), x)

kaxpy!(n :: Integer, s :: T, x :: Vector{T}, y :: Vector{T}) where T <: BLAS.BlasFloat = BLAS.axpy!(n, s, x, 1, y, 1)
kaxpy!(n :: Integer, s :: T, x :: AbstractVector{T}, y :: AbstractVector{T}) where T <: FloatOrComplex = axpy!(s, x, y)
kaxpy!(n :: Integer, s :: T, x :: AbstractVector{Complex{T}}, y :: AbstractVector{Complex{T}}) where T <: AbstractFloat = kaxpy!(n, Complex{T}(s), x, y)

kaxpby!(n :: Integer, s :: T, x :: Vector{T}, t :: T, y :: Vector{T}) where T <: BLAS.BlasFloat = BLAS.axpby!(n, s, x, 1, t, y, 1)
kaxpby!(n :: Integer, s :: T, x :: AbstractVector{T}, t :: T, y :: AbstractVector{T}) where T <: FloatOrComplex = axpby!(s, x, t, y)
kaxpby!(n :: Integer, s :: T, x :: AbstractVector{Complex{T}}, t :: Complex{T}, y :: AbstractVector{Complex{T}}) where T <: AbstractFloat = kaxpby!(n, Complex{T}(s), x, t, y)
kaxpby!(n :: Integer, s :: Complex{T}, x :: AbstractVector{Complex{T}}, t :: T, y :: AbstractVector{Complex{T}}) where T <: AbstractFloat = kaxpby!(n, s, x, Complex{T}(t), y)
kaxpby!(n :: Integer, s :: T, x :: AbstractVector{Complex{T}}, t :: T, y :: AbstractVector{Complex{T}}) where T <: AbstractFloat = kaxpby!(n, Complex{T}(s), x, Complex{T}(t), y)

kcopy!(n :: Integer, y :: Vector{T}, x :: Vector{T}) where T <: BLAS.BlasFloat = BLAS.blascopy!(n, x, 1, y, 1)
kcopy!(n :: Integer, y :: AbstractVector, x :: AbstractVector) = copyto!(y, x)

kfill!(x :: AbstractArray{T}, val :: T) where T <: FloatOrComplex = fill!(x, val)

kref!(n, x, y, c, s) = reflect!(x, y, c, s)

kgeqrf!(A :: AbstractMatrix{T}, tau :: AbstractVector{T}) where T <: BLAS.BlasFloat = LAPACK.geqrf!(A, tau)
korgqr!(A :: AbstractMatrix{T}, tau :: AbstractVector{T}) where T <: BLAS.BlasFloat = LAPACK.orgqr!(A, tau)
kormqr!(side :: Char, trans :: Char, A :: AbstractMatrix{T}, tau :: AbstractVector{T}, C :: AbstractMatrix{T}) where T <: BLAS.BlasFloat = LAPACK.ormqr!(side, trans, A, tau, C)

macro kswap!(x, y)
  quote
    local tmp = $(esc(x))
    $(esc(x)) = $(esc(y))
    $(esc(y)) = tmp
  end
end

"""
    roots = to_boundary(n, x, d, radius; flip, xNorm2, dNorm2)

Given a trust-region radius `radius`, a vector `x` lying inside the
trust-region and a direction `d`, return `σ1` and `σ2` such that

    ‖x + σi d‖ = radius, i = 1, 2

in the Euclidean norm.
`n` is the length of vectors `x` and `d`.
If known, ‖x‖² and ‖d‖² may be supplied with `xNorm2` and `dNorm2`.

If `flip` is set to `true`, `σ1` and `σ2` are computed such that

    ‖x - σi d‖ = radius, i = 1, 2.
"""
function to_boundary(n :: Int, x :: AbstractVector{FC}, d :: AbstractVector{FC}, z::AbstractVector{FC}, radius :: T; flip :: Bool=false, xNorm2 :: T=zero(T), dNorm2 :: T=zero(T), M=I, ldiv :: Bool = false) where {T <: AbstractFloat, FC <: FloatOrComplex{T}}
  radius > 0 || error("radius must be positive")

  if M === I
    # ‖d‖² σ² + (xᴴd + dᴴx) σ + (‖x‖² - Δ²).
    rxd = kdotr(n, x, d)
    dNorm2 == zero(T) && (dNorm2 = kdotr(n, d, d))
    xNorm2 == zero(T) && (xNorm2 = kdotr(n, x, x))
  else
    # (dᴴMd) σ² + (xᴴMd + dᴴMx) σ + (xᴴMx - Δ²).
    mulorldiv!(z, M, x, ldiv)
    rxd = kdot(n, z, d)
    xNorm2 = kdotr(n, z, x)
    mulorldiv!(z, M, d, ldiv) 
    dNorm2 = kdotr(n, z, d)
  end 
  dNorm2 == zero(T) && error("zero direction")
  flip && (rxd = -rxd)
  
  radius2 = radius * radius
  (xNorm2 ≤ radius2) || error(@sprintf("outside of the trust region: ‖x‖²=%7.1e, Δ²=%7.1e", xNorm2, radius2))

  # q₂ = ‖d‖², q₁ = xᴴd + dᴴx, q₀ = ‖x‖² - Δ²
  # ‖x‖² ≤ Δ² ⟹ (q₁)² - 4 * q₂ * q₀ ≥ 0
  roots = roots_quadratic(dNorm2, 2 * real(rxd), xNorm2 - radius2)

  return roots  # `σ1` and `σ2`
end

"""
    arguments = extract_parameters(ex::Expr)

Extract the arguments of an expression that is keyword parameter tuple.
Implementation suggested by Mitchell J. O'Sullivan (@mosullivan93).
"""
function extract_parameters(ex::Expr)
  Meta.isexpr(ex, :tuple, 1) &&
  Meta.isexpr((@inbounds p = ex.args[1]), :parameters) &&
  Base.Docs.validcall(p.args[]) || throw(ArgumentError("Given expression is not a kw parameter tuple [e.g. :(; x)]: $ex"))
  return p.args[]
end
