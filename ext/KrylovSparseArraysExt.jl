module KrylovSparseArraysExt

using Krylov, SparseArrays
using Krylov.LinearAlgebra
using Krylov.LinearAlgebra: NoPivot, LowerTriangular
using Krylov: FloatOrComplex, reduced_qr!, ktypeof, vector_to_matrix, knorm, kdot, kaxpy!, kdotr, kfill!

function Krylov.ktypeof(v::S) where S <: SparseVector
  T = eltype(S)
  return Vector{T}
end

function Krylov.ktypeof(v::S) where S <: AbstractSparseVector
  return S.types[2]  # return `CuVector` for a `CuSparseVector`
end

# Standard processes

"""
    V, β, T = hermitian_lanczos(A, b, k; allow_breakdown=false)

#### Input arguments

* `A`: a linear operator that models a Hermitian matrix of dimension `n`;
* `b`: a vector of length `n`;
* `k`: the number of iterations of the Hermitian Lanczos process.

#### Keyword argument

* `allow_breakdown`: specify whether to continue the process or raise an error when an exact breakdown occurs.

#### Output arguments

* `V`: a dense `n × (k+1)` matrix;
* `β`: a coefficient such that `βv₁ = b`;
* `T`: a sparse `(k+1) × k` tridiagonal matrix.

#### Reference

* C. Lanczos, [*An Iteration Method for the Solution of the Eigenvalue Problem of Linear Differential and Integral Operators*](https://doi.org/10.6028/jres.045.026), Journal of Research of the National Bureau of Standards, 45(4), pp. 225--280, 1950.
"""
function Krylov.hermitian_lanczos(A, b::AbstractVector{FC}, k::Int;
                           allow_breakdown::Bool=false) where FC <: FloatOrComplex
  m, n = size(A)
  R = real(FC)
  S = ktypeof(b)
  M = vector_to_matrix(S)

  colptr = zeros(Int, k+1)
  rowval = zeros(Int, 3k-1)
  nzval = zeros(R, 3k-1)

  colptr[1] = 1
  for i = 1:k
    pos = colptr[i]
    colptr[i+1] = 3i
    if i == 1
      rowval[pos] = i
      rowval[pos+1] = i+1
    else
      rowval[pos] = i-1
      rowval[pos+1] = i
      rowval[pos+2] = i+1
    end
  end

  β₁ = zero(R)
  V = M(undef, n, k+1)
  T = SparseMatrixCSC(k+1, k, colptr, rowval, nzval)

  pαᵢ = 1  # Position of αᵢ in the vector `nzval`
  for i = 1:k
    vᵢ = view(V,:,i)
    vᵢ₊₁ = q = view(V,:,i+1)
    if i == 1
      β₁ = knorm(n, b)
      if β₁ == 0
        !allow_breakdown && error("Exact breakdown β₁ == 0.")
        kfill!(vᵢ, zero(FC))
      else
        vᵢ .= b ./ β₁
      end
    end
    mul!(q, A, vᵢ)
    if i ≥ 2
      vᵢ₋₁ = view(V,:,i-1)
      βᵢ = nzval[pαᵢ-2]  # βᵢ = Tᵢ.ᵢ₋₁
      nzval[pαᵢ-1] = βᵢ  # Tᵢ₋₁.ᵢ = βᵢ
      kaxpy!(n, -βᵢ, vᵢ₋₁, q)
    end
    αᵢ = kdotr(n, vᵢ, q)
    nzval[pαᵢ] = αᵢ  # Tᵢ.ᵢ = αᵢ
    kaxpy!(n, -αᵢ, vᵢ, q)
    βᵢ₊₁ = knorm(n, q)
    if βᵢ₊₁ == 0
      !allow_breakdown && error("Exact breakdown βᵢ₊₁ == 0 at iteration i = $i.")
      kfill!(vᵢ₊₁, zero(FC))
    else
      vᵢ₊₁ .= q ./ βᵢ₊₁
    end
    nzval[pαᵢ+1] = βᵢ₊₁  # Tᵢ₊₁.ᵢ = βᵢ₊₁
    pαᵢ = pαᵢ + 3
  end
  return V, β₁, T
end

"""
    V, β, T, U, γᴴ, Tᴴ = nonhermitian_lanczos(A, b, c, k; allow_breakdown=false)

#### Input arguments

* `A`: a linear operator that models a square matrix of dimension `n`;
* `b`: a vector of length `n`;
* `c`: a vector of length `n`;
* `k`: the number of iterations of the non-Hermitian Lanczos process.

#### Keyword argument

* `allow_breakdown`: specify whether to continue the process or raise an error when an exact breakdown occurs.

#### Output arguments

* `V`: a dense `n × (k+1)` matrix;
* `β`: a coefficient such that `βv₁ = b`;
* `T`: a sparse `(k+1) × k` tridiagonal matrix;
* `U`: a dense `n × (k+1)` matrix;
* `γᴴ`: a coefficient such that `γᴴu₁ = c`;
* `Tᴴ`: a sparse `(k+1) × k` tridiagonal matrix.

#### Reference

* C. Lanczos, [*An Iteration Method for the Solution of the Eigenvalue Problem of Linear Differential and Integral Operators*](https://doi.org/10.6028/jres.045.026), Journal of Research of the National Bureau of Standards, 45(4), pp. 225--280, 1950.
"""
function Krylov.nonhermitian_lanczos(A, b::AbstractVector{FC}, c::AbstractVector{FC}, k::Int;
                              allow_breakdown::Bool=false) where FC <: FloatOrComplex
  m, n = size(A)
  Aᴴ = A'
  R = real(FC)
  S = ktypeof(b)
  M = vector_to_matrix(S)

  colptr = zeros(Int, k+1)
  rowval = zeros(Int, 3k-1)
  nzval_T = zeros(FC, 3k-1)
  nzval_Tᴴ = zeros(FC, 3k-1)

  colptr[1] = 1
  for i = 1:k
    pos = colptr[i]
    colptr[i+1] = 3i
    if i == 1
      rowval[pos] = i
      rowval[pos+1] = i+1
    else
      rowval[pos] = i-1
      rowval[pos+1] = i
      rowval[pos+2] = i+1
    end
  end

  β₁ = γ₁ᴴ = zero(R)
  V = M(undef, n, k+1)
  U = M(undef, n, k+1)
  T = SparseMatrixCSC(k+1, k, colptr, rowval, nzval_T)
  Tᴴ = SparseMatrixCSC(k+1, k, colptr, rowval, nzval_Tᴴ)

  pαᵢ = 1  # Position of αᵢ and ᾱᵢ in the vectors `nzval_T` and `nzval_Tᴴ`
  for i = 1:k
    vᵢ = view(V,:,i)
    uᵢ = view(U,:,i)
    vᵢ₊₁ = q = view(V,:,i+1)
    uᵢ₊₁ = p = view(U,:,i+1)
    if i == 1
      cᴴb = kdot(n, c, b)
      if cᴴb == 0
        !allow_breakdown && error("Exact breakdown β₁γ₁ == 0.")
        βᵢ₊₁ = zero(FC)
        γᵢ₊₁ = zero(FC)
        kfill!(vᵢ₊₁, zero(FC))
        kfill!(uᵢ₊₁, zero(FC))
      else
        β₁ = √(abs(cᴴb))
        γ₁ᴴ = conj(cᴴb / β₁)
        vᵢ .= b ./ β₁
        uᵢ .= c ./ γ₁ᴴ
      end
    end
    mul!(q, A , vᵢ)
    mul!(p, Aᴴ, uᵢ)
    if i ≥ 2
      vᵢ₋₁ = view(V,:,i-1)
      uᵢ₋₁ = view(U,:,i-1)
      βᵢ = nzval_T[pαᵢ-2]  # βᵢ = Tᵢ.ᵢ₋₁
      γᵢ = nzval_T[pαᵢ-1]  # γᵢ = Tᵢ₋₁.ᵢ
      kaxpy!(n, -     γᵢ , vᵢ₋₁, q)
      kaxpy!(n, -conj(βᵢ), uᵢ₋₁, p)
    end
    αᵢ = kdot(n, uᵢ, q)
    nzval_T[pαᵢ]  = αᵢ        # Tᵢ.ᵢ  = αᵢ
    nzval_Tᴴ[pαᵢ] = conj(αᵢ)  # Tᴴᵢ.ᵢ = ᾱᵢ
    kaxpy!(m, -     αᵢ , vᵢ, q)
    kaxpy!(n, -conj(αᵢ), uᵢ, p)
    pᴴq = kdot(n, p, q)
    if pᴴq == 0
      !allow_breakdown && error("Exact breakdown βᵢ₊₁γᵢ₊₁ == 0 at iteration i = $i.")
      βᵢ₊₁ = zero(FC)
      γᵢ₊₁ = zero(FC)
      kfill!(vᵢ₊₁, zero(FC))
      kfill!(uᵢ₊₁, zero(FC))
    else
      βᵢ₊₁ = √(abs(pᴴq))
      γᵢ₊₁ = pᴴq / βᵢ₊₁
      vᵢ₊₁ .= q ./ βᵢ₊₁
      uᵢ₊₁ .= p ./ conj(γᵢ₊₁)
    end
    nzval_T[pαᵢ+1]  = βᵢ₊₁        # Tᵢ₊₁.ᵢ  = βᵢ₊₁
    nzval_Tᴴ[pαᵢ+1] = conj(γᵢ₊₁)  # Tᴴᵢ₊₁.ᵢ = γ̄ᵢ₊₁
    if i ≤ k-1
      nzval_T[pαᵢ+2]  = γᵢ₊₁        # Tᵢ.ᵢ₊₁  = γᵢ₊₁
      nzval_Tᴴ[pαᵢ+2] = conj(βᵢ₊₁)  # Tᴴᵢ.ᵢ₊₁ = β̄ᵢ₊₁
    end
    pαᵢ = pαᵢ + 3
  end
  return V, β₁, T, U, γ₁ᴴ, Tᴴ
end

"""
    V, U, β, L = golub_kahan(A, b, k; allow_breakdown=false)

#### Input arguments

* `A`: a linear operator that models a matrix of dimension `m × n`;
* `b`: a vector of length `m`;
* `k`: the number of iterations of the Golub-Kahan process.

#### Keyword argument

* `allow_breakdown`: specify whether to continue the process or raise an error when an exact breakdown occurs.

#### Output arguments

* `V`: a dense `n × (k+1)` matrix;
* `U`: a dense `m × (k+1)` matrix;
* `β`: a coefficient such that `βu₁ = b`;
* `L`: a sparse `(k+1) × (k+1)` lower bidiagonal matrix.

#### References

* G. H. Golub and W. Kahan, [*Calculating the Singular Values and Pseudo-Inverse of a Matrix*](https://doi.org/10.1137/0702016), SIAM Journal on Numerical Analysis, 2(2), pp. 225--224, 1965.
* C. C. Paige, [*Bidiagonalization of Matrices and Solution of Linear Equations*](https://doi.org/10.1137/0711019), SIAM Journal on Numerical Analysis, 11(1), pp. 197--209, 1974.
"""
function Krylov.golub_kahan(A, b::AbstractVector{FC}, k::Int;
                     allow_breakdown::Bool=false) where FC <: FloatOrComplex
  m, n = size(A)
  R = real(FC)
  Aᴴ = A'
  S = ktypeof(b)
  M = vector_to_matrix(S)

  colptr = zeros(Int, k+2)
  rowval = zeros(Int, 2k+1)
  nzval = zeros(R, 2k+1)

  colptr[1] = 1
  for i = 1:k+1
    pos = colptr[i]
    if i ≤ k
      colptr[i+1] = pos + 2
      rowval[pos] = i
      rowval[pos+1] = i+1
    else
      colptr[i+1] = pos + 1
      rowval[pos] = i
    end
  end

  β₁ = zero(R)
  V = M(undef, n, k+1)
  U = M(undef, m, k+1)
  L = SparseMatrixCSC(k+1, k+1, colptr, rowval, nzval)

  pαᵢ = 1  # Position of αᵢ in the vector `nzval`
  for i = 1:k
    uᵢ = view(U,:,i)
    vᵢ = view(V,:,i)
    uᵢ₊₁ = q = view(U,:,i+1)
    vᵢ₊₁ = p = view(V,:,i+1)
    if i == 1
      wᵢ = vᵢ
      β₁ = knorm(m, b)
      if β₁ == 0
        !allow_breakdown && error("Exact breakdown β₁ == 0.")
        kfill!(uᵢ, zero(FC))
      else
        uᵢ .= b ./ β₁
      end
      mul!(wᵢ, Aᴴ, uᵢ)
      αᵢ = knorm(n, wᵢ)
      if αᵢ == 0
        !allow_breakdown && error("Exact breakdown α₁ == 0.")
        kfill!(vᵢ, zero(FC))
      else
        vᵢ .= wᵢ ./ αᵢ
      end
      nzval[pαᵢ] = αᵢ  # Lᵢ.ᵢ = αᵢ
    end
    mul!(q, A, vᵢ)
    αᵢ = nzval[pαᵢ]  # αᵢ = Lᵢ.ᵢ
    kaxpy!(m, -αᵢ, uᵢ, q)
    βᵢ₊₁ = knorm(m, q)
    if βᵢ₊₁ == 0
      !allow_breakdown && error("Exact breakdown βᵢ₊₁ == 0 at iteration i = $i.")
      kfill!(uᵢ₊₁, zero(FC))
    else
      uᵢ₊₁ .= q ./ βᵢ₊₁
    end
    mul!(p, Aᴴ, uᵢ₊₁)
    kaxpy!(n, -βᵢ₊₁, vᵢ, p)
    αᵢ₊₁ = knorm(n, p)
    if αᵢ₊₁ == 0
      !allow_breakdown && error("Exact breakdown αᵢ₊₁ == 0 at iteration i = $i.")
      kfill!(vᵢ₊₁, zero(FC))
    else
      vᵢ₊₁ .= p ./ αᵢ₊₁
    end
    nzval[pαᵢ+1] = βᵢ₊₁  # Lᵢ₊₁.ᵢ   = βᵢ₊₁
    nzval[pαᵢ+2] = αᵢ₊₁  # Lᵢ₊₁.ᵢ₊₁ = αᵢ₊₁
    pαᵢ = pαᵢ + 2
  end
  return V, U, β₁, L
end

"""
    V, β, T, U, γᴴ, Tᴴ = saunders_simon_yip(A, b, c, k; allow_breakdown=false)

#### Input arguments

* `A`: a linear operator that models a matrix of dimension `m × n`;
* `b`: a vector of length `m`;
* `c`: a vector of length `n`;
* `k`: the number of iterations of the Saunders-Simon-Yip process.

#### Keyword argument

* `allow_breakdown`: specify whether to continue the process or raise an error when an exact breakdown occurs.

#### Output arguments

* `V`: a dense `m × (k+1)` matrix;
* `β`: a coefficient such that `βv₁ = b`;
* `T`: a sparse `(k+1) × k` tridiagonal matrix;
* `U`: a dense `n × (k+1)` matrix;
* `γᴴ`: a coefficient such that `γᴴu₁ = c`;
* `Tᴴ`: a sparse `(k+1) × k` tridiagonal matrix.

#### Reference

* M. A. Saunders, H. D. Simon, and E. L. Yip, [*Two Conjugate-Gradient-Type Methods for Unsymmetric Linear Equations*](https://doi.org/10.1137/0725052), SIAM Journal on Numerical Analysis, 25(4), pp. 927--940, 1988.
"""
function Krylov.saunders_simon_yip(A, b::AbstractVector{FC}, c::AbstractVector{FC}, k::Int;
                            allow_breakdown::Bool=false) where FC <: FloatOrComplex
  m, n = size(A)
  Aᴴ = A'
  R = real(FC)
  S = ktypeof(b)
  M = vector_to_matrix(S)

  colptr = zeros(Int, k+1)
  rowval = zeros(Int, 3k-1)
  nzval_T = zeros(FC, 3k-1)
  nzval_Tᴴ = zeros(FC, 3k-1)

  colptr[1] = 1
  for i = 1:k
    pos = colptr[i]
    colptr[i+1] = 3i
    if i == 1
      rowval[pos] = i
      rowval[pos+1] = i+1
    else
      rowval[pos] = i-1
      rowval[pos+1] = i
      rowval[pos+2] = i+1
    end
  end

  β₁ = γ₁ᴴ = zero(R)
  V = M(undef, m, k+1)
  U = M(undef, n, k+1)
  T = SparseMatrixCSC(k+1, k, colptr, rowval, nzval_T)
  Tᴴ = SparseMatrixCSC(k+1, k, colptr, rowval, nzval_Tᴴ)

  pαᵢ = 1  # Position of αᵢ and ᾱᵢ in the vectors `nzval_T` and `nzval_Tᴴ`
  for i = 1:k
    vᵢ = view(V,:,i)
    uᵢ = view(U,:,i)
    vᵢ₊₁ = q = view(V,:,i+1)
    uᵢ₊₁ = p = view(U,:,i+1)
    if i == 1
      β₁ = knorm(m, b)
      if β₁ == 0
        !allow_breakdown && error("Exact breakdown β₁ == 0.")
        kfill!(vᵢ, zero(FC))
      else
        vᵢ .= b ./ β₁
      end
      γ₁ᴴ = knorm(n, c)
      if γ₁ᴴ == 0
        !allow_breakdown && error("Exact breakdown γ₁ᴴ == 0.")
        kfill!(uᵢ, zero(FC))
      else
        uᵢ .= c ./ γ₁ᴴ
      end
    end
    mul!(q, A , uᵢ)
    mul!(p, Aᴴ, vᵢ)
    if i ≥ 2
      vᵢ₋₁ = view(V,:,i-1)
      uᵢ₋₁ = view(U,:,i-1)
      βᵢ = nzval_T[pαᵢ-2]  # βᵢ = Tᵢ.ᵢ₋₁
      γᵢ = nzval_T[pαᵢ-1]  # γᵢ = Tᵢ₋₁.ᵢ
      kaxpy!(m, -γᵢ, vᵢ₋₁, q)
      kaxpy!(n, -βᵢ, uᵢ₋₁, p)
    end
    αᵢ = kdot(m, vᵢ, q)
    nzval_T[pαᵢ]  = αᵢ        # Tᵢ.ᵢ  = αᵢ
    nzval_Tᴴ[pαᵢ] = conj(αᵢ)  # Tᴴᵢ.ᵢ = ᾱᵢ
    kaxpy!(m, -     αᵢ , vᵢ, q)
    kaxpy!(n, -conj(αᵢ), uᵢ, p)
    βᵢ₊₁ = knorm(m, q)
    if βᵢ₊₁ == 0
      !allow_breakdown && error("Exact breakdown βᵢ₊₁ == 0 at iteration i = $i.")
      kfill!(vᵢ₊₁, zero(FC))
    else
      vᵢ₊₁ .= q ./ βᵢ₊₁
    end
    γᵢ₊₁ = knorm(n, p)
    if γᵢ₊₁ == 0
      !allow_breakdown && error("Exact breakdown γᵢ₊₁ == 0 at iteration i = $i.")
      kfill!(uᵢ₊₁, zero(FC))
    else
      uᵢ₊₁ .= p ./ γᵢ₊₁
    end
    nzval_T[pαᵢ+1]  = βᵢ₊₁  # Tᵢ₊₁.ᵢ  = βᵢ₊₁
    nzval_Tᴴ[pαᵢ+1] = γᵢ₊₁  # Tᴴᵢ₊₁.ᵢ = γᵢ₊₁
    if i ≤ k-1
      nzval_T[pαᵢ+2]  = γᵢ₊₁  # Tᵢ.ᵢ₊₁  = γᵢ₊₁
      nzval_Tᴴ[pαᵢ+2] = βᵢ₊₁  # Tᴴᵢ.ᵢ₊₁ = βᵢ₊₁
    end
    pαᵢ = pαᵢ + 3
  end
  return V, β₁, T, U, γ₁ᴴ, Tᴴ
end

# Block processes

"""
    V, Ψ, T = hermitian_lanczos(A, B, k; algo="householder")

#### Input arguments

* `A`: a linear operator that models a Hermitian matrix of dimension `n`;
* `B`: a matrix of size `n × p`;
* `k`: the number of iterations of the block Hermitian Lanczos process.

#### Keyword arguments

* `algo`: the algorithm to perform reduced QR factorizations (`"gs"`, `"mgs"`, `"givens"` or `"householder"`).

#### Output arguments

* `V`: a dense `n × p(k+1)` matrix;
* `Ψ`: a dense `p × p` upper triangular matrix such that `V₁Ψ = B`;
* `T`: a sparse `p(k+1) × pk` block tridiagonal matrix with a bandwidth `p`.
"""
function Krylov.hermitian_lanczos(A, B::AbstractMatrix{FC}, k::Int; algo::String="householder") where FC <: FloatOrComplex
  m, n = size(A)
  t, p = size(B)

  nnzT = p*p + (k-1)*p*(2*p+1) + div(p*(p+1), 2)
  colptr = zeros(Int, p*k+1)
  rowval = zeros(Int, nnzT)
  nzval = zeros(FC, nnzT)

  colptr[1] = 1
  for j = 1:k*p
    pos = colptr[j]
    for i = max(1, j-p):j+p
      rowval[pos] = i
      pos += 1
    end
    colptr[j+1] = pos
  end

  V = zeros(FC, n, (k+1)*p)
  T = SparseMatrixCSC((k+1)*p, k*p, colptr, rowval, nzval)

  α = -one(FC)
  β = one(FC)
  q = zeros(FC, n, p)
  ψ₁ = zeros(FC, p, p)
  Ωᵢ = Ψᵢ = Ψᵢ₊₁ = zeros(FC, p, p)

  for i = 1:k
    pos1 = (i-1)*p + 1
    pos2 = i*p
    pos3 = pos1 + p
    pos4 = pos2 + p
    vᵢ = view(V,:,pos1:pos2)
    vᵢ₊₁ = view(V,:,pos3:pos4)

    if i == 1
      q .= B
      reduced_qr!(q, ψ₁, algo)
      vᵢ .= q
    end

    mul!(q, A, vᵢ)

    if i ≥ 2
      pos5 = pos1 - p
      pos6 = pos2 - p
      vᵢ₋₁ = view(V,:,pos5:pos6)
      mul!(q, vᵢ₋₁, Ψᵢ', α, β)  # q = q - vᵢ₋₁ * Ψᵢᴴ
    end

    mul!(Ωᵢ, vᵢ', q)       # Ωᵢ = vᵢᴴ * q
    mul!(q, vᵢ, Ωᵢ, α, β)  # q = q - vᵢ * Ωᵢᴴ

    # Store the block Ωᵢ in Tₖ₊₁.ₖ
    for ii = 1:p
      indi = pos1+ii-1
      for jj = 1:p
        indj = pos1+jj-1
        T[indi,indj] = Ωᵢ[ii,jj]
      end
    end

    reduced_qr!(q, Ψᵢ₊₁, algo)
    vᵢ₊₁ .= q

    # Store the block Ψᵢ₊₁ in Tₖ₊₁.ₖ
    for ii = 1:p
      indi = pos3+ii-1
      for jj = 1:p
        indj = pos1+jj-1
        (ii ≤ jj) && (T[indi,indj] = Ψᵢ₊₁[ii,jj])
        (ii ≤ jj) && (i < k) && (T[indj,indi] = conj(Ψᵢ₊₁[ii,jj]))
      end
    end
  end
  return V, ψ₁, T
end

"""
    V, Ψ, T, U, Φᴴ, Tᴴ = nonhermitian_lanczos(A, B, C, k)

#### Input arguments

* `A`: a linear operator that models a square matrix of dimension `n`;
* `B`: a matrix of size `n × p`;
* `C`: a matrix of size `n × p`;
* `k`: the number of iterations of the block non-Hermitian Lanczos process.

#### Output arguments

* `V`: a dense `n × p(k+1)` matrix;
* `Ψ`: a dense `p × p` upper triangular matrix such that `V₁Ψ = B`;
* `T`: a sparse `p(k+1) × pk` block tridiagonal matrix with a bandwidth `p`;
* `U`: a dense `n × p(k+1)` matrix;
* `Φᴴ`: a dense `p × p` upper triangular matrix such that `U₁Φᴴ = C`;
* `Tᴴ`: a sparse `p(k+1) × pk` block tridiagonal matrix with a bandwidth `p`.
"""
function Krylov.nonhermitian_lanczos(A, B::AbstractMatrix{FC}, C::AbstractMatrix{FC}, k::Int) where FC <: FloatOrComplex
  m, n = size(A)
  t, p = size(B)
  Aᴴ = A'
  pivoting = VERSION < v"1.9" ? Val{false}() : NoPivot()

  nnzT = p*p + (k-1)*p*(2*p+1) + div(p*(p+1), 2)
  colptr = zeros(Int, p*k+1)
  rowval = zeros(Int, nnzT)
  nzval_T = zeros(FC, nnzT)
  nzval_Tᴴ = zeros(FC, nnzT)

  colptr[1] = 1
  for j = 1:k*p
    pos = colptr[j]
    for i = max(1, j-p):j+p
      rowval[pos] = i
      pos += 1
    end
    colptr[j+1] = pos
  end

  V = zeros(FC, n, (k+1)*p)
  U = zeros(FC, n, (k+1)*p)
  T = SparseMatrixCSC((k+1)*p, k*p, colptr, rowval, nzval_T)
  Tᴴ = SparseMatrixCSC((k+1)*p, k*p, colptr, rowval, nzval_Tᴴ)

  α = -one(FC)
  β = one(FC)
  qᵥ = zeros(FC, n, p)
  qᵤ = zeros(FC, n, p)
  D = Ωᵢ = zeros(FC, p, p)
  Ψ₁ = zeros(FC, p, p)
  Φ₁ᴴ = zeros(FC, p, p)

  local Φᵢ, Ψᵢ

  for i = 1:k
    pos1 = (i-1)*p + 1
    pos2 = i*p
    pos3 = pos1 + p
    pos4 = pos2 + p
    vᵢ = view(V,:,pos1:pos2)
    vᵢ₊₁ = view(V,:,pos3:pos4)
    uᵢ = view(U,:,pos1:pos2)
    uᵢ₊₁ = view(U,:,pos3:pos4)

    if i == 1
      mul!(D, C', B)  # D = Cᴴ * B
      F = lu(D, pivoting)
      Φᵢ, Ψᵢ = F.L, F.U   # Φᵢ = F.P' * Φᵢ with pivoting
      Ψ₁ .= Ψᵢ
      Φ₁ᴴ .= Φᵢ'
      # vᵢ .= (Ψᵢ' \ B')'
      # uᵢ .= (Φᵢ \ C')'
      ldiv!(vᵢ', UpperTriangular(Ψᵢ)', B')
      ldiv!(uᵢ', LowerTriangular(Φᵢ), C')
    end

    mul!(qᵥ, A, vᵢ)
    mul!(qᵤ, Aᴴ, uᵢ)

    if i ≥ 2
      pos5 = pos1 - p
      pos6 = pos2 - p
      vᵢ₋₁ = view(V,:,pos5:pos6)
      uᵢ₋₁ = view(U,:,pos5:pos6)

      mul!(qᵥ, vᵢ₋₁, Φᵢ, α, β)   # qᵥ = qᵥ - vᵢ₋₁ * Φᵢ
      mul!(qᵤ, uᵢ₋₁, Ψᵢ', α, β)  # qᵤ = qᵤ - uᵢ₋₁ * Ψᵢᴴ
    end

    mul!(Ωᵢ, uᵢ', qᵥ)
    mul!(qᵥ, vᵢ, Ωᵢ, α, β)   # qᵥ = qᵥ - vᵢ * Ωᵢ
    mul!(qᵤ, uᵢ, Ωᵢ', α, β)  # qᵤ = qᵤ - uᵢ * Ωᵢᴴ

    # Store the block Ωᵢ in Tₖ₊₁.ₖ
    for ii = 1:p
      indi = pos1+ii-1
      for jj = 1:p
        indj = pos1+jj-1
        T[indi,indj] = Ωᵢ[ii,jj]
        Tᴴ[indi,indj] = conj(Ωᵢ[jj,ii])
      end
    end

    mul!(D, qᵤ', qᵥ)  # D = qᵤᴴ * qᵥ
    F = lu(D, pivoting)
    Φᵢ₊₁, Ψᵢ₊₁ = F.L, F.U  # Φᵢ₊₁ = F.P' * Φᵢ₊₁ with pivoting
    # vᵢ₊₁ .= (Ψᵢ₊₁' \ qᵥ')'
    # uᵢ₊₁ .= (Φᵢ₊₁ \ qᵤ')'
    ldiv!(vᵢ₊₁', UpperTriangular(Ψᵢ₊₁)', qᵥ')
    ldiv!(uᵢ₊₁', LowerTriangular(Φᵢ₊₁), qᵤ')
    Φᵢ = Φᵢ₊₁
    Ψᵢ = Ψᵢ₊₁

    # Store the blocks Ψᵢ₊₁ and Φᵢ₊₁ in Tₖ₊₁.ₖ
    for ii = 1:p
      indi = pos3+ii-1
      for jj = 1:p
        indj = pos1+jj-1
        (ii ≤ jj) && (T[indi,indj] = Ψᵢ₊₁[ii,jj])
        (ii ≤ jj) && (Tᴴ[indi,indj] = conj(Φᵢ₊₁[jj,ii]))
        (ii ≤ jj) && (i < k) && (Tᴴ[indj,indi] = conj(Ψᵢ₊₁[ii,jj]))
        (ii ≤ jj) && (i < k) && (T[indj,indi] = Φᵢ₊₁[jj,ii])
      end
    end
  end
  return V, Ψ₁, T, U, Φ₁ᴴ, Tᴴ
end

"""
    V, U, Ψ, L = golub_kahan(A, B, k; algo="householder")

#### Input arguments

* `A`: a linear operator that models a matrix of dimension `m × n`;
* `B`: a matrix of size `m × p`;
* `k`: the number of iterations of the block Golub-Kahan process.

#### Keyword argument

* `algo`: the algorithm to perform reduced QR factorizations (`"gs"`, `"mgs"`, `"givens"` or `"householder"`).

#### Output arguments

* `V`: a dense `n × p(k+1)` matrix;
* `U`: a dense `m × p(k+1)` matrix;
* `Ψ`: a dense `p × p` upper triangular matrix such that `U₁Ψ = B`;
* `L`: a sparse `p(k+1) × p(k+1)` block lower bidiagonal matrix with a lower bandwidth `p`.
"""
function Krylov.golub_kahan(A, B::AbstractMatrix{FC}, k::Int; algo::String="householder") where FC <: FloatOrComplex
  m, n = size(A)
  t, p = size(B)
  Aᴴ = A'

  nnzL = p*k*(p+1) + div(p*(p+1), 2)
  colptr = zeros(Int, p*(k+1)+1)
  rowval = zeros(Int, nnzL)
  nzval = zeros(FC, nnzL)

  colptr[1] = 1
  for j = 1:(k+1)*p
    pos = colptr[j]
    for i = j:min((k+1)*p,j+p)
      rowval[pos] = i
      pos += 1
    end
    colptr[j+1] = pos
  end

  V = zeros(FC, n, (k+1)*p)
  U = zeros(FC, m, (k+1)*p)
  L = SparseMatrixCSC((k+1)*p, (k+1)*p, colptr, rowval, nzval)

  α = -one(FC)
  β = one(FC)
  qᵥ = zeros(FC, n, p)
  qᵤ = zeros(FC, m, p)
  Ψ₁ = zeros(FC, p, p)
  Ψᵢ₊₁ = TΩᵢ = TΩᵢ₊₁ = zeros(FC, p, p)

  for i = 1:k
    pos1 = (i-1)*p + 1
    pos2 = i*p
    pos3 = pos1 + p
    pos4 = pos2 + p
    vᵢ = view(V,:,pos1:pos2)
    vᵢ₊₁ = view(V,:,pos3:pos4)
    uᵢ = view(U,:,pos1:pos2)
    uᵢ₊₁ = view(U,:,pos3:pos4)

    if i == 1
      qᵤ .= B
      reduced_qr!(qᵤ, Ψ₁, algo)
      uᵢ .= qᵤ

      mul!(qᵥ, Aᴴ, uᵢ)
      reduced_qr!(qᵥ, TΩᵢ, algo)
      vᵢ .= qᵥ

      # Store the block Ω₁ in Lₖ₊₁.ₖ₊₁
      for ii = 1:p
        indi = pos1+ii-1
        for jj = 1:p
          indj = pos1+jj-1
          (ii ≤ jj) && (L[indj,indi] = conj(TΩᵢ[ii,jj]))
        end
      end
    end

    mul!(qᵤ, A, vᵢ)
    mul!(qᵤ, uᵢ, TΩᵢ', α, β)  # qᵤ = qᵤ - uᵢ * Ωᵢ

    reduced_qr!(qᵤ, Ψᵢ₊₁, algo)
    uᵢ₊₁ .= qᵤ

    # Store the block Ψᵢ₊₁ in Lₖ₊₁.ₖ₊₁
    for ii = 1:p
      indi = pos3+ii-1
      for jj = 1:p
        indj = pos1+jj-1
        (ii ≤ jj) && (L[indi,indj] = Ψᵢ₊₁[ii,jj])
      end
    end

    mul!(qᵥ, Aᴴ, uᵢ₊₁)
    mul!(qᵥ, vᵢ, Ψᵢ₊₁', α, β)  # qᵥ = qᵥ - vᵢ * Ψᵢ₊₁ᴴ

    reduced_qr!(qᵥ, TΩᵢ₊₁, algo)
    vᵢ₊₁ .= qᵥ

    # Store the block Ωᵢ₊₁ in Lₖ₊₁.ₖ₊₁
    for ii = 1:p
      indi = pos3+ii-1
      for jj = 1:p
        indj = pos3+jj-1
        (ii ≤ jj) && (L[indj,indi] = conj(TΩᵢ₊₁[ii,jj]))
      end
    end
  end
  return V, U, Ψ₁, L
end

"""
    V, Ψ, T, U, Φᴴ, Tᴴ = saunders_simon_yip(A, B, C, k; algo="householder")

#### Input arguments

* `A`: a linear operator that models a matrix of dimension `m × n`;
* `B`: a matrix of size `m × p`;
* `C`: a matrix of size `n × p`;
* `k`: the number of iterations of the block Saunders-Simon-Yip process.

#### Keyword argument

* `algo`: the algorithm to perform reduced QR factorizations (`"gs"`, `"mgs"`, `"givens"` or `"householder"`).

#### Output arguments

* `V`: a dense `m × p(k+1)` matrix;
* `Ψ`: a dense `p × p` upper triangular matrix such that `V₁Ψ = B`;
* `T`: a sparse `p(k+1) × pk` block tridiagonal matrix with a bandwidth `p`;
* `U`: a dense `n × p(k+1)` matrix;
* `Φᴴ`: a dense `p × p` upper triangular matrix such that `U₁Φᴴ = C`;
* `Tᴴ`: a sparse `p(k+1) × pk` block tridiagonal matrix with a bandwidth `p`.
"""
function Krylov.saunders_simon_yip(A, B::AbstractMatrix{FC}, C::AbstractMatrix{FC}, k::Int; algo::String="householder") where FC <: FloatOrComplex
  m, n = size(A)
  t, p = size(B)
  Aᴴ = A'

  nnzT = p*p + (k-1)*p*(2*p+1) + div(p*(p+1), 2)
  colptr = zeros(Int, p*k+1)
  rowval = zeros(Int, nnzT)
  nzval_T = zeros(FC, nnzT)
  nzval_Tᴴ = zeros(FC, nnzT)

  colptr[1] = 1
  for j = 1:k*p
    pos = colptr[j]
    for i = max(1, j-p):j+p
      rowval[pos] = i
      pos += 1
    end
    colptr[j+1] = pos
  end

  V = zeros(FC, m, (k+1)*p)
  U = zeros(FC, n, (k+1)*p)
  T = SparseMatrixCSC((k+1)*p, k*p, colptr, rowval, nzval_T)
  Tᴴ = SparseMatrixCSC((k+1)*p, k*p, colptr, rowval, nzval_Tᴴ)

  α = -one(FC)
  β = one(FC)
  qᵥ = zeros(FC, m, p)
  qᵤ = zeros(FC, n, p)
  Ψ₁ = zeros(FC, p, p)
  Φ₁ᴴ = zeros(FC, p, p)
  Ωᵢ = Ψᵢ = Ψᵢ₊₁ = TΦᵢ = TΦᵢ₊₁ = zeros(FC, p, p)

  for i = 1:k
    pos1 = (i-1)*p + 1
    pos2 = i*p
    pos3 = pos1 + p
    pos4 = pos2 + p
    vᵢ = view(V,:,pos1:pos2)
    vᵢ₊₁ = view(V,:,pos3:pos4)
    uᵢ = view(U,:,pos1:pos2)
    uᵢ₊₁ = view(U,:,pos3:pos4)

    if i == 1
      qᵥ .= B
      reduced_qr!(qᵥ, Ψ₁, algo)
      vᵢ .= qᵥ
      qᵤ .= C
      reduced_qr!(qᵤ, Φ₁ᴴ, algo)
      uᵢ .= qᵤ
    end

    mul!(qᵥ, A, uᵢ)
    mul!(qᵤ, Aᴴ, vᵢ)

    if i ≥ 2
      pos5 = pos1 - p
      pos6 = pos2 - p
      vᵢ₋₁ = view(V,:,pos5:pos6)
      uᵢ₋₁ = view(U,:,pos5:pos6)

      mul!(qᵥ, vᵢ₋₁, TΦᵢ', α, β)  # qᵥ = qᵥ - vᵢ₋₁ * Φᵢ
      for ii = 1:p
        indi = pos1+ii-1
        for jj = 1:p
          indj = pos5+jj-1
          Ψᵢ[ii,jj] = T[indi,indj]
        end
      end
      mul!(qᵤ, uᵢ₋₁, Ψᵢ', α, β)  # qᵤ = qᵤ - uᵢ₋₁ * Ψᵢᴴ
    end

    mul!(Ωᵢ, vᵢ', qᵥ)
    mul!(qᵥ, vᵢ, Ωᵢ, α, β)   # qᵥ = qᵥ - vᵢ * Ωᵢ
    mul!(qᵤ, uᵢ, Ωᵢ', α, β)  # qᵤ = qᵤ - uᵢ * Ωᵢᴴ

    # Store the block Ωᵢ in Tₖ₊₁.ₖ
    for ii = 1:p
      indi = pos1+ii-1
      for jj = 1:p
        indj = pos1+jj-1
        T[indi,indj] = Ωᵢ[ii,jj]
        Tᴴ[indi,indj] = conj(Ωᵢ[jj,ii])
      end
    end

    reduced_qr!(qᵥ, Ψᵢ₊₁, algo)
    vᵢ₊₁ .= qᵥ

    # Store the block Ψᵢ₊₁ in Tₖ₊₁.ₖ
    for ii = 1:p
      indi = pos3+ii-1
      for jj = 1:p
        indj = pos1+jj-1
        (ii ≤ jj) && (T[indi,indj] = Ψᵢ₊₁[ii,jj])
        (ii ≤ jj) && (i < k) && (Tᴴ[indj,indi] = conj(Ψᵢ₊₁[ii,jj]))
      end
    end

    reduced_qr!(qᵤ, TΦᵢ₊₁, algo)
    uᵢ₊₁ .= qᵤ

    # Store the block Φᵢ₊₁ in Tₖ₊₁.ₖ
    for ii = 1:p
      indi = pos3+ii-1
      for jj = 1:p
        indj = pos1+jj-1
        (ii ≤ jj) && (Tᴴ[indi,indj] = TΦᵢ₊₁[ii,jj])
        (ii ≤ jj) && (i < k) && (T[indj,indi] = conj(TΦᵢ₊₁[ii,jj]))
      end
    end
  end
  return V, Ψ₁, T, U, Φ₁ᴴ, Tᴴ
end

end