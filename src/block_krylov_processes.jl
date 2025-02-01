"""
    V, Γ, H = arnoldi(A, B, k; algo="householder", reorthogonalization=false)

#### Input arguments

* `A`: a linear operator that models a square matrix of dimension `n`;
* `B`: a matrix of size `n × p`;
* `k`: the number of iterations of the block Arnoldi process.

#### Keyword arguments

* `algo`: the algorithm to perform reduced QR factorizations (`"gs"`, `"mgs"`, `"givens"` or `"householder"`).
* `reorthogonalization`: reorthogonalize the new matrices of the block Krylov basis against all previous matrices.

#### Output arguments

* `V`: a dense `n × p(k+1)` matrix;
* `Γ`: a dense `p × p` upper triangular matrix such that `V₁Γ = B`;
* `H`: a dense `p(k+1) × pk` block upper Hessenberg matrix with a lower bandwidth `p`.
"""
function arnoldi(A, B::AbstractMatrix{FC}, k::Int; algo::String="householder", reorthogonalization::Bool=false) where FC <: FloatOrComplex
  m, n = size(A)
  t, p = size(B)

  V = zeros(FC, n, (k+1)*p)
  H = zeros(FC, (k+1)*p, k*p)

  α = -one(FC)
  β = one(FC)
  q = zeros(FC, n, p)
  Γ = zeros(FC, p, p)
  Ψᵢⱼ = Ψtmp = zeros(FC, p, p)

  for j = 1:k
    pos1 = (j-1)*p + 1
    pos2 = j*p
    pos3 = pos1 + p
    pos4 = pos2 + p
    vⱼ = view(V,:,pos1:pos2)
    vⱼ₊₁ = view(V,:,pos3:pos4)

    if j == 1
      q .= B
      reduced_qr!(q, Γ, algo)
      vⱼ .= q
    end

    mul!(q, A, vⱼ)

    for i = 1:j
      pos5 = (i-1)*p + 1
      pos6 = i*p
      vᵢ = view(V,:,pos5:pos6)
      mul!(Ψᵢⱼ, vᵢ', q)       # Ψᵢⱼ = vᵢᴴ * q
      mul!(q, vᵢ, Ψᵢⱼ, α, β)  # q = q - vᵢ * Ψᵢⱼ
      H[pos5:pos6,pos1:pos2] .= Ψᵢⱼ
    end

    if reorthogonalization
      for i = 1:j
        pos5 = (i-1)*p + 1
        pos6 = i*p
        vᵢ = view(V,:,pos5:pos6)
        mul!(Ψtmp, vᵢ', q)       # Ψtmp = vᵢᴴ * q
        mul!(q, vᵢ, Ψtmp, α, β)  # q = q - vᵢ * Ψtmp
        Hᵢⱼ = view(H,pos5:pos6,pos1:pos2)
        Hᵢⱼ .= Hᵢⱼ .+ Ψtmp
      end
    end

    Ψ = view(H, pos3:pos4,pos1:pos2)
    reduced_qr!(q, Ψ, algo)
    vⱼ₊₁ .= q
  end
  return V, Γ, H
end

"""
    V, Γ, H, U, Λ, F = montoison_orban(A, B, D, C, k; algo="householder", reorthogonalization=false)

#### Input arguments

* `A`: a linear operator that models a matrix of dimension `m × n`;
* `B`: a linear operator that models a matrix of dimension `n × m`;
* `D`: a matrix of size `m × p`;
* `C`: a matrix of size `n × p`;
* `k`: the number of iterations of the block Montoison-Orban process.

#### Keyword arguments

* `algo`: the algorithm to perform reduced QR factorizations (`"gs"`, `"mgs"`, `"givens"` or `"householder"`).
* `reorthogonalization`: reorthogonalize the new matrices of the block Krylov basis against all previous matrices.

#### Output arguments

* `V`: a dense `m × p(k+1)` matrix;
* `Γ`: a dense `p × p` upper triangular matrix such that `V₁Γ = D`;
* `H`: a dense `p(k+1) × pk` block upper Hessenberg matrix with a lower bandwidth `p`;
* `U`: a dense `n × p(k+1)` matrix;
* `Λ`: a dense `p × p` upper triangular matrix such that `U₁Λ = C`;
* `F`: a dense `p(k+1) × pk` block upper Hessenberg matrix with a lower bandwidth `p`.
"""
function montoison_orban(A, B, D::AbstractMatrix{FC}, C::AbstractMatrix{FC}, k::Int; algo::String="householder", reorthogonalization::Bool=false) where FC <: FloatOrComplex
  m, n = size(A)
  t, p = size(D)

  V = zeros(FC, m, (k+1)*p)
  U = zeros(FC, n, (k+1)*p)
  H = zeros(FC, (k+1)*p, k*p)
  F = zeros(FC, (k+1)*p, k*p)

  α = -one(FC)
  β = one(FC)
  qᵥ = zeros(FC, m, p)
  qᵤ = zeros(FC, n, p)
  Γ = zeros(FC, p, p)
  Λ = zeros(FC, p, p)
  Ψᵢⱼ = Φᵢⱼ = Ψtmp = Φtmp = zeros(FC, p, p)

  for j = 1:k
    pos1 = (j-1)*p + 1
    pos2 = j*p
    pos3 = pos1 + p
    pos4 = pos2 + p
    vⱼ = view(V,:,pos1:pos2)
    vⱼ₊₁ = view(V,:,pos3:pos4)
    uⱼ = view(U,:,pos1:pos2)
    uⱼ₊₁ = view(U,:,pos3:pos4)

    if j == 1
      qᵥ .= D
      reduced_qr!(qᵥ, Γ, algo)
      vⱼ .= qᵥ

      qᵤ .= C
      reduced_qr!(qᵤ, Λ, algo)
      uⱼ .= qᵤ
    end

    mul!(qᵥ, A, uⱼ)
    mul!(qᵤ, B, vⱼ)

    for i = 1:j
      pos5 = (i-1)*p + 1
      pos6 = i*p
      vᵢ = view(V,:,pos5:pos6)
      uᵢ = view(U,:,pos5:pos6)

      mul!(Ψᵢⱼ, vᵢ', qᵥ)       # Ψᵢⱼ = vᵢᴴ * qᵥ
      mul!(qᵥ, vᵢ, Ψᵢⱼ, α, β)  # qᵥ = qᵥ - vᵢ * Ψᵢⱼ
      H[pos5:pos6,pos1:pos2] .= Ψᵢⱼ

      mul!(Φᵢⱼ, uᵢ', qᵤ)       # Φᵢⱼ = uᵢᴴ * qᵤ
      mul!(qᵤ, uᵢ, Φᵢⱼ, α, β)  # qᵤ = qᵤ - uᵢ * Φᵢⱼ
      F[pos5:pos6,pos1:pos2] .= Φᵢⱼ
    end

    if reorthogonalization
      for i = 1:j
        pos5 = (i-1)*p + 1
        pos6 = i*p
        vᵢ = view(V,:,pos5:pos6)
        uᵢ = view(U,:,pos5:pos6)

        mul!(Ψtmp, vᵢ', qᵥ)       # Ψtmp = vᵢᴴ * qᵥ
        mul!(qᵥ, vᵢ, Ψtmp, α, β)  # qᵥ = qᵥ - vᵢ * Ψtmp
        Hᵢⱼ = view(H,pos5:pos6,pos1:pos2)
        Hᵢⱼ .= Hᵢⱼ .+ Ψtmp

        mul!(Φtmp, uᵢ', qᵤ)       # Φtmp = uᵢᴴ * qᵤ
        mul!(qᵤ, uᵢ, Φtmp, α, β)  # qᵤ = qᵤ - uᵢ * Φtmp
        Fᵢⱼ = view(F,pos5:pos6,pos1:pos2)
        Fᵢⱼ .= Fᵢⱼ .+ Φtmp
      end
    end

    Ψ = view(H, pos3:pos4,pos1:pos2)
    reduced_qr!(qᵥ, Ψ, algo)
    vⱼ₊₁ .= qᵥ

    Φ = view(F, pos3:pos4,pos1:pos2)
    reduced_qr!(qᵤ, Φ, algo)
    uⱼ₊₁ .= qᵤ
  end
  return V, Γ, H, U, Λ, F
end
