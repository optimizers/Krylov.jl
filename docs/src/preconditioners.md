# [Preconditioners](@id preconditioners)

The solvers in Krylov.jl support preconditioners, i.e., transformations that modify a linear system $Ax = b$ into an equivalent form that may yield faster convergence in finite-precision arithmetic.
Preconditioning can be used to reduce the condition number of the problem or cluster its eigenvalues or singular values for instance.

The design of preconditioners is highly dependent on the origin of the problem and most preconditioners need to take application-dependent information and structure into account.
Specialized preconditioners generally outperform generic preconditioners such as incomplete factorizations.

The construction of a preconditioner necessitates trade-offs because we need to apply it at least once per iteration within a Krylov method.
Hence, a preconditioner must be constructed such that it is cheap to apply, while also capturing the characteristics of the original system in some sense.

There exist three variants of preconditioning:

| Left preconditioning               | Two-sided preconditioning                                              | Right preconditioning                        |
|:----------------------------------:|:----------------------------------------------------------------------:|:--------------------------------------------:|
| $P_{\ell}^{-1}Ax = P_{\ell}^{-1}b$ | $P_{\ell}^{-1}AP_r^{-1}y = P_{\ell}^{-1}b~~\text{with}~~x = P_r^{-1}y$ | $AP_r^{-1}y = b~~\text{with}~~x = P_r^{-1}y$ |

where $P_{\ell}$ and $P_r$ are square and nonsingular.

The left preconditioning preserves the error $x_k - x^{\star}$ whereas the right preconditioning keeps invariant the residual $b - A x_k$.
Two-sided preconditioning is the only variant that allows to preserve the hermicity of a linear system.

!!! note
    Because det$(P^{-1}A - \lambda I)$ = det$(A - \lambda P)$ det$(P^{-1})$ = det$(AP^{-1} - \lambda I)$, the eigenvalues of $P^{-1}A$ and $AP^{-1}$ are identical. If $P = LL^{H}$, $L^{-1}AL^{-H}$ also has the same spectrum.

In Krylov.jl, we call $P_{\ell}^{-1}$ and $P_r^{-1}$ the preconditioners and we assume that we can apply them with the operation $y \leftarrow P^{-1} * x$.
It is also common to call $P_{\ell}$ and $P_r$ the preconditioners if the equivalent operation $y \leftarrow P~\backslash~x$ is available.
Krylov.jl supports both approaches thanks to the argument `ldiv` of the Krylov solvers.

## How to use preconditioners in Krylov.jl?

!!! info
    - A preconditioner only needs to support the operation `mul!(y, P⁻¹, x)` when `ldiv=false` or `ldiv!(y, P, x)` when `ldiv=true` to be used in Krylov.jl.
    - Additional support for `adjoint` with preconditioners is required in the methods [`BILQ`](@ref bilq) and [`QMR`](@ref qmr).
    - The default value of a preconditioner in Krylov.jl is the identity operator `I`.

### Square non-Hermitian linear systems

Methods concerned: [`CGS`](@ref cgs), [`BILQ`](@ref bilq), [`QMR`](@ref qmr), [`BiCGSTAB`](@ref bicgstab), [`DQGMRES`](@ref dqgmres), [`GMRES`](@ref gmres), [`BLOCK-GMRES`](@ref block_gmres), [`FGMRES`](@ref fgmres), [`DIOM`](@ref diom) and [`FOM`](@ref fom).

A Krylov method dedicated to non-Hermitian linear systems allows the three variants of preconditioning.

| Preconditioners | $P_{\ell}^{-1}$       | $P_{\ell}$           | $P_r^{-1}$            | $P_r$                |
|:---------------:|:---------------------:|:--------------------:|:---------------------:|:--------------------:|
| Arguments       | `M` with `ldiv=false` | `M` with `ldiv=true` | `N` with `ldiv=false` | `N` with `ldiv=true` |

### Hermitian linear systems

Methods concerned: [`SYMMLQ`](@ref symmlq), [`CG`](@ref cg), [`CG-LANCZOS`](@ref cg_lanczos), [`CG-LANCZOS-SHIFT`](@ref cg_lanczos_shift), [`CR`](@ref cr), [`CAR`](@ref car), [`MINRES`](@ref minres), [`BLOCK-MINRES`](@ref block_minres), [`MINRES-QLP`](@ref minres_qlp) and [`MINARES`](@ref minares).

When $A$ is Hermitian, we can only use centered preconditioning $L^{-1}AL^{-H}y = L^{-1}b$ with $x = L^{-H}y$.
Centered preconditioning is a special case of two-sided preconditioning with $P_{\ell} = L = P_r^H$ that maintains hermicity.
However, there is no need to specify $L$ and one may specify $P_c = LL^H$ or its inverse directly.

| Preconditioners | $P_c^{-1}$                | $P_c$                |
|:---------------:|:-------------------------:|:--------------------:|
| Arguments       | `M` with `ldiv=false`     | `M` with `ldiv=true` |

!!! warning
    The preconditioner `M` must be hermitian and positive definite.

### Linear least-squares problems

Methods concerned: [`CGLS`](@ref cgls), [`CGLS-LANCZOS-SHIFT`](@ref cgls_lanczos_shift), [`CRLS`](@ref crls), [`LSLQ`](@ref lslq), [`LSQR`](@ref lsqr) and [`LSMR`](@ref lsmr).

| Formulation           | Without preconditioning              | With preconditioning                        |
|:---------------------:|:------------------------------------:|:-------------------------------------------:|
| least-squares problem | $\min \tfrac{1}{2} \\|b - Ax\\|^2_2$ | $\min \tfrac{1}{2} \\|b - Ax\\|^2_{E^{-1}}$ |
| Normal equation       | $A^HAx = A^Hb$                       | $A^HE^{-1}Ax = A^HE^{-1}b$                  |
| Augmented system      | $\begin{bmatrix} I & A \\ A^H & 0 \end{bmatrix} \begin{bmatrix} r \\ x \end{bmatrix} = \begin{bmatrix} b \\ 0 \end{bmatrix}$ | $\begin{bmatrix} E & A \\ A^H & 0 \end{bmatrix} \begin{bmatrix} r \\ x \end{bmatrix} = \begin{bmatrix} b \\ 0 \end{bmatrix}$ |

[`LSLQ`](@ref lslq), [`LSQR`](@ref lsqr) and [`LSMR`](@ref lsmr) also handle regularized least-squares problems.

| Formulation           | Without preconditioning                                                   | With preconditioning                                                             |
|:---------------------:|:-------------------------------------------------------------------------:|:--------------------------------------------------------------------------------:|
| least-squares problem | $\min \tfrac{1}{2} \\|b - Ax\\|^2_2 + \tfrac{1}{2} \lambda^2 \\|x\\|^2_2$ | $\min \tfrac{1}{2} \\|b - Ax\\|^2_{E^{-1}} + \tfrac{1}{2} \lambda^2 \\|x\\|^2_F$ |
| Normal equation       | $(A^HA + \lambda^2 I)x = A^Hb$                                            | $(A^HE^{-1}A + \lambda^2 F)x = A^HE^{-1}b$                                       |
| Augmented system      | $\begin{bmatrix} I & A \\ A^H & -\lambda^2 I \end{bmatrix} \begin{bmatrix} r \\ x \end{bmatrix} = \begin{bmatrix} b \\ 0 \end{bmatrix}$ | $\begin{bmatrix} E & A \\ A^H & -\lambda^2 F \end{bmatrix} \begin{bmatrix} r \\ x \end{bmatrix} = \begin{bmatrix} b \\ 0 \end{bmatrix}$ |

| Preconditioners | $E^{-1}$                | $E$                  | $F^{-1}$                | $F$                  |
|:---------------:|:-----------------------:|:--------------------:|:-----------------------:|:--------------------:|
| Arguments       | `M` with `ldiv=false`   | `M` with `ldiv=true` | `N` with `ldiv=false`   | `N` with `ldiv=true` |

!!! warning
    The preconditioners `M` and `N` must be hermitian and positive definite.

### Linear least-norm problems

Methods concerned: [`CGNE`](@ref cgne), [`CRMR`](@ref crmr), [`LNLQ`](@ref lnlq), [`CRAIG`](@ref craig) and [`CRAIGMR`](@ref craigmr).

| Formulation          | Without preconditioning                              | With preconditioning                                 |
|:--------------------:|:----------------------------------------------------:|:----------------------------------------------------:|
| minimum-norm problem | $\min \tfrac{1}{2} \\|x\\|^2_2~~\text{s.t.}~~Ax = b$ | $\min \tfrac{1}{2} \\|x\\|^2_F~~\text{s.t.}~~Ax = b$ |
| Normal equation      | $AA^Hy = b~~\text{with}~~x = A^Hy$                   | $AF^{-1}A^Hy = b~~\text{with}~~x = F^{-1}A^Hy$       |
| Augmented system     | $\begin{bmatrix} -I & A^H \\ \phantom{-}A & 0 \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} 0 \\ b \end{bmatrix}$ | $\begin{bmatrix} -F & A^H \\ \phantom{-}A & 0 \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} 0 \\ b \end{bmatrix}$ |

[`LNLQ`](@ref lslq), [`CRAIG`](@ref lsqr) and [`CRAIGMR`](@ref lsmr) also handle penalized minimum-norm problems.

| Formulation          | Without preconditioning                                                                       | With preconditioning                                                                           |
|:--------------------:|:---------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------:|
| minimum-norm problem | $\min \tfrac{1}{2} \\|x\\|^2_2 + \tfrac{1}{2} \\|y\\|^2_2~~\text{s.t.}~~Ax + \lambda^2 y = b$ | $\min \tfrac{1}{2} \\|x\\|^2_F + \tfrac{1}{2} \\|y\\|^2_E~~\text{s.t.}~~Ax + \lambda^2 Ey = b$ |
| Normal equation      | $(AA^H + \lambda^2 I)y = b~~\text{with}~~x = A^Hy$                                            | $(AF^{-1}A^H + \lambda^2 E)y = b~~\text{with}~~x = F^{-1}A^Hy$                                 |
| Augmented system     | $\begin{bmatrix} -I & A^H \\ \phantom{-}A & \lambda^2 I \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} 0 \\ b \end{bmatrix}$ | $\begin{bmatrix} -F & A^H \\ \phantom{-}A & \lambda^2 E \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} 0 \\ b \end{bmatrix}$ |

| Preconditioners | $E^{-1}$                | $E$                  | $F^{-1}$                | $F$                  |
|:---------------:|:-----------------------:|:--------------------:|:-----------------------:|:--------------------:|
| Arguments       | `M` with `ldiv=false`   | `M` with `ldiv=true` | `N` with `ldiv=false`   | `N` with `ldiv=true` |

!!! warning
    The preconditioners `M` and `N` must be hermitian and positive definite.

### Saddle-point and symmetric quasi-definite systems

[`TriCG`](@ref tricg), [`TriMR`](@ref trimr) and [`USYMLQR`](@ref usymlqr) can take advantage of the structure of Hermitian systems $Kz = d$ with the 2x2 block structure
```math
  \begin{bmatrix} \tau E & \phantom{-}A \\ A^H & \nu F \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} b \\ c \end{bmatrix},
```
| Preconditioners | $E^{-1}$              | $E$                  | $F^{-1}$              | $F$                  |
|:---------------:|:---------------------:|:--------------------:|:---------------------:|:--------------------:|
| Arguments       | `M` with `ldiv=false` | `M` with `ldiv=true` | `N` with `ldiv=false` | `N` with `ldiv=true` |

!!! warning
    The preconditioners `M` and `N` must be hermitian and positive definite.

### Generalized saddle-point and unsymmetric partitioned systems

[`GPMR`](@ref gpmr) can take advantage of the structure of general square systems $Kz = d$ with the 2x2 block structure
```math
  \begin{bmatrix} \lambda M & A \\ B & \mu N \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} b \\ c \end{bmatrix},
```
| Relations       | $CE = M^{-1}$                 | $EC = M$                     | $DF = N^{-1}$                 | $FD = N$                     |
|:---------------:|:-----------------------------:|:----------------------------:|:-----------------------------:|:----------------------------:|
| Arguments       | `C` and `E` with `ldiv=false` | `C` and `E` with `ldiv=true` | `D` and `F` with `ldiv=false` | `D` and `F` with `ldiv=true` |

!!! note
    Our implementations of [`BiLQ`](@ref bilq), [`QMR`](@ref qmr), [`BiLQR`](@ref bilqr), [`USYMLQ`](@ref usymlq), [`USYMQR`](@ref usymqr) and [`TriLQR`](@ref trilqr) don't support preconditioning.

## Packages that provide preconditioners

- [KrylovPreconditioners.jl](https://github.com/JuliaSmoothOptimizers/KrylovPreconditioners.jl) implements block-Jacobi, IC(0) and ILU(0) preconditioners.
- [ILUZero.jl](https://github.com/mcovalt/ILUZero.jl) is a Julia implementation of incomplete LU factorization with zero level of fill-in. 
- [LimitedLDLFactorizations.jl](https://github.com/JuliaSmoothOptimizers/LimitedLDLFactorizations.jl) for limited-memory LDLᵀ factorization of symmetric matrices.
- [AlgebraicMultigrid.jl](https://github.com/JuliaLinearAlgebra/AlgebraicMultigrid.jl) provides two algebraic multigrid (AMG) preconditioners.
- [RandomizedPreconditioners.jl](https://github.com/tjdiamandis/RandomizedPreconditioners.jl) uses randomized numerical linear algebra to construct approximate inverses of matrices.
- [BasicLU.jl](https://github.com/JuliaSmoothOptimizers/BasicLU.jl) uses a sparse LU factorization to compute a maximum volume basis that can be used as a preconditioner for least-norm and least-squares problems.

## Examples

```julia
using KrylovPreconditioners, Krylov

P⁻¹ = BlockJacobiPreconditioner(A)  # Block-Jacobi preconditioner
x, stats = gmres(A, b, M=P⁻¹)
```

```julia
using Krylov
n, m = size(A)
d = [A[i,i] ≠ 0 ? 1 / abs(A[i,i]) : 1 for i=1:n]  # Jacobi preconditioner
P⁻¹ = diagm(d)
x, stats = symmlq(A, b, M=P⁻¹)
```

```julia
using Krylov
n, m = size(A)
d = [1 / norm(A[:,i]) for i=1:m]  # diagonal preconditioner
P⁻¹ = diagm(d)
x, stats = minres(A, b, M=P⁻¹)
```

```julia
using KrylovPreconditioners, Krylov
Pℓ = ilu(A)
x, stats = gmres(A, b, M=Pℓ, ldiv=true)  # left preconditioning
```

```julia
using LimitedLDLFactorizations, Krylov
P = lldl(A)
P.D .= abs.(P.D)
x, stats = cg(A, b, M=P, ldiv=true)  # centered preconditioning
```

```julia
using ILUZero, Krylov
Pᵣ = ilu0(A)
x, stats = bicgstab(A, b, N=Pᵣ, ldiv=true)  # right preconditioning
```

```julia
using LDLFactorizations, Krylov

M = ldl(E)
N = ldl(F)

# [E   A] [x] = [b]
# [Aᴴ -F] [y]   [c]
x, y, stats = tricg(A, b, c, M=M, N=N, ldiv=true)
```

```julia
using RandomizedPreconditioners, Krylov

Â = NystromSketch(A, k, r)

P = NystromPreconditioner(Â, μ)
x, stats = cg(A + μ*I, b; M=P, ldiv=true)

P⁻¹ = NystromPreconditionerInverse(Â, μ)
x, stats = cg(A + μ*I, b; M=P⁻¹)
```

```julia
using SuiteSparse, Krylov
import LinearAlgebra.ldiv!

M = cholesky(E)

# ldiv! is not implemented for the sparse Cholesky factorization (SuiteSparse.CHOLMOD)
ldiv!(y::Vector{T}, F::SuiteSparse.CHOLMOD.Factor{T}, x::Vector{T}) where T = (y .= F \ x)

# [E  A] [x] = [b]
# [Aᴴ 0] [y]   [c]
x, y, stats = trimr(A, b, c, M=M, sp=true, ldiv=true)
```

```julia
using Krylov

C = lu(M)

# [M  A] [x] = [b]
# [B  0] [y]   [c]
x, y, stats = gpmr(A, B, b, c, C=C, gsp=true, ldiv=true)
```

```julia
import BasicLU
using LinearOperators, Krylov

# Least-squares problem
m, n = size(A)
Aᴴ = sparse(A')
basis, B = BasicLU.maxvolbasis(Aᴴ)
opA = LinearOperator(A)
B⁻ᴴ = LinearOperator(Float64, n, n, false, false, (y, v) -> (y .= v ; BasicLU.solve!(B, y, 'T')),
                                                  (y, v) -> (y .= v ; BasicLU.solve!(B, y, 'N')),
                                                  (y, v) -> (y .= v ; BasicLU.solve!(B, y, 'N')))

d, stats = lsmr(opA * B⁻ᴴ, b)  # min ‖AB⁻ᴴd - b‖₂
x = B⁻ᴴ * d                    # recover the solution of min ‖Ax - b‖₂

# Least-norm problem
m, n = size(A)
basis, B = maxvolbasis(A)
opA = LinearOperator(A)
B⁻¹ = LinearOperator(Float64, m, m, false, false, (y, v) -> (y .= v ; BasicLU.solve!(B, y, 'N')),
                                                  (y, v) -> (y .= v ; BasicLU.solve!(B, y, 'T')),
                                                  (y, v) -> (y .= v ; BasicLU.solve!(B, y, 'T')))

x, y, stats = craigmr(B⁻¹ * opA, B⁻¹ * b)  # min ‖x‖₂  s.t.  B⁻¹Ax = B⁻¹b
```
