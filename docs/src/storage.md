```@meta
# Thanks Morten Piibeleht for the hack with the tables!
```

```@raw html
<style>
.content table td {
    border-right-width: 1px;
}
.content table th {
    border-right-width: 1px;
}
.content table td:last-child {
    border-right-width: 0px;
}
.content table th:last-child {
    border-right-width: 0px;
}
html.theme--documenter-dark .content table td {
    border-right-width: 1px;
}
html.theme--documenter-dark .content table th {
    border-right-width: 1px;
}
html.theme--documenter-dark .content table td:last-child {
    border-right-width: 0px;
}
html.theme--documenter-dark .content table th:last-child {
    border-right-width: 0px;
}
</style>
```

# [Storage requirements](@id storage-requirements)

This section provides the storage requirements of all Krylov methods available in Krylov.jl.

### Notation

We denote by $m$ and $n$ the number of rows and columns of the linear problem.
The memory parameter of DIOM, FOM, DQGMRES, GMRES, FGMRES and GPMR is $k$.
The numbers of shifts of CG-LANCZOS-SHIFT and CGLS-LANCZOS-SHIFT is $p$.

## Theoretical storage requirements

The following tables provide the number of coefficients that must be allocated for each Krylov method.
The coefficients have the same type as those that compose the linear problem we seek to solve.
Each table summarizes the storage requirements of Krylov methods recommended to a specific linear problem.

#### Hermitian positive definite linear systems

| Methods | [`CG`](@ref cg) | [`CR`](@ref cr) | [`CAR`](@ref car) | [`CG-LANCZOS`](@ref cg_lanczos) | [`CG-LANCZOS-SHIFT`](@ref cg_lanczos_shift) |
|:-------:|:---------------:|:---------------:|:-----------------:|:-------------------------------:|:-------------------------------------------:|
 Storage  | $4n$            | $5n$            | $7n$              | $5n$                            | $3n + 2np + 5p$                             |

#### Hermitian indefinite linear systems

| Methods | [`SYMMLQ`](@ref symmlq) | [`MINRES`](@ref minres) | [`MINRES-QLP`](@ref minres_qlp) | [`MINARES`](@ref minares) |
|:-------:|:-----------------------:|:-----------------------:|:-------------------------------:|:-------------------------:|
| Storage | $5n$                    | $6n$                    | $6n$                            | $8n$                      |

#### Non-Hermitian square linear systems

| Methods | [`CGS`](@ref cgs) | [`BICGSTAB`](@ref bicgstab) | [`BiLQ`](@ref bilq) | [`QMR`](@ref qmr) |
|:-------:|:-----------------:|:---------------------------:|:-------------------:|:-----------------:|
| Storage | $6n$              | $6n$                        | $8n$                | $9n$              |

| Methods | [`DIOM`](@ref diom) | [`DQGMRES`](@ref dqgmres) |
|:-------:|:-------------------:|:-------------------------:|
| Storage | $n(2k+1) + 2k - 1$  | $n(2k+2) + 3k + 1$        |

| Methods | [`FOM`](@ref fom)                                  | [`GMRES`](@ref gmres)                   | [`FGMRES`](@ref fgmres)                  |
|:-------:|:--------------------------------------------------:|:---------------------------------------:|:----------------------------------------:|
| Storage$\dfrac{}{}$ | $\!n(2+k) +2k + \dfrac{k(k + 1)}{2}\!$ | $\!n(2+k) + 3k + \dfrac{k(k + 1)}{2}\!$ | $\!n(2+2k) + 3k + \dfrac{k(k + 1)}{2}\!$ |

#### Least-norm problems

| Methods | [`USYMLQ`](@ref usymlq) | [`CGNE`](@ref cgne) | [`CRMR`](@ref crmr) | [`LNLQ`](@ref lnlq) | [`CRAIG`](@ref craig) | [`CRAIGMR`](@ref craigmr) |
|:-------:|:-----------------------:|:-------------------:|:-------------------:|:-------------------:|:---------------------:|:-------------------------:|
| Storage | $5n + 3m$               | $3n + 2m$           | $3n + 2m$           | $3n + 4m$           | $3n + 4m$             | $4n + 5m$                 |

#### Least-squares problems

| Methods | [`USYMQR`](@ref usymqr) | [`CGLS`](@ref cgls) | [`CGLS-LANCZOS-SHIFT`](@ref cgls_lanczos_shift) | [`CRLS`](@ref crls) | [`LSLQ`](@ref lslq) | [`LSQR`](@ref lsqr) | [`LSMR`](@ref lsmr) |
|:-------:|:-----------------------:|:-------------------:|:-----------------------------------------------:|:-------------------:|:-------------------:|:-------------------:|:-------------------:|
| Storage | $6n + 3m$               | $3n + 2m$           | $3n + 2m + 5p + 2np$                            | $4n + 3m$           | $4n + 2m$           | $4n + 2m$           | $5n + 2m$           |

#### Adjoint systems

| Methods | [`BiLQR`](@ref bilqr) | [`TriLQR`](@ref trilqr) |
|:-------:|:---------------------:|:-----------------------:|
| Storage | $11n$                 | $6m + 5n$               |

#### Saddle-point and Hermitian quasi-definite systems

| Methods  | [`TriCG`](@ref tricg) | [`TriMR`](@ref trimr) | [`USYMLQR`](@ref usymlqr) |
|:--------:|:---------------------:|:---------------------:|:-------------------------:|
| Storage  | $6n + 6m$             | $8n + 8m$             | $7n + 6m$                 |

#### Generalized saddle-point and non-Hermitian partitioned systems

| Method  | [`GPMR`](@ref gpmr)       |
|:-------:|:-------------------------:|
| Storage | $(2+k)(n+m) + 2k^2 + 11k$ |

## Practical storage requirements

Each method has its own `KrylovSolver` that contains all the storage needed by the method.
In the REPL, the size in bytes of each attribute and the total amount of memory allocated by the solver are displayed when we show a `KrylovSolver`.

```@example storage
using Krylov

m = 5000
n = 12000
A = rand(Float64, m, n)
b = rand(Float64, m)
solver = LsmrSolver(A, b)
show(stdout, solver, show_stats=false)
```

If we want the total number of bytes used by the solver, we can call `nbytes = sizeof(solver)`.

```@example storage
nbytes = sizeof(solver)
```

Thereafter, we can use `Base.format_bytes(nbytes)` to recover what is displayed in the REPL.

```@example storage
Base.format_bytes(nbytes)
```

To verify that we match the theoretical results, we just need to multiply the storage requirement of a method by the number of bytes associated to the precision of the linear problem.
For instance, we need 4 bytes for the precision `Float32`, 8 bytes for precisions `Float64` and `ComplexF32`, and 16 bytes for the precision `ComplexF64`.

```@example storage
FC = Float64                            # precision of the least-squares problem
ncoefs_lsmr = 5*n + 2*m                 # number of coefficients
nbytes_lsmr = sizeof(FC) * ncoefs_lsmr  # number of bytes
```

Therefore, you can check that you have enough memory in RAM to allocate a `KrylovSolver`.

```@example storage
free_nbytes = Sys.free_memory()
Base.format_bytes(free_nbytes)  # Total free memory in RAM in bytes.
```

!!! note
    - Beyond having faster operations, using low precisions, such as simple precision, allows to store more coefficients in RAM and solve larger linear problems.
    - In the file [test_allocations.jl](https://github.com/JuliaSmoothOptimizers/Krylov.jl/blob/main/test/test_allocations.jl), we use the macro `@allocated` to test that we match the expected storage requirement of each method with a tolerance of 2%.
