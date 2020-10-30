using Krylov, LinearOperators, IncompleteLU
using LinearAlgebra, Printf

include("../test/test_utils.jl")

A, b = polar_poisson()
# A, b = kron_unsymmetric()
n = length(b)
F = ilu(A, τ = 0.05)

@printf("nnz(ILU) / nnz(A): %7.1e\n", nnz(F) / nnz(A))

# Solve Ax = b with CGS and an incomplete LU factorization
# Remark: BICGSTAB can be used in the same way
yM = zeros(n)
yN = zeros(n)
yP = zeros(n)
opM = LinearOperator(Float64, n, n, false, false, y -> (yM .= y ; IncompleteLU.forward_substitution_without_diag!(F.L, yM)))
opN = LinearOperator(Float64, n, n, false, false, y -> (yN .= y ; IncompleteLU.transposed_backward_substitution!(F.U, yN)))
opP = LinearOperator(Float64, n, n, false, false, y -> ldiv!(yP, F, y))

# Without preconditioning
x, stats = cgs(A, b)
r = b - A * x
@printf("[Without preconditioning] Residual norm: %8.1e\n", norm(r))
@printf("[Without preconditioning] Number of iterations: %3d\n", length(stats.residuals) - 1)

# Split preconditioning
x, stats = cgs(A, b, M=opM, N=opN)
r = b - A * x
@printf("[Split preconditioning] Residual norm: %8.1e\n", norm(r))
@printf("[Split preconditioning] Number of iterations: %3d\n", length(stats.residuals) - 1)

# Left preconditioning
x, stats = cgs(A, b, M=opP)
r = b - A * x
@printf("[Left preconditioning] Residual norm: %8.1e\n", norm(r))
@printf("[Left preconditioning] Number of iterations: %3d\n", length(stats.residuals) - 1)

# Right preconditioning
x, stats = cgs(A, b, N=opP)
r = b - A * x
@printf("[Right preconditioning] Residual norm: %8.1e\n", norm(r))
@printf("[Right preconditioning] Number of iterations: %3d\n", length(stats.residuals) - 1)
