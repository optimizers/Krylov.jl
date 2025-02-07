## Origin of Krylov methods

__Cayley-Hamilton theorem__:
If $A$ is a square matrix of size ``n`` and
```math
p(X) = \det(XI_n - A) = X^n + p_{n-1} X^{n-1} + \dots + p_1 X + p_0
```
is its characteristic polynomial, then
```math
p(A) = A^n + p_{n-1} A^{n-1} + \dots + p_1 A + p_0 I_n = 0_n.
```
If $A$ is nonsingular, $p_0 \ne 0$ and
```math
A^{-1} = -\dfrac{1}{p_0}(A^{n-1} + p_{n-1} A^{n-2} + \dots + p_1 I_n).
```
Thus,
```math
x^{\star} = A^{-1}b \implies x^{\star} \in \mathcal{K}_n(A, b) = \mathop{\mathrm{Span}} \{b, Ab, \dots, A^{n-1}b\}
```
where ``\mathcal{K}_n(A, b)`` is a *Krylov subspace*.

## Principle of Krylov methods

Krylov methods build iteratively a solution ``x_k \in \mathcal{K}_k(A,b)`` of ``Ax = b``.

A process is used to build a basis ``V_k`` of ``\mathcal{K}_k(A, b)``.
We have the Lanczos process for square symmetric matrices and the Arnoldi process for square unsymmetric matrices.
The projection of ``A`` into the Krylov subspace has a workable structure.
The projection is always a tridiagonal matrix with the Lanczos process and an upper Hessenberg matrix with the Arnoldi process regardless the structure of ``A``.

Iterates have the form ``x_k = V_k y_k`` where ``y_k \in \mathbb{R}^k`` is determined by solving a subproblem that uses the projection of ``A``.
Depending on the subproblem used, the iterates ``x_k`` have different properties, such as monotonically
decreasing the residual norms ``\|b - A x_k\|`` or error norms ``\|x_k - x^{\star}\|``.

When ``A`` is rectangular, we use the Golub-Kahan process to build orthogonal bases of ``\mathcal{K}_k(A^T A, A^T b)`` and ``\mathcal{K}_k(A A^T, b)``, and the normal equations, to solve linear least-squares and least-norm problems.

## Convergence of Krylov methods

Because the minimal polynomial of a matrix ``A`` is a polynomial of minimal degree ``m`` such that ``q(A) = 0``, it divides all polynomials such that ``r(A) = 0``.
It notably divides the characteristic polynomial of ``A`` and it has the same roots:
```math
A^{-1} = -\dfrac{1}{q_0}(A^{m-1} + q_{m-1} A^{m-2} + \dots + q_1 I_n)
```
and
```math
x^{\star} = A^{-1}b \implies x^{\star} \in \mathcal{K}_m(A, b).
```
To have ``m \ll n``, the number of distinct roots of ``p(A)`` must be small.
It means that the matrix ``A`` has only a few distinct eigenvalues.
