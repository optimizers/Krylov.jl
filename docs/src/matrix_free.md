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

## [Matrix-free operators](@id matrix-free)

All methods are matrix-free, which means that you only need to provide operator-vector products.

The `A` or `B` input arguments of Krylov.jl solvers can be any object that represents a linear operator. That object must implement `mul!`, for multiplication with a vector, `size()` and `eltype()`. For certain methods it must also implement `adjoint()`.

Some methods only require `A * v` products, whereas other ones also require `A' * u` products. In the latter case, `adjoint(A)` must also be implemented.

| A * v                                           | A * v and A' * u                         |
|:-----------------------------------------------:|:----------------------------------------:|
| CG, CR, CAR                                     | CGLS, CRLS, CGNE, CRMR                   |
| SYMMLQ, CG-LANCZOS, MINRES, MINRES-QLP, MINARES | LSLQ, LSQR, LSMR, LNLQ, CRAIG, CRAIGMR   |
| DIOM, FOM, DQGMRES, GMRES, FGMRES, BLOCK-GMRES  | BiLQ, QMR, BiLQR, USYMLQ, USYMQR, TriLQR |
| CGS, BICGSTAB                                   | TriCG, TriMR, USYMLQR                    |
| CG-LANCZOS-SHIFT                                | CGLS-LANCZOS-SHIFT                       |

!!! info
    GPMR is the only method that requires `A * v` and `B * w` products.

Preconditioners `M`, `N`, `C`, `D`, `E` or `F` can be also linear operators and must implement `mul!` or `ldiv!`.

We strongly recommend [LinearOperators.jl](https://github.com/JuliaSmoothOptimizers/LinearOperators.jl) to model matrix-free operators, but other packages such as [LinearMaps.jl](https://github.com/JuliaLinearAlgebra/LinearMaps.jl), [DiffEqOperators.jl](https://github.com/SciML/DiffEqOperators.jl) or your own operator can be used as well.

With `LinearOperators.jl`, operators are defined as

```julia
A = LinearOperator(type, nrows, ncols, symmetric, hermitian, prod, tprod, ctprod)
```

where
* `type` is the operator element type;
* `nrow` and `ncol` are its dimensions;
* `symmetric` and `hermitian` should be set to `true` or `false`;
* `prod(y, v)`, `tprod(y, w)` and `ctprod(u, w)` are called when writing `mul!(y, A, v)`, `mul!(y, transpose(A), w)`, and `mul!(y, A', u)`, respectively.

See the [tutorial](https://jso.dev/tutorials/introduction-to-linear-operators/) and the detailed [documentation](https://jso.dev/LinearOperators.jl/dev/) for more information on `LinearOperators.jl`.

## Examples with automatic differentiation

In the field of nonlinear optimization, finding critical points of a continuous function frequently involves linear systems with a Hessian or Jacobian as coefficient. Materializing such operators as matrices is expensive in terms of operations and memory consumption and is unreasonable for high-dimensional problems. However, it is often possible to implement efficient Hessian-vector and Jacobian-vector products, for example with the help of automatic differentiation tools, and used within Krylov solvers. We now illustrate variants with explicit matrices and with matrix-free operators for two well-known optimization methods.

### Example 1: Newton's Method for convex optimization

At each iteration of Newton's method applied to a $\mathcal{C}^2$ strictly convex function $f : \mathbb{R}^n \rightarrow \mathbb{R}$, a descent direction direction is determined by minimizing the quadratic Taylor model of $f$:

```math
\min_{d \in \mathbb{R}^n}~~f(x_k) + \nabla f(x_k)^T d + \tfrac{1}{2}~d^T \nabla^2 f(x_k) d
```
which is equivalent to solving the symmetric and positive-definite system
```math
\nabla^2 f(x_k) d  = -\nabla f(x_k).
```
The system above can be solved with the conjugate gradient method as follows, using the explicit Hessian:

```@nlp
using ForwardDiff, Krylov

xk = -ones(4)

f(x) = (x[1] - 1)^2 + (x[2] - 2)^2 + (x[3] - 3)^2 + (x[4] - 4)^2

g(x) = ForwardDiff.gradient(f, x)

H(x) = ForwardDiff.hessian(f, x)

d, stats = cg(H(xk), -g(xk))
```

The explicit Hessian can be replaced by a linear operator that only computes Hessian-vector products:

```@example hessian_operator
using ForwardDiff, LinearOperators, Krylov

xk = -ones(4)

f(x) = (x[1] - 1)^2 + (x[2] - 2)^2 + (x[3] - 3)^2 + (x[4] - 4)^2

g(x) = ForwardDiff.gradient(f, x)

H(y, v) = ForwardDiff.derivative!(y, t -> g(xk + t * v), 0)
opH = LinearOperator(Float64, 4, 4, true, true, (y, v) -> H(y, v))

cg(opH, -g(xk))
```

### Example 2: The Gauss-Newton Method for Nonlinear Least Squares

At each iteration of the Gauss-Newton method applied to a nonlinear least-squares objective $f(x) = \tfrac{1}{2}\| F(x)\|^2$ where $F : \mathbb{R}^n \rightarrow \mathbb{R}^m$ is $\mathcal{C}^1$, we solve the subproblem:

```math
\min_{d \in \mathbb{R}^n}~~\tfrac{1}{2}~\|J(x_k) d + F(x_k)\|^2,
```
where $J(x)$ is the Jacobian of $F$ at $x$.

An appropriate iterative method to solve the above linear least-squares problems is LSMR. We could pass the explicit Jacobian to LSMR as follows:

```@nls
using ForwardDiff, Krylov

xk = ones(2)

F(x) = [x[1]^4 - 3; exp(x[2]) - 2; log(x[1]) - x[2]^2]

J(x) = ForwardDiff.jacobian(F, x)

d, stats = lsmr(J(xk), -F(xk))
```

However, the explicit Jacobian can be replaced by a linear operator that only computes Jacobian-vector and transposed Jacobian-vector products:

```@example jacobian_operator
using LinearAlgebra, ForwardDiff, LinearOperators, Krylov

xk = ones(2)

F(x) = [x[1]^4 - 3; exp(x[2]) - 2; log(x[1]) - x[2]^2]

J(y, v) = ForwardDiff.derivative!(y, t -> F(xk + t * v), 0)
Jᵀ(y, u) = ForwardDiff.gradient!(y, x -> dot(F(x), u), xk)
opJ = LinearOperator(Float64, 3, 2, false, false, (y, v) -> J(y, v),
                                                  (y, w) -> Jᵀ(y, w),
                                                  (y, u) -> Jᵀ(y, u))

lsmr(opJ, -F(xk))
```

## Example with FFT and IFFT

### Example 3: Solving the Poisson equation with FFT and IFFT

In applications related to partial differential equations (PDEs), linear systems can arise from discretizing differential operators.
Storing such operators as explicit matrices is computationally expensive and unnecessary when matrix-free methods can be used, particularly with structured grids.

The FFT is an algorithm that computes the discrete Fourier transform (DFT) of a sequence, transforming data from the spatial domain to the frequency domain.
In the context of solving PDEs, it simplifies the application of differential operators like the Laplacian by converting derivatives into algebraic operations.

For a function $u(x)$ discretized on a periodic grid with $n$ points, the FFT of $u$ is:

```math
\hat{u}_k = \sum_{j=0}^{n-1} u_j e^{-i k x_j},
```

where $\hat{u}_k$ represents the Fourier coefficients for the frequency $k$, and $u_j$ is the value of $u$ at the grid point $x_j$ defined as $x_j = \frac{2 \pi j}{L}$ with period $L$.
The inverse FFT (IFFT) reconstructs $u$ from its Fourier coefficients:

```math
u_j = \frac{1}{n} \sum_{k=0}^{n-1} \hat{u}_k e^{i k x_j}.
```

In Fourier space, the Laplacian operator $\frac{d^2}{dx^2}$ becomes a simple multiplication by $-k^2$, where $k$ is the wavenumber derived from the grid size.
This transforms the Poisson equation $\frac{d^2 u(x)}{dx^2} = f(x)$ into an algebraic equation in the frequency domain:

```math
-k^2 \hat{u}_k = \hat{f}_k.
```

By solving for $\hat{u}_k$ and applying the IFFT, we can recover the solution $u(x)$ efficiently.

The inverse FFT is used to convert data from the frequency domain back to the spatial domain.
Once the solution in frequency space is obtained by dividing the Fourier coefficients $\hat{f}_k$ by $-k^2$ for $k \neq 0$,
the IFFT is applied to transform the result back to the original grid points in the spatial domain.
At $k = 0$, the equation $-k^2 \hat{u}_0 = \hat{f}_0$ becomes indeterminate since $k^2 = 0$.
This situation corresponds to the zero-frequency component $\hat{f}_0$, which represents the mean of $f(x)$.
In such cases, $\hat{u}_0$ is treated separately.
It is typically set to 0 to remove the constant mode, or adjusted based on boundary conditions or other constraints.

In some cases, even though the FFT provides an efficient way to apply differential operators (such as the Laplacian)
in the frequency domain, a direct solution may not be feasible due to complex boundary conditions,
variable coefficients, or grid irregularities.
In these situations, the FFT must be coupled with a Krylov method to iteratively solve the problem.

This example consists of solving the 1D Poisson equation on a periodic domain $[0, 4\pi]$:

```math
\frac{d^2 u(x)}{dx^2} = f(x),
```

where $u(x)$ is the unknown solution, and $f(x)$ is the given source term.
We solve this equation using [FFTW.jl](https://github.com/JuliaMath/FFTW.jl) to compute the matrix-free action of the Laplacian within the conjugate gradient solver.

Note that while a direct FFT-based approach can be used here due to the simplicity of the periodic boundary conditions,
this example illustrates how a Krylov method can be employed to solve more challenging problems.

```@example fft_poisson
using FFTW, Krylov, LinearAlgebra

# Define the problem size and domain
n = 32768                         # Number of grid points (2^15)
L = 4π                            # Length of the domain
x = LinRange(0, L, n+1)[1:end-1]  # Periodic grid (excluding the last point)

# Define the source term f(x)
f = sin.(x)

# Define a matrix-free operator using FFT and IFFT
struct FFTPoissonOperator
    n::Int
    L::Float64
    complex::Bool
    k::Vector{Float64}  # Store Fourier wave numbers
end

function FFTPoissonOperator(n::Int, L::Float64, complex::Bool)
    if complex
        k = Vector{Float64}(undef, n)
    else
        k = Vector{Float64}(undef, n÷2 + 1)
    end
    k[1] = sum(f) / n  # average value of f(x) over the domain
    for j in 1:(n÷2)
        k[j+1] = 2 * π * j / L  # Positive wave numbers
    end
    if complex
        for j in 1:(n÷2 - 1)
            k[n-j+1] = -2 * π * j / L  # Negative wave numbers
        end
    end
    return FFTPoissonOperator(n, L, complex, k)
end

Base.size(A::FFTPoissonOperator) = (n, n)

function Base.eltype(A::FFTPoissonOperator)
    type = A.complex ? ComplexF64 : Float64
    return type
end

function LinearAlgebra.mul!(y::Vector, A::FFTPoissonOperator, u::Vector)
    # Transform the input vector `u` to the frequency domain using `fft` or `rfft`.
    # If the operator is complex, use the full FFT; otherwise, use the real FFT.
    if A.complex
        u_hat = fft(u)
    else
        u_hat = rfft(u)
    end

    # In Fourier space, solve the system by multiplying with -k^2 (corresponding to the second derivative).
    # This step applies the Laplacian operator in the frequency domain.
    u_hat .= -u_hat .* (A.k .^ 2)

    # Transform the result back to the spatial domain using `ifft` or `irfft`.
    # If the operator is complex, use the full inverse FFT; otherwise, use the inverse real FFT.
    if A.complex
        y .= ifft(u_hat)
    else
        y .= irfft(u_hat, A.n)
    end

    return y
end

# Create the matrix-free operator for the Poisson equation
complex = false
A = FFTPoissonOperator(n, L, complex)

# Solve the linear system using CG
u_sol, stats = cg(A, f, atol=1e-10, rtol=0.0, verbose=1)
```

```@example fft_poisson
# The exact solution is u(x) = -sin(x)
u_star = -sin.(x)
u_star ≈ u_sol
```

## Example with discretized PDE

### Example 4: Solving the 3D Helmholtz equation

The Helmholtz equation in 3D is a fundamental equation used in various fields like acoustics, electromagnetism, and quantum mechanics to model stationary wave phenomena.

The equation is given by:
```math
\nabla^2 u(x,y,z) + k^2 u(x,y,z) = f(x,y,z)
```
In this equation, $u(x, y, z)$ represents the unknown function, which could describe a pressure field in acoustics, a scalar potential in electromagnetism, or a wave function in quantum mechanics.
The operator $\nabla^2$ denotes the Laplacian in three dimensions. The wave number $k$ is related to the frequency of the wave through the equation $k = \frac{2\pi}{\lambda}$, where $\lambda$ is the wavelength.
Finally, $f(x,y,z)$ is a source term that drives the wave phenomena, acting as a forcing function or external influence.

To discretize the Helmholtz equation, we use finite differences on a uniform 3D grid with grid spacings $\Delta x$, $\Delta y$, and $\Delta z$.
For a grid point $(i, j, k)$, the second derivatives are approximated as follows:

- In the $x$-direction:
```math
\frac{\partial^2 u}{\partial x^2} \approx \frac{u_{i+1,j,k} - 2u_{i,j,k} + u_{i-1,j,k}}{\Delta x^2}
```
- In the $y$-direction:
```math
\frac{\partial^2 u}{\partial y^2} \approx \frac{u_{i,j+1,k} - 2u_{i,j,k} + u_{i,j-1,k}}{\Delta y^2}
```
- In the $z$-direction:
```math
\frac{\partial^2 u}{\partial z^2} \approx \frac{u_{i,j,k+1} - 2u_{i,j,k} + u_{i,j,k-1}}{\Delta z^2}
```
Combining these, the discretized Helmholtz equation becomes:
```math
\frac{u_{i+1,j,k} - 2u_{i,j,k} + u_{i-1,j,k}}{\Delta x^2} + \frac{u_{i,j+1,k} - 2u_{i,j,k} + u_{i,j-1,k}}{\Delta y^2} + \frac{u_{i,j,k+1} - 2u_{i,j,k} + u_{i,j,k-1}}{\Delta z^2} + k^2 u_{i,j,k} = f_{i,j,k}
```

This discretization results in an equation at each grid point, resulting in a large and sparse linear system when assembled across the entire 3D grid.
To simplify the example, we impose Dirichlet boundary conditions with the solution $u(x, y, z) = 0$ on the boundary of the cubic domain.

Explicitly constructing this large sparse matrix is often impractical and unnecessary.
Instead, we can define a function that directly applies the Helmholtz operator to the 3D grid, avoiding the need to form the matrix explicitly.

Krylov.jl operates on vectors, so we must vectorize both the solution and the computational domain.
However, we can still maintain the structure of the original 3D operator by using `reshape` and `vec`.
This approach enables a simpler and efficient application of the operator in 3D while leveraging the vectorized framework for linear algebra operations.

```@example helmholtz
using Krylov, LinearAlgebra

# Parameters
L = 1.0                 # Length of the cubic domain
Nx = 200                # Number of interior grid points in x
Ny = 200                # Number of interior grid points in y
Nz = 200                # Number of interior grid points in z
Δx = L / (Nx + 1)       # Grid spacing in x
Δy = L / (Ny + 1)       # Grid spacing in y
Δz = L / (Nz + 1)       # Grid spacing in z
wavelength = 0.5        # Wavelength of the wave
k = 2 * π / wavelength  # Wave number

# Create the grid points
x = 0:Δx:L  # Points in x dimension (Nx + 2)
y = 0:Δy:L  # Points in y dimension (Ny + 2)
z = 0:Δz:L  # Points in z dimension (Nz + 2)

# Define a matrix-free Helmholtz operator
struct HelmholtzOperator
    Nx::Int
    Ny::Int
    Nz::Int
    Δx::Float64
    Δy::Float64
    Δz::Float64
    k::Float64
end

Base.size(A::HelmholtzOperator) = (A.Nx * A.Ny * A.Nz, A.Nx * A.Ny * A.Nz)

Base.eltype(A::HelmholtzOperator) = Float64

function LinearAlgebra.mul!(y::Vector, A::HelmholtzOperator, u::Vector)
    # Reshape vectors y and u into 3D arrays
    U = reshape(u, A.Nx, A.Ny, A.Nz)
    Y = reshape(y, A.Nx, A.Ny, A.Nz)

    # Apply the discrete Laplacian in 3D with k^2 * u
    for i in 1:A.Nx
        for j in 1:A.Ny
            for k in 1:A.Nz
                if i == 1
                    dx2 = (U[i+1,j,k] -2 * U[i,j,k]) / (A.Δx)^2
                elseif i == A.Nx
                    dx2 = (-2 * U[i,j,k] + U[i-1,j,k]) / (A.Δx)^2
                else
                    dx2 = (U[i+1,j,k] -2 * U[i,j,k] + U[i-1,j,k]) / (A.Δx)^2
                end

                if j == 1
                    dy2 = (U[i,j+1,k] -2 * U[i,j,k]) / (A.Δy)^2
                elseif j == A.Ny
                    dy2 = (-2 * U[i,j,k] + U[i,j-1,k]) / (A.Δy)^2
                else
                    dy2 = (U[i,j+1,k] -2 * U[i,j,k] + U[i,j-1,k]) / (A.Δy)^2
                end

                if k == 1
                    dz2 = (U[i,j,k+1] -2 * U[i,j,k]) / (A.Δz)^2
                elseif k == A.Nz
                    dz2 = (-2 * U[i,j,k] + U[i,j,k-1]) / (A.Δz)^2
                else
                    dz2 = (U[i,j,k+1] -2 * U[i,j,k] + U[i,j,k-1]) / (A.Δz)^2
                end

                Y[i,j,k] = dx2 + dy2 + dz2 + (A.k)^2 * U[i,j,k]
            end
        end
    end

    return y
end

# Create the matrix-free operator for the Helmholtz equation
A = HelmholtzOperator(Nx, Ny, Nz, Δx, Δy, Δz, k)

# Source term f(x, y, z) = -2k² * sin(kx) * sin(ky) * sin(kz)
F = [-2 * k^2 * sin(k * x[ii+1]) * sin(k * y[jj+1]) * sin(k * z[kk+1]) for ii in 1:Nx, jj in 1:Ny, kk in 1:Nz]
f = vec(F)

# Solve the linear system using MinAres
u_sol, stats = minares(A, f, atol=1e-10, rtol=0.0, verbose=1)
```

```@example helmholtz
# Solution as 3D array
U_sol = reshape(u_sol, Nx, Ny, Nz)

# Exact solution u(x,y,z) = sin(kx) * sin(ky) * sin(kz)
U_star = [sin(k * x[ii+1]) * sin(k * y[jj+1]) * sin(k * z[kk+1]) for ii in 1:Nx, jj in 1:Ny, kk in 1:Nz]

# Compute the maximum error between the numerical solution U_sol and the exact solution U_star
norm(U_sol - U_star, Inf)
```

Note that preconditioners can be also implemented as abstract operators.
For instance, we could compute the Cholesky factorization of $M$ and $N$ and create linear operators that perform the forward and backsolves.

Krylov methods combined with factorization free operators allow to reduce computation time and memory requirements considerably by avoiding building and storing the system matrix.
In the field of partial differential equations, the implementation of high-performance factorization free operators and assembly free preconditioning is a subject of active research.
