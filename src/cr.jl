# A truncated version of Stiefel’s Conjugate Residual method
# cr(A, b, Δ, rtol, itmax, verbose) solves the linear system 'A * x = b' or the least-squares problem :
# 'min ‖b - A * x‖²' within a region of fixed radius Δ.
#
# Marie-Ange Dahito, <marie-ange.dahito@polymtl.ca>
# Montreal, QC, June 2017

export cr

"""A truncated version of Stiefel’s Conjugate Residual method to solve the symmetric linear system Ax=b.
The matrix A must be positive semi-definite
"""
function cr{T <: Number}(A :: AbstractLinearOperator, b :: Vector{T}, atol :: Float64=1.0e-8, rtol :: Float64=1.0e-6, itmax :: Int=0, radius :: Float64=0., verbose :: Bool=true)

  n = size(b, 1) # size of the problem
  (size(A, 1) == n & size(A, 2) == n) || error("Inconsistent problem size")
  verbose && @printf("CR: system of %d equations in %d variables\n", n, n)

  # Initial state.
  x = zeros(T, n) # initial estimation x = 0
  r = copy(b) # initial residual r = b - Ax = b
  Ar = A * r
  ρ = @kdot(n, r, Ar)
  ρ == 0.0 && return (x, Krylov.SimpleStats(true, false, [0.0], [], "x = 0 is a zero-residual solution"))
  p = copy(r)
  q = copy(Ar)

  iter = 0
  itmax == 0 && (itmax = 2 * n)

  m = 0.0
  mvalues = [m] # values of the quadratic model
  xNorm = 0.0
  xNorms = [xNorm] # Values of ‖x‖
  rNorm = @knrm2(n, r) # ‖r‖
  rNorms = [rNorm] # Values of ‖r‖
  ArNorm = @knrm2(n, Ar) # ‖Ar‖
  ArNorms = [ArNorm]
  ε = atol + rtol * rNorm
  verbose && @printf("%5s %6s %10s %10s %10s %10s\n", "Iter", "‖x‖", "‖r‖", "q", "α", "σ")
  verbose && @printf("    %d  %8.1e    %8.1e    %8.1e", iter, xNorm, rNorm, m)

  solved = rNorm <= ε
  tired = iter >= itmax
  on_boundary = false
  status = "unknown"

  while ! (solved || tired)
    α = ρ / @knrm2(n, q)^2 # step

    # Compute step size to boundary if applicable.
    σ = radius > 0.0 ? to_boundary(x, p, radius) : α

    verbose && @printf("  %7.1e   %7.1e\n", α, σ);

    # Move along p from x to the boundary if either
    # the next step leads outside the trust region or
    # we have nonpositive curvature.
    if (radius > 0.0) & (α > σ)
      α = σ
      on_boundary = true
    end

    @kaxpy!(n,  α,  p, x)
    xNorm = @knrm2(n, x)
    push!(xNorms, xNorm)
    Ax = A * x
    m = @kdot(n, -b, x) + 1/2 * @kdot(n, x, Ax)
    push!(mvalues, m)
    @kaxpy!(n, -α, q, r) # residual
    rNorm = @knrm2(n, r)
    push!(rNorms, rNorm)
    Ar = A * r
    ArNorm = @knrm2(n, Ar)
    push!(ArNorms, ArNorm)

    solved = (rNorm <= ε) | on_boundary
    tired = iter >= itmax

    if !solved
      ρbar = ρ
      ρ = @kdot(n, r, Ar)
      β = ρ / ρbar # step for the direction calculus
      p = r + β * p # descent direction
      q = Ar + β * q
    end
    iter = iter + 1
    verbose && @printf("    %d  %8.1e    %8.1e    %8.1e", iter, xNorm, rNorm, m)

  end
  verbose && @printf("\n")

  status = on_boundary ? "on trust-region boundary" : (tired ? "maximum number of iterations exceeded" : "solution good enough given atol and rtol")
  stats = Krylov.SimpleStats(solved, false, rNorms, ArNorms, status)
  return (x, stats)
end
