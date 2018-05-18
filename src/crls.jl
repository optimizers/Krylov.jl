# An implementation of CRLS for the solution of the
# over-determined linear least-squares problem
#
#  minimize ‖Ax - b‖
#
# equivalently, of the linear system
#
#  A'Ax = A'b.
#
# This implementation follows the formulation given in
#
# D. C.-L. Fong, Minimum-Residual Methods for Sparse
# Least-Squares using Golubg-Kahan Bidiagonalization,
# Ph.D. Thesis, Stanford University, 2011
#
# with the difference that it also recurs r = b - Ax.
#
# Dominique Orban, <dominique.orban@gerad.ca>
# Princeton, NJ, March 2015.

export crls


"""Solve the linear least-squares problem

  minimize ‖b - Ax‖₂² + λ ‖x‖₂²

using the Conjugate Residuals (CR) method. This method is equivalent to
applying MINRES to the normal equations

  (A'A + λI) x = A'b.

This implementation recurs the residual r := b - Ax.

CRLS produces monotonic residuals ‖r‖₂ and optimality residuals ‖A'r‖₂.
It is formally equivalent to LSMR, though can be slightly less accurate,
but simpler to implement.
"""
function crls{T <: Number}(A :: AbstractLinearOperator, b :: Vector{T};
                           M :: AbstractLinearOperator=opEye(size(b,1)),
                           λ :: Float64=0.0, atol :: Float64=1.0e-8, rtol :: Float64=1.0e-6,
                           radius :: Float64=0.0, itmax :: Int=0, verbose :: Bool=false)

  m, n = size(A);
  size(b, 1) == m || error("Inconsistent problem size");
  verbose && @printf("CRLS: system of %d equations in %d variables\n", m, n);

  x = zeros(T, n)
  r  = copy(b)
  bNorm = @knrm2(m, r)  # norm(b - A * x0) if x0 ≠ 0.
  bNorm == 0 && return x, SimpleStats(true, false, [0.0], [0.0], "x = 0 is a zero-residual solution");
  Mr = M * r;

  # The following vector copy takes care of the case where A is a LinearOperator
  # with preallocation, so as to avoid overwriting vectors used later. In other
  # cases, this should only add minimum overhead.
  Ar = copy(A' * Mr);  # - λ * x0 if x0 ≠ 0.

  s  = A * Ar;
  Ms = M * s;

  p  = copy(Ar);
  Ap = copy(s);
  q  = A' * Ms;  # Ap;
  λ > 0 && @kaxpy!(n, λ, p, q)  # q = q + λ * p;
  γ  = @kdot(m, s, Ms)  # Faster than γ = dot(s, Ms);
  iter = 0;
  itmax == 0 && (itmax = m + n);

  rNorm = bNorm;  # + λ * ‖x0‖ if x0 ≠ 0 and λ > 0.
  ArNorm = @knrm2(n, Ar)  # Marginally faster than norm(Ar);
  λ > 0 && (γ += λ * ArNorm * ArNorm);
  rNorms = [rNorm;];
  ArNorms = [ArNorm;];
  ε = atol + rtol * ArNorm;
  verbose && @printf("%5s  %8s  %8s\n", "Aprod", "‖A'r‖", "‖r‖")
  verbose && @printf("%5d  %8.2e  %8.2e\n", 3, ArNorm, rNorm);

  status = "unknown";
  on_boundary = false
  solved = ArNorm <= ε;
  tired = iter >= itmax;

  while ! (solved || tired)
    α = γ / @kdot(n, q, q)     # dot(q, q);

    # if a trust-region constraint is give, compute step to the boundary
    # (note that α > 0 in CRLS)
    if radius > 0.0
      σ = maximum(to_boundary(x, p, radius))
      ApNorm² = @knrm2(m, Ap)^2
      if ApNorm² = 0 # q is linear in the direction p
        spd = true # det(A'A) = 0
        pAr = @kdot(n, p, Ar) # p'A'r
        if pAr = 0 # q is constant in the direction p
          p = Ar # p = A'r
          q = A' * s
          α = minimum(ArNorm² / (2 * γ), maximum(to_boundary(x, Ar, radius)) # q is minimal in the direction A'r for ν = ‖Ar‖²/(2γ)
        else
          descent = pAr > 0.0
          if !descent 
            σ = minimum(to_boundary(x, p, radius))
          end
          ArNorm² = ArNorm^2
          ν = minimum(ArNorm² / (2 * γ), maximum(to_boundary(x, Ar, radius)) 
          δ = -σ * pAr + ν * ArNorm² + σ^2 * ApNorm² - ν^2 * γ
          if δ > 0.0
            # direction A'r engenders a bigger decrease
            p = Ar # p = A'r
            q = A' * s
            α = ν
          else
            α = σ
          end
        end
        if α > σ
          α = σ
          on_boundary = true
      end       

    @kaxpy!(n,  α, p,   x)     # Faster than  x =  x + α *  p;
    @kaxpy!(n, -α, q,  Ar)     # Faster than Ar = Ar - α *  q;
    ArNorm = @knrm2(n, Ar)
    solved = ArNorm <= ε
    (solved || on_boundary || spd) && continue
    @kaxpy!(m, -α, Ap,  r)     # Faster than  r =  r - α * Ap;
    s = A * Ar;
    Ms = M * s;
    γ_next = @kdot(m, s, Ms)   # Faster than γ_next = dot(s, s);
    λ > 0 && (γ_next += λ * ArNorm * ArNorm);
    β = γ_next / γ;

    @kscal!(n, β, p)
    @kaxpy!(n, 1.0, Ar, p)    # Faster than  p = Ar + β *  p;
    # The combined call uses less memory but tends to trigger more gc.
    #     BLAS.axpy!(n, 1.0, Ar, 1, BLAS.scal!(n, β, p, 1), 1);

    @kscal!(m, β, Ap)
    @kaxpy!(m, 1.0, s, Ap)    # Faster than Ap =  s + β * Ap;
    q = A' * M * Ap;
    λ > 0 && @kaxpy!(n, λ, p, q)  # q = q + λ * p;

    γ = γ_next;
    if λ > 0
      rNorm = sqrt(@kdot(m, r, r) + λ * @kdot(n, x, x))
    else
      rNorm = @knrm2(m, r)  # norm(r);
    end
    #     ArNorm = norm(Ar);
    push!(rNorms, rNorm);
    push!(ArNorms, ArNorm);
    iter = iter + 1;
    verbose && @printf("%5d  %8.2e  %8.2e\n", 3 + 2 * iter, ArNorm, rNorm);
    solved = (ArNorm <= ε) | on_boundary
    tired = iter >= itmax;
  end

  status = on_boundary ? "on trust-region boundary" : (tired ? "maximum number of iterations exceeded" : "solution good enough given atol and rtol")
  stats = SimpleStats(solved, false, rNorms, ArNorms, status);
  return (x, stats);
end
