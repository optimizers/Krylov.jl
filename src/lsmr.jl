# An implementation of LSMR for the solution of the
# over-determined linear least-squares problem
#
#  minimize ‖Ax - b‖₂
#
# equivalently, of the normal equations
#
#  AᴴAx = Aᴴb.
#
# LSMR is formally equivalent to applying MINRES to the normal equations
# but should be more stable. It is also formally equivalent to CRLS though
# LSMR should be expected to be more stable on ill-conditioned or poorly
# scaled problems.
#
# This implementation follows the original implementation by
# Michael Saunders described in
#
# D. C.-L. Fong and M. A. Saunders, LSMR: An Iterative Algorithm for Sparse
# Least Squares Problems, SIAM Journal on Scientific Computing, 33(5),
# pp. 2950--2971, 2011.
#
# Dominique Orban, <dominique.orban@gerad.ca>
# Montreal, QC, May 2015.

export lsmr, lsmr!


"""
    (x, stats) = lsmr(A, b::AbstractVector{FC};
                      M=I, N=I, sqd::Bool=false, λ::T=zero(T),
                      axtol::T=√eps(T), btol::T=√eps(T),
                      atol::T=zero(T), rtol::T=zero(T),
                      etol::T=√eps(T), window::Int=5,
                      itmax::Int=0, conlim::T=1/√eps(T),
                      radius::T=zero(T), verbose::Int=0,
                      history::Bool=false, ldiv::Bool=false,
                      callback=solver->false)

`T` is an `AbstractFloat` such as `Float32`, `Float64` or `BigFloat`.
`FC` is `T` or `Complex{T}`.

Solve the regularized linear least-squares problem

    minimize ‖b - Ax‖₂² + λ²‖x‖₂²

using the LSMR method, where λ ≥ 0 is a regularization parameter.
LSMR is formally equivalent to applying MINRES to the normal equations

    (AᴴA + λ²I) x = Aᴴb

(and therefore to CRLS) but is more stable.

LSMR produces monotonic residuals ‖r‖₂ and optimality residuals ‖Aᴴr‖₂.
It is formally equivalent to CRLS, though can be substantially more accurate.

LSMR can be also used to find a null vector of a singular matrix A
by solving the problem `min ‖Aᴴx - b‖` with any nonzero vector `b`.
At a minimizer, the residual vector `r = b - Aᴴx` will satisfy `Ar = 0`.

If `λ > 0`, we solve the symmetric and quasi-definite system

    [ E      A ] [ r ]   [ b ]
    [ Aᴴ  -λ²F ] [ x ] = [ 0 ],

where E and F are symmetric and positive definite.
Preconditioners M = E⁻¹ ≻ 0 and N = F⁻¹ ≻ 0 may be provided in the form of linear operators.
If `sqd=true`, `λ` is set to the common value `1`.

The system above represents the optimality conditions of

    minimize ‖b - Ax‖²_E⁻¹ + λ²‖x‖²_F.

For a symmetric and positive definite matrix `K`, the K-norm of a vector `x` is `‖x‖²_K = xᴴKx`.
LSMR is then equivalent to applying MINRES to `(AᴴE⁻¹A + λ²F)x = AᴴE⁻¹b` with `r = E⁻¹(b - Ax)`.

If `λ = 0`, we solve the symmetric and indefinite system

    [ E    A ] [ r ]   [ b ]
    [ Aᴴ   0 ] [ x ] = [ 0 ].

The system above represents the optimality conditions of

    minimize ‖b - Ax‖²_E⁻¹.

In this case, `N` can still be specified and indicates the weighted norm in which `x` and `Aᴴr` should be measured.
`r` can be recovered by computing `E⁻¹(b - Ax)`.

The callback is called as `callback(solver)` and should return `true` if the main loop should terminate,
and `false` otherwise.

#### Input arguments

* `A`: a linear operator that models a matrix of dimension n × m;
* `b`: a vector of length n.

#### Output arguments

* `x`: a dense vector of length m;
* `stats`: statistics collected on the run in a [`LsmrStats`](@ref) structure.

#### Reference

* D. C.-L. Fong and M. A. Saunders, [*LSMR: An Iterative Algorithm for Sparse Least Squares Problems*](https://doi.org/10.1137/10079687X), SIAM Journal on Scientific Computing, 33(5), pp. 2950--2971, 2011.
"""
function lsmr end

function lsmr(A, b :: AbstractVector{FC}; window :: Int=5, kwargs...) where FC <: FloatOrComplex
  solver = LsmrSolver(A, b, window=window)
  lsmr!(solver, A, b; kwargs...)
  return (solver.x, solver.stats)
end

"""
    solver = lsmr!(solver::LsmrSolver, A, b; kwargs...)

where `kwargs` are keyword arguments of [`lsmr`](@ref).

See [`LsmrSolver`](@ref) for more details about the `solver`.
"""
function lsmr! end

function lsmr!(solver :: LsmrSolver{T,FC,S}, A, b :: AbstractVector{FC};
               M=I, N=I, sqd :: Bool=false, λ :: T=zero(T),
               axtol :: T=√eps(T), btol :: T=√eps(T),
               atol :: T=zero(T), rtol :: T=zero(T),
               etol :: T=√eps(T), itmax :: Int=0, conlim :: T=1/√eps(T),
               radius :: T=zero(T), verbose :: Int=0, history :: Bool=false,
               ldiv :: Bool=false, callback = solver -> false) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: DenseVector{FC}}

  m, n = size(A)
  length(b) == m || error("Inconsistent problem size")
  (verbose > 0) && @printf("LSMR: system of %d equations in %d variables\n", m, n)

  # Check sqd and λ parameters
  sqd && (λ ≠ 0) && error("sqd cannot be set to true if λ ≠ 0 !")
  sqd && (λ = one(T))

  # Tests M = Iₙ and N = Iₘ
  MisI = (M === I)
  NisI = (N === I)

  # Check type consistency
  eltype(A) == FC || error("eltype(A) ≠ $FC")
  ktypeof(b) == S || error("ktypeof(b) ≠ $S")

  # Compute the adjoint of A
  Aᴴ = A'

  # Set up workspace.
  allocate_if(!MisI, solver, :u, S, m)
  allocate_if(!NisI, solver, :v, S, n)
  x, Nv, Aᴴu, h, hbar = solver.x, solver.Nv, solver.Aᴴu, solver.h, solver.hbar
  Mu, Av, err_vec, stats = solver.Mu, solver.Av, solver.err_vec, solver.stats
  rNorms, ArNorms = stats.residuals, stats.Aresiduals
  reset!(stats)
  u = MisI ? Mu : solver.u
  v = NisI ? Nv : solver.v

  ctol = conlim > 0 ? 1/conlim : zero(T)
  x .= zero(FC)

  # Initialize Golub-Kahan process.
  # β₁ M u₁ = b.
  Mu .= b
  MisI || mulorldiv!(u, M, Mu, ldiv)
  β₁ = sqrt(@kdotr(m, u, Mu))
  if β₁ == 0
    stats.niter = 0
    stats.solved, stats.inconsistent = true, false
    stats.status = "x = 0 is a zero-residual solution"
    history && push!(rNorms, zero(T))
    history && push!(ArNorms, zero(T))
    return solver
  end
  β = β₁

  @kscal!(m, one(FC)/β₁, u)
  MisI || @kscal!(m, one(FC)/β₁, Mu)
  mul!(Aᴴu, Aᴴ, u)
  Nv .= Aᴴu
  NisI || mulorldiv!(v, N, Nv, ldiv)
  α = sqrt(@kdotr(n, v, Nv))

  ζbar = α * β
  αbar = α
  ρ = one(T)
  ρbar = one(T)
  cbar = one(T)
  sbar = zero(T)

  # Initialize variables for estimation of ‖r‖.
  βdd = β
  βd = zero(T)
  ρdold = one(T)
  τtildeold = zero(T)
  θtilde = zero(T)
  ζ = zero(T)
  d = zero(T)

  # Initialize variables for estimation of ‖A‖, cond(A) and xNorm.
  Anorm² = α * α
  maxrbar = zero(T)
  minrbar = min(floatmax(T), T(1.0e+100))
  Acond = maxrbar / minrbar
  Anorm = sqrt(Anorm²)
  xNorm = zero(T)

  # Items for use in stopping rules.
  ctol = conlim > 0 ? 1 / conlim : zero(T)
  rNorm = β
  history && push!(rNorms, rNorm)
  ArNorm = ArNorm0 = α * β
  history && push!(ArNorms, ArNorm)

  xENorm² = zero(T)
  err_lbnd = zero(T)
  window = length(err_vec)
  err_vec .= zero(T)

  iter = 0
  itmax == 0 && (itmax = m + n)

  (verbose > 0) && @printf("%5s  %7s  %7s  %7s  %7s  %8s  %8s  %7s\n", "k", "‖r‖", "‖Aᴴr‖", "β", "α", "cos", "sin", "‖A‖²")
  kdisplay(iter, verbose) && @printf("%5d  %7.1e  %7.1e  %7.1e  %7.1e  %8.1e  %8.1e  %7.1e\n", iter, β₁, α, β₁, α, 0, 1, Anorm²)

  # Aᴴb = 0 so x = 0 is a minimum least-squares solution
  if α == 0
    stats.niter = 0
    stats.solved, stats.inconsistent = true, false
    stats.status = "x = 0 is a minimum least-squares solution"
    return solver
  end
  @kscal!(n, one(FC)/α, v)
  NisI || @kscal!(n, one(FC)/α, Nv)

  h .= v
  hbar .= zero(FC)

  status = "unknown"
  on_boundary = false
  solved = solved_mach = solved_lim = (rNorm ≤ axtol)
  tired  = iter ≥ itmax
  ill_cond = ill_cond_mach = ill_cond_lim = false
  zero_resid = zero_resid_mach = zero_resid_lim = false
  fwd_err = false
  user_requested_exit = false

  while ! (solved || tired || ill_cond || user_requested_exit)
    iter = iter + 1

    # Generate next Golub-Kahan vectors.
    # 1. βₖ₊₁Muₖ₊₁ = Avₖ - αₖMuₖ
    mul!(Av, A, v)
    @kaxpby!(m, one(FC), Av, -α, Mu)
    MisI || mulorldiv!(u, M, Mu, ldiv)
    β = sqrt(@kdotr(m, u, Mu))
    if β ≠ 0
      @kscal!(m, one(FC)/β, u)
      MisI || @kscal!(m, one(FC)/β, Mu)

      # 2. αₖ₊₁Nvₖ₊₁ = Aᴴuₖ₊₁ - βₖ₊₁Nvₖ
      mul!(Aᴴu, Aᴴ, u)
      @kaxpby!(n, one(FC), Aᴴu, -β, Nv)
      NisI || mulorldiv!(v, N, Nv, ldiv)
      α = sqrt(@kdotr(n, v, Nv))
      if α ≠ 0
        @kscal!(n, one(FC)/α, v)
        NisI || @kscal!(n, one(FC)/α, Nv)
      end
    end

    # Continue QR factorization
    (chat, shat, αhat) = sym_givens(αbar, λ)

    ρold = ρ
    (c, s, ρ) = sym_givens(αhat, β)
    θnew = s * α
    αbar = c * α

    ρbarold = ρbar
    ζold = ζ
    θbar = sbar * ρ
    ρtemp = cbar * ρ
    (cbar, sbar, ρbar) = sym_givens(ρtemp, θnew)
    ζ = cbar * ζbar
    ζbar = -sbar * ζbar

    xENorm² = xENorm² + ζ * ζ
    err_vec[mod(iter, window) + 1] = ζ
    iter ≥ window && (err_lbnd = @knrm2(window, err_vec))

    # Update h, hbar and x.
    δ = θbar * ρ / (ρold * ρbarold) # δₖ = θbarₖ * ρₖ / (ρₖ₋₁ * ρbarₖ₋₁)
    @kaxpby!(n, one(FC), h, -δ, hbar)   # ĥₖ = hₖ - δₖ * ĥₖ₋₁

    # if a trust-region constraint is given, compute step to the boundary
    # the step ϕ/ρ is not necessarily positive
    σ = ζ / (ρ * ρbar)
    if radius > 0
      t1, t2 = to_boundary(n, x, hbar, radius)
      tmax, tmin = max(t1, t2), min(t1, t2)
      on_boundary = σ > tmax || σ < tmin
      σ = σ > 0 ? min(σ, tmax) : max(σ, tmin)
    end

    @kaxpy!(n, σ, hbar, x) # xₖ = xₖ₋₁ + σₖ * ĥₖ
    @kaxpby!(n, one(FC), v, -θnew / ρ, h) # hₖ₊₁ = vₖ₊₁ - (θₖ₊₁/ρₖ) * hₖ

    # Estimate ‖r‖.
    βacute =  chat * βdd
    βcheck = -shat * βdd

    βhat =  c * βacute
    βdd  = -s * βacute

    θtildeold = θtilde
    (ctildeold, stildeold, ρtildeold) = sym_givens(ρdold, θbar)
    θtilde = stildeold * ρbar
    ρdold = ctildeold * ρbar
    βd = -stildeold * βd + ctildeold * βhat

    τtildeold = (ζold - θtildeold * τtildeold) / ρtildeold
    τd = (ζ - θtilde * τtildeold) / ρdold
    d = d + βcheck * βcheck
    rNorm = sqrt(d + (βd - τd)^2 + βdd * βdd)
    history && push!(rNorms, rNorm)

    # Estimate ‖A‖.
    Anorm² += β * β
    Anorm   = sqrt(Anorm²)
    Anorm² += α * α

    # Estimate cond(A).
    maxrbar = max(maxrbar, ρbarold)
    iter > 1 && (minrbar = min(minrbar, ρbarold))
    Acond = max(maxrbar, ρtemp) / min(minrbar, ρtemp)

    # Test for convergence.
    ArNorm = abs(ζbar)
    history && push!(ArNorms, ArNorm)
    xNorm = @knrm2(n, x)

    test1 = rNorm / β₁
    test2 = ArNorm / (Anorm * rNorm)
    test3 = 1 / Acond
    t1    = test1 / (one(T) + Anorm * xNorm / β₁)
    rNormtol  = btol + axtol * Anorm * xNorm / β₁

    kdisplay(iter, verbose) && @printf("%5d  %7.1e  %7.1e  %7.1e  %7.1e  %8.1e  %8.1e  %7.1e\n", iter, rNorm, ArNorm, β, α, c, s, Anorm²)

    # Stopping conditions that do not depend on user input.
    # This is to guard against tolerances that are unreasonably small.
    ill_cond_mach = (one(T) + test3 ≤ one(T))
    solved_mach = (one(T) + test2 ≤ one(T))
    zero_resid_mach = (one(T) + t1 ≤ one(T))

    # Stopping conditions based on user-provided tolerances.
    user_requested_exit = callback(solver) :: Bool
    tired  = iter ≥ itmax
    ill_cond_lim = (test3 ≤ ctol)
    solved_lim = (test2 ≤ axtol)
    solved_opt = ArNorm ≤ atol + rtol * ArNorm0
    zero_resid_lim = (test1 ≤ rNormtol)
    iter ≥ window && (fwd_err = err_lbnd ≤ etol * sqrt(xENorm²))

    ill_cond = ill_cond_mach | ill_cond_lim
    zero_resid = zero_resid_mach | zero_resid_lim
    solved = solved_mach | solved_lim | solved_opt | zero_resid | fwd_err | on_boundary
  end
  (verbose > 0) && @printf("\n")

  tired               && (status = "maximum number of iterations exceeded")
  ill_cond_mach       && (status = "condition number seems too large for this machine")
  ill_cond_lim        && (status = "condition number exceeds tolerance")
  solved              && (status = "found approximate minimum least-squares solution")
  zero_resid          && (status = "found approximate zero-residual solution")
  fwd_err             && (status = "truncated forward error small enough")
  on_boundary         && (status = "on trust-region boundary")
  user_requested_exit && (status = "user-requested exit")

  # Update stats
  stats.residual = rNorm
  stats.Aresidual = ArNorm
  stats.Acond = Acond
  stats.Anorm = Anorm
  stats.xNorm = xNorm
  stats.niter = iter
  stats.solved = solved
  stats.inconsistent = !zero_resid
  stats.status = status
  return solver
end
