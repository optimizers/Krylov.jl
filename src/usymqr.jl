# An implementation of USYMQR for the solution of linear system Ax = b.
#
# This method is described in
#
# M. A. Saunders, H. D. Simon, and E. L. Yip
# Two Conjugate-Gradient-Type Methods for Unsymmetric Linear Equations.
# SIAM Journal on Numerical Analysis, 25(4), pp. 927--940, 1988.
#
# L. Reichel and Q. Ye
# A generalized LSQR algorithm.
# Numerical Linear Algebra with Applications, 15(7), pp. 643--660, 2008.
#
# A. Buttari, D. Orban, D. Ruiz and D. Titley-Peloquin
# A tridiagonalization method for symmetric saddle-point and quasi-definite systems.
# SIAM Journal on Scientific Computing, 41(5), pp. 409--432, 2019.
#
# A. Montoison and D. Orban
# BiLQ: An Iterative Method for Nonsymmetric Linear Systems with a Quasi-Minimum Error Property.
# SIAM Journal on Matrix Analysis and Applications, 41(3), pp. 1145--1166, 2020.
#
# Alexis Montoison, <alexis.montoison@polymtl.ca>
# Montreal, November 2018.

export usymqr, usymqr!

"""
    (x, stats) = usymqr(A, b::AbstractVector{FC}, c::AbstractVector{FC};
                        atol::T=√eps(T), rtol::T=√eps(T), itmax::Int=0,
                        timemax::Float64=Inf, verbose::Int=0, history::Bool=false,
                        callback=solver->false, iostream::IO=kstdout)

`T` is an `AbstractFloat` such as `Float32`, `Float64` or `BigFloat`.
`FC` is `T` or `Complex{T}`.

    (x, stats) = usymqr(A, b, c, x0::AbstractVector; kwargs...)

USYMQR can be warm-started from an initial guess `x0` where `kwargs` are the same keyword arguments as above.

USYMQR solves the linear least-squares problem min ‖b - Ax‖² of size m × n.
USYMQR solves Ax = b if it is consistent.

USYMQR is based on the orthogonal tridiagonalization process and requires two initial nonzero vectors `b` and `c`.
The vector `c` is only used to initialize the process and a default value can be `b` or `Aᴴb` depending on the shape of `A`.
The residual norm ‖b - Ax‖ monotonously decreases in USYMQR.
When `A` is Hermitian and `b = c`, QMR is equivalent to MINRES.
USYMQR is considered as a generalization of MINRES.

It can also be applied to under-determined and over-determined problems.
USYMQR finds the minimum-norm solution if problems are inconsistent.

#### Input arguments

* `A`: a linear operator that models a matrix of dimension `m × n`;
* `b`: a vector of length `m`;
* `c`: a vector of length `n`.

#### Optional argument

* `x0`: a vector of length `n` that represents an initial guess of the solution `x`.

#### Keyword arguments

* `atol`: absolute stopping tolerance based on the residual norm;
* `rtol`: relative stopping tolerance based on the residual norm;
* `itmax`: the maximum number of iterations. If `itmax=0`, the default number of iterations is set to `m+n`;
* `timemax`: the time limit in seconds;
* `verbose`: additional details can be displayed if verbose mode is enabled (verbose > 0). Information will be displayed every `verbose` iterations;
* `history`: collect additional statistics on the run such as residual norms, or Aᴴ-residual norms;
* `callback`: function or functor called as `callback(solver)` that returns `true` if the Krylov method should terminate, and `false` otherwise;
* `iostream`: stream to which output is logged.

#### Output arguments

* `x`: a dense vector of length `n`;
* `stats`: statistics collected on the run in a [`SimpleStats`](@ref) structure.

#### References

* M. A. Saunders, H. D. Simon, and E. L. Yip, [*Two Conjugate-Gradient-Type Methods for Unsymmetric Linear Equations*](https://doi.org/10.1137/0725052), SIAM Journal on Numerical Analysis, 25(4), pp. 927--940, 1988.
* L. Reichel and Q. Ye, [*A generalized LSQR algorithm*](https://doi.org/10.1002/nla.611), Numerical Linear Algebra with Applications, 15(7), pp. 643--660, 2008.
* A. Buttari, D. Orban, D. Ruiz and D. Titley-Peloquin, [*A tridiagonalization method for symmetric saddle-point and quasi-definite systems*](https://doi.org/10.1137/18M1194900), SIAM Journal on Scientific Computing, 41(5), pp. 409--432, 2019.
* A. Montoison and D. Orban, [*BiLQ: An Iterative Method for Nonsymmetric Linear Systems with a Quasi-Minimum Error Property*](https://doi.org/10.1137/19M1290991), SIAM Journal on Matrix Analysis and Applications, 41(3), pp. 1145--1166, 2020.
"""
function usymqr end

"""
    solver = usymqr!(solver::UsymqrSolver, A, b, c; kwargs...)
    solver = usymqr!(solver::UsymqrSolver, A, b, c, x0; kwargs...)

where `kwargs` are keyword arguments of [`usymqr`](@ref).

See [`UsymqrSolver`](@ref) for more details about the `solver`.
"""
function usymqr! end

def_args_usymqr = (:(A                    ),
                   :(b::AbstractVector{FC}),
                   :(c::AbstractVector{FC}))

def_optargs_usymqr = (:(x0::AbstractVector),)

def_kwargs_usymqr = (:(; atol::T = √eps(T)         ),
                     :(; rtol::T = √eps(T)         ),
                     :(; itmax::Int = 0            ),
                     :(; timemax::Float64 = Inf    ),
                     :(; verbose::Int = 0          ),
                     :(; history::Bool = false     ),
                     :(; callback = solver -> false),
                     :(; iostream::IO = kstdout    ))

def_kwargs_usymqr = extract_parameters.(def_kwargs_usymqr)

args_usymqr = (:A, :b, :c)
optargs_usymqr = (:x0,)
kwargs_usymqr = (:atol, :rtol, :itmax, :timemax, :verbose, :history, :callback, :iostream)

@eval begin
  function usymqr!(solver :: UsymqrSolver{T,FC,S}, $(def_args_usymqr...); $(def_kwargs_usymqr...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: AbstractVector{FC}}

    # Timer
    start_time = time_ns()
    timemax_ns = 1e9 * timemax

    m, n = size(A)
    (m == solver.m && n == solver.n) || error("(solver.m, solver.n) = ($(solver.m), $(solver.n)) is inconsistent with size(A) = ($m, $n)")
    length(b) == m || error("Inconsistent problem size")
    length(c) == n || error("Inconsistent problem size")
    (verbose > 0) && @printf(iostream, "USYMQR: system of %d equations in %d variables\n", m, n)

    # Check type consistency
    eltype(A) == FC || @warn "eltype(A) ≠ $FC. This could lead to errors or additional allocations in operator-vector products."
    ktypeof(b) <: S || error("ktypeof(b) is not a subtype of $S")
    ktypeof(c) <: S || error("ktypeof(c) is not a subtype of $S")

    # Compute the adjoint of A
    Aᴴ = A'

    # Set up workspace.
    vₖ₋₁, vₖ, q, Δx, x, p = solver.vₖ₋₁, solver.vₖ, solver.q, solver.Δx, solver.x, solver.p
    wₖ₋₂, wₖ₋₁, uₖ₋₁, uₖ, stats = solver.wₖ₋₂, solver.wₖ₋₁, solver.uₖ₋₁, solver.uₖ, solver.stats
    warm_start = solver.warm_start
    rNorms, AᴴrNorms = stats.residuals, stats.Aresiduals
    reset!(stats)
    r₀ = warm_start ? q : b

    if warm_start
      mul!(r₀, A, Δx)
      kaxpby!(n, one(FC), b, -one(FC), r₀)
    end

    # Initial solution x₀ and residual norm ‖r₀‖.
    kfill!(x, zero(FC))
    rNorm = knorm(m, r₀)
    history && push!(rNorms, rNorm)
    if rNorm == 0
      stats.niter = 0
      stats.solved = true
      stats.inconsistent = false
      stats.timer = start_time |> ktimer
      stats.status = "x = 0 is a zero-residual solution"
      solver.warm_start = false
      return solver
    end

    iter = 0
    itmax == 0 && (itmax = m+n)

    ε = atol + rtol * rNorm
    κ = zero(T)
    (verbose > 0) && @printf(iostream, "%5s  %7s  %8s  %5s\n", "k", "‖rₖ‖", "‖Aᴴrₖ₋₁‖", "timer")
    kdisplay(iter, verbose) && @printf(iostream, "%5d  %7.1e  %8s  %.2fs\n", iter, rNorm, " ✗ ✗ ✗ ✗", start_time |> ktimer)

    βₖ = knorm(m, r₀)            # β₁ = ‖v₁‖ = ‖r₀‖
    γₖ = knorm(n, c)             # γ₁ = ‖u₁‖ = ‖c‖
    kfill!(vₖ₋₁, zero(FC))       # v₀ = 0
    kfill!(uₖ₋₁, zero(FC))       # u₀ = 0
    vₖ .= r₀ ./ βₖ               # v₁ = (b - Ax₀) / β₁
    uₖ .= c ./ γₖ                # u₁ = c / γ₁
    cₖ₋₂ = cₖ₋₁ = cₖ = one(T)    # Givens cosines used for the QR factorization of Tₖ₊₁.ₖ
    sₖ₋₂ = sₖ₋₁ = sₖ = zero(FC)  # Givens sines used for the QR factorization of Tₖ₊₁.ₖ
    kfill!(wₖ₋₂, zero(FC))       # Column k-2 of Wₖ = Uₖ(Rₖ)⁻¹
    kfill!(wₖ₋₁, zero(FC))       # Column k-1 of Wₖ = Uₖ(Rₖ)⁻¹
    ζbarₖ = βₖ                   # ζbarₖ is the last component of z̅ₖ = (Qₖ)ᴴβ₁e₁

    # Stopping criterion.
    solved = rNorm ≤ ε
    inconsistent = false
    tired = iter ≥ itmax
    status = "unknown"
    user_requested_exit = false
    overtimed = false

    while !(solved || tired || inconsistent || user_requested_exit || overtimed)
      # Update iteration index.
      iter = iter + 1

      # Continue the SSY tridiagonalization process.
      # AUₖ  = VₖTₖ    + βₖ₊₁vₖ₊₁(eₖ)ᵀ = Vₖ₊₁Tₖ₊₁.ₖ
      # AᴴVₖ = Uₖ(Tₖ)ᴴ + γₖ₊₁uₖ₊₁(eₖ)ᵀ = Uₖ₊₁(Tₖ.ₖ₊₁)ᴴ

      mul!(q, A , uₖ)  # Forms vₖ₊₁ : q ← Auₖ
      mul!(p, Aᴴ, vₖ)  # Forms uₖ₊₁ : p ← Aᴴvₖ

      if iter ≥ 2
        kaxpy!(m, -γₖ, vₖ₋₁, q) # q ← q - γₖ * vₖ₋₁
        kaxpy!(n, -βₖ, uₖ₋₁, p) # p ← p - βₖ * uₖ₋₁
      end

      αₖ = kdot(m, vₖ, q)  # αₖ = ⟨vₖ,q⟩

      kaxpy!(m, -     αₖ , vₖ, q)  # q ← q - αₖ * vₖ
      kaxpy!(n, -conj(αₖ), uₖ, p)  # p ← p - ᾱₖ * uₖ

      βₖ₊₁ = knorm(m, q)  # βₖ₊₁ = ‖q‖
      γₖ₊₁ = knorm(n, p)  # γₖ₊₁ = ‖p‖

      # Update the QR factorization of Tₖ₊₁.ₖ = Qₖ [ Rₖ ].
      #                                            [ Oᵀ ]
      # [ α₁ γ₂ 0  •  •  •   0  ]      [ δ₁ λ₁ ϵ₁ 0  •  •  0  ]
      # [ β₂ α₂ γ₃ •         •  ]      [ 0  δ₂ λ₂ •  •     •  ]
      # [ 0  •  •  •  •      •  ]      [ •  •  δ₃ •  •  •  •  ]
      # [ •  •  •  •  •  •   •  ] = Qₖ [ •     •  •  •  •  0  ]
      # [ •     •  •  •  •   0  ]      [ •        •  •  • ϵₖ₋₂]
      # [ •        •  •  •   γₖ ]      [ •           •  • λₖ₋₁]
      # [ •           •  βₖ  αₖ ]      [ 0  •  •  •  •  0  δₖ ]
      # [ 0  •  •  •  •  0  βₖ₊₁]      [ 0  •  •  •  •  •  0  ]
      #
      # If k = 1, we don't have any previous reflection.
      # If k = 2, we apply the last reflection.
      # If k ≥ 3, we only apply the two previous reflections.

      # Apply previous Givens reflections Qₖ₋₂.ₖ₋₁
      if iter ≥ 3
        # [cₖ₋₂  sₖ₋₂] [0 ] = [  ϵₖ₋₂ ]
        # [s̄ₖ₋₂ -cₖ₋₂] [γₖ]   [λbarₖ₋₁]
        ϵₖ₋₂    =  sₖ₋₂ * γₖ
        λbarₖ₋₁ = -cₖ₋₂ * γₖ
      end

      # Apply previous Givens reflections Qₖ₋₁.ₖ
      if iter ≥ 2
        iter == 2 && (λbarₖ₋₁ = γₖ)
        # [cₖ₋₁  sₖ₋₁] [λbarₖ₋₁] = [λₖ₋₁ ]
        # [s̄ₖ₋₁ -cₖ₋₁] [   αₖ  ]   [δbarₖ]
        λₖ₋₁  =      cₖ₋₁  * λbarₖ₋₁ + sₖ₋₁ * αₖ
        δbarₖ = conj(sₖ₋₁) * λbarₖ₋₁ - cₖ₋₁ * αₖ
      end

      # Compute and apply current Givens reflection Qₖ.ₖ₊₁
      iter == 1 && (δbarₖ = αₖ)
      # [cₖ  sₖ] [δbarₖ] = [δₖ]
      # [s̄ₖ -cₖ] [βₖ₊₁ ]   [0 ]
      (cₖ, sₖ, δₖ) = sym_givens(δbarₖ, βₖ₊₁)

      # Update z̅ₖ₊₁ = Qₖ.ₖ₊₁ [ z̄ₖ ]
      #                      [ 0  ]
      #
      # [cₖ  sₖ] [ζbarₖ] = [   ζₖ  ]
      # [s̄ₖ -cₖ] [  0  ]   [ζbarₖ₊₁]
      ζₖ      =      cₖ  * ζbarₖ
      ζbarₖ₊₁ = conj(sₖ) * ζbarₖ

      # Compute the direction wₖ, the last column of Wₖ = Uₖ(Rₖ)⁻¹ ⟷ (Rₖ)ᵀ(Wₖ)ᵀ = (Uₖ)ᵀ.
      # w₁ = u₁ / δ₁
      if iter == 1
        wₖ = wₖ₋₁
        kaxpy!(n, one(FC), uₖ, wₖ)
        wₖ .= wₖ ./ δₖ
      end
      # w₂ = (u₂ - λ₁w₁) / δ₂
      if iter == 2
        wₖ = wₖ₋₂
        kaxpy!(n, -λₖ₋₁, wₖ₋₁, wₖ)
        kaxpy!(n, one(FC), uₖ, wₖ)
        wₖ .= wₖ ./ δₖ
      end
      # wₖ = (uₖ - λₖ₋₁wₖ₋₁ - ϵₖ₋₂wₖ₋₂) / δₖ
      if iter ≥ 3
        kscal!(n, -ϵₖ₋₂, wₖ₋₂)
        wₖ = wₖ₋₂
        kaxpy!(n, -λₖ₋₁, wₖ₋₁, wₖ)
        kaxpy!(n, one(FC), uₖ, wₖ)
        wₖ .= wₖ ./ δₖ
      end

      # Compute solution xₖ.
      # xₖ ← xₖ₋₁ + ζₖ * wₖ
      kaxpy!(n, ζₖ, wₖ, x)

      # Compute ‖rₖ‖ = |ζbarₖ₊₁|.
      rNorm = abs(ζbarₖ₊₁)
      history && push!(rNorms, rNorm)

      # Compute ‖Aᴴrₖ₋₁‖ = |ζbarₖ| * √(|δbarₖ|² + |λbarₖ|²).
      AᴴrNorm = abs(ζbarₖ) * √(abs2(δbarₖ) + abs2(cₖ₋₁ * γₖ₊₁))
      history && push!(AᴴrNorms, AᴴrNorm)

      # Compute vₖ₊₁ and uₖ₊₁.
      kcopy!(m, vₖ₋₁, vₖ)  # vₖ₋₁ ← vₖ
      kcopy!(n, uₖ₋₁, uₖ)  # uₖ₋₁ ← uₖ

      if βₖ₊₁ ≠ zero(T)
        vₖ .= q ./ βₖ₊₁  # βₖ₊₁vₖ₊₁ = q
      end
      if γₖ₊₁ ≠ zero(T)
        uₖ .= p ./ γₖ₊₁  # γₖ₊₁uₖ₊₁ = p
      end

      # Update directions for x.
      if iter ≥ 2
        @kswap!(wₖ₋₂, wₖ₋₁)
      end

      # Update sₖ₋₂, cₖ₋₂, sₖ₋₁, cₖ₋₁, ζbarₖ, γₖ, βₖ.
      if iter ≥ 2
        sₖ₋₂ = sₖ₋₁
        cₖ₋₂ = cₖ₋₁
      end
      sₖ₋₁  = sₖ
      cₖ₋₁  = cₖ
      ζbarₖ = ζbarₖ₊₁
      γₖ    = γₖ₊₁
      βₖ    = βₖ₊₁

      # Update stopping criterion.
      iter == 1 && (κ = atol + rtol * AᴴrNorm)
      user_requested_exit = callback(solver) :: Bool
      solved = rNorm ≤ ε
      inconsistent = !solved && AᴴrNorm ≤ κ
      tired = iter ≥ itmax
      timer = time_ns() - start_time
      overtimed = timer > timemax_ns
      kdisplay(iter, verbose) && @printf(iostream, "%5d  %7.1e  %8.1e  %.2fs\n", iter, rNorm, AᴴrNorm, start_time |> ktimer)
    end
    (verbose > 0) && @printf(iostream, "\n")

    # Termination status
    tired               && (status = "maximum number of iterations exceeded")
    solved              && (status = "solution good enough given atol and rtol")
    user_requested_exit && (status = "user-requested exit")
    overtimed           && (status = "time limit exceeded")

    # Update x
    warm_start && kaxpy!(n, one(FC), Δx, x)
    solver.warm_start = false

    # Update stats
    stats.niter = iter
    stats.solved = solved
    stats.inconsistent = inconsistent
    stats.timer = start_time |> ktimer
    stats.status = status
    return solver
  end
end
